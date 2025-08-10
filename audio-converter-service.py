#!/usr/bin/env python3
"""
🎵 YouTube Audio Converter Service
=================================
FastAPI service that extracts audio from YouTube videos using yt-dlp,
processes with ffmpeg, and uploads to Supabase Storage.

Deploy this to Railway, Render, Fly.io, or any Python hosting service.
"""

import os
import tempfile
import subprocess
import uuid
import asyncio
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
import uvicorn

# Initialize FastAPI
app = FastAPI(
    title="YouTube Audio Converter",
    description="Convert YouTube videos to audio and upload to storage",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment variables are required")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

class ConversionRequest(BaseModel):
    videoUrl: str
    videoId: Optional[str] = None
    userId: str
    format: str = "mp3"
    quality: str = "high"

class ConversionResponse(BaseModel):
    success: bool
    audioUrl: Optional[str] = None
    duration: Optional[float] = None
    fileSize: Optional[int] = None
    error: Optional[str] = None

@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "YouTube Audio Converter",
        "version": "1.0.0",
        "dependencies": {
            "yt-dlp": check_command("yt-dlp"),
            "ffmpeg": check_command("ffmpeg"),
            "supabase": bool(SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY)
        }
    }

@app.post("/convert", response_model=ConversionResponse)
async def convert_video(request: ConversionRequest, background_tasks: BackgroundTasks):
    """
    Convert YouTube video to audio and upload to Supabase Storage
    """
    try:
        print(f"🎵 Starting conversion for: {request.videoUrl}")
        
        # Extract video ID if not provided
        video_id = request.videoId or extract_video_id(request.videoUrl)
        if not video_id:
            raise HTTPException(400, "Could not extract video ID from URL")

        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix=f"ytdl_{video_id}_")
        temp_path = Path(temp_dir)
        
        try:
            # Step 1: Download audio using yt-dlp
            print("📥 Downloading audio with yt-dlp...")
            audio_file = await download_audio(request.videoUrl, temp_path, request.format, request.quality)
            
            if not audio_file or not audio_file.exists():
                raise HTTPException(500, "Failed to download audio")

            print(f"✅ Audio downloaded: {audio_file}")

            # Step 2: Process with ffmpeg (optional optimization)
            print("🎛️ Processing audio with ffmpeg...")
            processed_file = await process_audio(audio_file, request.format)
            
            print(f"✅ Audio processed: {processed_file}")

            # Step 3: Upload to Supabase Storage
            print("☁️ Uploading to Supabase Storage...")
            upload_result = await upload_to_storage(processed_file, video_id, request.userId, request.format)
            
            if not upload_result["success"]:
                raise HTTPException(500, f"Upload failed: {upload_result['error']}")

            print(f"✅ Upload completed: {upload_result['publicUrl']}")

            # Step 4: Get file metadata
            file_size = processed_file.stat().st_size
            duration = await get_audio_duration(processed_file)

            # Schedule cleanup
            background_tasks.add_task(cleanup_temp_dir, temp_path)

            return ConversionResponse(
                success=True,
                audioUrl=upload_result["publicUrl"],
                duration=duration,
                fileSize=file_size
            )

        except Exception as e:
            # Cleanup on error
            background_tasks.add_task(cleanup_temp_dir, temp_path)
            raise e

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Conversion failed: {str(e)}")
        raise HTTPException(500, f"Conversion failed: {str(e)}")

async def download_audio(video_url: str, temp_dir: Path, format: str, quality: str) -> Path:
    """Download audio using yt-dlp"""
    
    # Quality settings
    quality_map = {
        "low": "worst[ext=m4a]/worst[ext=mp3]/worst",
        "medium": "best[height<=720][ext=m4a]/best[height<=720][ext=mp3]/best[height<=720]",
        "high": "best[ext=m4a]/best[ext=mp3]/best"
    }
    
    output_template = str(temp_dir / "%(title)s.%(ext)s")
    
    # yt-dlp command
    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", format,
        "--audio-quality", "0",  # Best quality
        "--format", quality_map.get(quality, quality_map["high"]),
        "--output", output_template,
        "--no-playlist",
        "--no-warnings",
        video_url
    ]
    
    print(f"🔄 Running: {' '.join(cmd)}")
    
    # Run yt-dlp
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    stdout, stderr = await process.communicate()
    
    if process.returncode != 0:
        error_msg = stderr.decode() if stderr else "Unknown yt-dlp error"
        print(f"❌ yt-dlp error: {error_msg}")
        raise Exception(f"yt-dlp failed: {error_msg}")
    
    # Find the downloaded file
    audio_files = list(temp_dir.glob(f"*.{format}"))
    if not audio_files:
        raise Exception("No audio file found after yt-dlp conversion")
    
    return audio_files[0]

async def process_audio(input_file: Path, format: str) -> Path:
    """Process audio with ffmpeg for optimization"""
    
    output_file = input_file.parent / f"{input_file.stem}_processed.{format}"
    
    # ffmpeg command for audio processing
    cmd = [
        "ffmpeg",
        "-i", str(input_file),
        "-acodec", "libmp3lame" if format == "mp3" else "aac",
        "-ab", "128k",  # Bitrate
        "-ar", "44100",  # Sample rate
        "-ac", "2",  # Stereo
        "-af", "loudnorm",  # Audio normalization
        "-y",  # Overwrite output
        str(output_file)
    ]
    
    print(f"🔄 Running: {' '.join(cmd)}")
    
    # Run ffmpeg
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    stdout, stderr = await process.communicate()
    
    if process.returncode != 0:
        error_msg = stderr.decode() if stderr else "Unknown ffmpeg error"
        print(f"❌ ffmpeg error: {error_msg}")
        # Return original file if processing fails
        return input_file
    
    return output_file

async def upload_to_storage(file_path: Path, video_id: str, user_id: str, format: str) -> dict:
    """Upload file to Supabase Storage"""
    
    try:
        # Read file content
        with open(file_path, "rb") as f:
            file_data = f.read()
        
        # Generate storage path
        filename = f"{video_id}.{format}"
        storage_path = f"audio/{user_id}/{filename}"
        
        print(f"☁️ Uploading to: {storage_path}")
        
        # Upload to Supabase Storage
        result = supabase.storage.from_("processed-audio").upload(
            storage_path, 
            file_data,
            {
                "content-type": f"audio/{format}",
                "upsert": True
            }
        )
        
        if result.get("error"):
            return {"success": False, "error": result["error"]["message"]}
        
        # Get public URL
        public_url_result = supabase.storage.from_("processed-audio").get_public_url(storage_path)
        public_url = public_url_result.get("publicUrl")
        
        if not public_url:
            return {"success": False, "error": "Failed to get public URL"}
        
        return {"success": True, "publicUrl": public_url}
        
    except Exception as e:
        print(f"❌ Upload error: {str(e)}")
        return {"success": False, "error": str(e)}

async def get_audio_duration(file_path: Path) -> float:
    """Get audio duration using ffprobe"""
    
    try:
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(file_path)
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            duration_str = stdout.decode().strip()
            return float(duration_str) if duration_str else 0.0
        
    except Exception as e:
        print(f"⚠️ Could not get duration: {e}")
    
    return 0.0

def extract_video_id(url: str) -> Optional[str]:
    """Extract video ID from YouTube URL"""
    
    import re
    
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([^&\n?#]+)',
        r'youtube\.com/v/([^&\n?#]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None

def check_command(command: str) -> bool:
    """Check if a command is available"""
    try:
        subprocess.run([command, "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

async def cleanup_temp_dir(temp_dir: Path):
    """Clean up temporary directory"""
    try:
        import shutil
        shutil.rmtree(temp_dir)
        print(f"🗑️ Cleaned up: {temp_dir}")
    except Exception as e:
        print(f"⚠️ Cleanup failed: {e}")

if __name__ == "__main__":
    # For local development
    uvicorn.run(
        "audio-converter-service:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=True
    ) 