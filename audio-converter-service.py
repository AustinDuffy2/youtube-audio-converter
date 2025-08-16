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
import base64
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
import ssl
import urllib3

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
import uvicorn
import yt_dlp
import requests

# Disable SSL warnings and verification
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

# Set SSL bypass environment variables
os.environ['PYTHONHTTPSVERIFY'] = '0'
os.environ['SSL_VERIFY'] = 'false'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
TEMP_DIR = os.environ.get("TEMP_DIR", "/tmp")

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

class CaptionRequest(BaseModel):
    video_url: str
    language: str = 'en'
    fallback_to_whisper: bool = True

class CaptionResponse(BaseModel):
    success: bool
    captions: Optional[List[Dict]] = None
    transcription: Optional[Dict] = None
    metadata: Optional[Dict] = None
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

async def download_audio_working(video_url: str, temp_dir: Path, format: str) -> Path:
    """Download audio using WORKING yt-dlp configuration that bypasses restrictions"""
    
    output_template = str(temp_dir / "%(title)s.%(ext)s")
    
    print("🚀 Using WORKING yt-dlp configuration...")
    
    # This is the EXACT configuration that works for other apps
    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", format,
        "--audio-quality", "0",
        "--format", "bestaudio/best",
        "--output", output_template,
        "--no-playlist",
        "--no-warnings",
        "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "--add-header", "Accept-Language:en-US,en;q=0.9",
        "--add-header", "Accept:text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "--add-header", "Accept-Encoding:gzip, deflate, br",
        "--add-header", "DNT:1",
        "--add-header", "Connection:keep-alive",
        "--add-header", "Upgrade-Insecure-Requests:1",
        "--extractor-args", "youtube:player_client=web",
        "--no-check-certificate",
        "--http-chunk-size", "10M",
        "--retries", "10",
        "--fragment-retries", "10",
        "--skip-unavailable-fragments",
        "--keep-fragments",
        video_url
    ]
    
    print(f"🔄 Running WORKING yt-dlp command...")
    
    # Run yt-dlp with SSL bypass environment
    env = os.environ.copy()
    env.update({
        'PYTHONHTTPSVERIFY': '0',
        'SSL_VERIFY': 'false',
        'CURL_CA_BUNDLE': '',
        'REQUESTS_CA_BUNDLE': '',
        'NODE_TLS_REJECT_UNAUTHORIZED': '0'
    })
    
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env
    )
    
    stdout, stderr = await process.communicate()
    
    if process.returncode == 0:
        # Success! Find the downloaded file
        audio_files = list(temp_dir.glob(f"*.{format}"))
        if audio_files:
            print(f"✅ SUCCESS with working yt-dlp config!")
            return audio_files[0]
    
    # If that fails, try WITHOUT cookies
    print("🔄 Retrying without cookies...")
    
    cmd_no_cookies = [
        "yt-dlp",
        "--extract-audio", 
        "--audio-format", format,
        "--audio-quality", "0",
        "--format", "bestaudio/best",
        "--output", output_template,
        "--no-playlist",
        "--no-warnings",
        "--user-agent", "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15",
        "--extractor-args", "youtube:player_client=ios",
        "--geo-bypass",
        "--socket-timeout", "30",
        video_url
    ]
    
    process2 = await asyncio.create_subprocess_exec(
        *cmd_no_cookies,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env
    )
    
    stdout2, stderr2 = await process2.communicate()
    
    if process2.returncode == 0:
        audio_files = list(temp_dir.glob(f"*.{format}"))
        if audio_files:
            print(f"✅ SUCCESS with iOS client!")
            return audio_files[0]
    
    # If still failing, show the actual error
    error_msg = stderr2.decode() if stderr2 else stderr.decode() if stderr else "Unknown error"
    print(f"❌ Both methods failed: {error_msg}")
    raise Exception(f"yt-dlp failed: {error_msg}")

async def download_audio(video_url: str, temp_dir: Path, format: str, quality: str) -> Path:
    """Download audio with Cobalt API first, yt-dlp as fallback"""
    
    # Try WORKING yt-dlp configuration first
    try:
        return await download_audio_working(video_url, temp_dir, format)
    except Exception as e:
        print(f"⚠️ Working config failed, trying fallback strategies: {str(e)}")
    
    # Fallback to yt-dlp (original code)
    # Quality settings
    quality_map = {
        "low": "worst[ext=m4a]/worst[ext=mp3]/worst",
        "medium": "best[height<=720][ext=m4a]/best[height<=720][ext=mp3]/best[height<=720]",
        "high": "best[ext=m4a]/best[ext=mp3]/best"
    }
    
    output_template = str(temp_dir / "%(title)s.%(ext)s")
    
    # Multiple extraction strategies to try
    strategies = [
        {
            "name": "iOS Mobile",
            "user_agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
            "extractor_args": "youtube:player_client=ios"
        },
        {
            "name": "Android Mobile", 
            "user_agent": "Mozilla/5.0 (Linux; Android 13; SM-G991B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
            "extractor_args": "youtube:player_client=android"
        },
        {
            "name": "Web Embedded",
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "extractor_args": "youtube:player_client=web_embedded"
        }
    ]
    
    # Try multiple strategies to bypass bot detection
    last_error = None
    
    for i, strategy in enumerate(strategies):
        try:
            print(f"🔄 Attempt {i+1}: Using {strategy['name']} strategy")
            
            cmd = [
                "yt-dlp",
                "--extract-audio",
                "--audio-format", format,
                "--audio-quality", "0",
                "--format", "bestaudio[ext=m4a]/bestaudio[ext=mp3]/bestaudio",
                "--output", output_template,
                "--no-playlist",
                "--no-warnings",
                "--user-agent", strategy["user_agent"],
                "--add-header", "Accept-Language:en-US,en;q=0.9",
                "--add-header", "Accept:*/*",
                "--add-header", "Origin:https://www.youtube.com",
                "--add-header", "Referer:https://www.youtube.com/",
                "--extractor-args", strategy["extractor_args"],
                "--no-check-certificate",
                "--ignore-config", 
                "--ignore-errors",
                "--socket-timeout", "30",
                "--retries", "2",
                "--legacy-server-connect",
                "--prefer-insecure",
                video_url
            ]
            
            print(f"🔄 Running: yt-dlp with {strategy['name']}")
            
            # Run yt-dlp with SSL bypass environment
            env = os.environ.copy()
            env.update({
                'PYTHONHTTPSVERIFY': '0',
                'SSL_VERIFY': 'false',
                'CURL_CA_BUNDLE': '',
                'REQUESTS_CA_BUNDLE': '',
                'NODE_TLS_REJECT_UNAUTHORIZED': '0'
            })
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                # Success! Check for downloaded file
                audio_files = list(temp_dir.glob(f"*.{format}"))
                if audio_files:
                    print(f"✅ Success with {strategy['name']} strategy!")
                    break
                else:
                    print(f"⚠️ {strategy['name']} completed but no file found")
                    continue
            else:
                error_msg = stderr.decode() if stderr else "Unknown error"
                print(f"❌ {strategy['name']} failed: {error_msg}")
                last_error = error_msg
                continue
                
        except Exception as e:
            print(f"❌ {strategy['name']} exception: {str(e)}")
            last_error = str(e)
            continue
    
    # Check if we got any files
    audio_files = list(temp_dir.glob(f"*.{format}"))
    if not audio_files:
        raise Exception(f"All extraction strategies failed. Last error: {last_error}")
    
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

async def extract_audio_and_transcribe_with_whisper(video_url: str, language: str = 'en') -> Dict:
    """Extract audio and transcribe with Whisper API - ENHANCED BYPASS VERSION"""
    try:
        logger.info(f"🎵 Starting ENHANCED audio extraction for Whisper: {video_url}")
        
        # Extract video ID
        video_id = extract_video_id(video_url)
        if not video_id:
            return {
                'success': False,
                'error': 'Could not extract video ID from URL'
            }
        
        logger.info(f"🎯 Video ID: {video_id}")
        logger.info(f"🚀 Using ADVANCED YouTube bypass techniques for audio extraction...")
        
        # ADVANCED YOUTUBE BOT DETECTION BYPASS WITH PROXY TUNNELING
        import random
        import requests
        
        # 🥷 ULTIMATE YOUTUBE EVASION SYSTEM - DISGUISED TERMINAL APPROACH
        def generate_stealth_browser_session():
            """Generate ultra-realistic browser session with complete fingerprinting evasion"""
            
            # Latest Chrome versions (updated frequently to stay current)
            chrome_versions = [
                '121.0.6167.85', '121.0.6167.139', '121.0.6167.184',
                '122.0.6261.57', '122.0.6261.94', '122.0.6261.111',
                '123.0.6312.58', '123.0.6312.86', '123.0.6312.105'
            ]
            
            # Realistic OS fingerprints with proper version distributions
            os_fingerprints = [
                {
                    'platform': 'Windows NT 10.0; Win64; x64',
                    'accept_language': 'en-US,en;q=0.9',
                    'timezone': 'America/New_York'
                },
                {
                    'platform': 'Macintosh; Intel Mac OS X 10_15_7', 
                    'accept_language': 'en-US,en;q=0.9',
                    'timezone': 'America/Los_Angeles'
                },
                {
                    'platform': 'X11; Linux x86_64',
                    'accept_language': 'en-US,en;q=0.9',
                    'timezone': 'America/Chicago'
                }
            ]
            
            chrome_version = random.choice(chrome_versions)
            os_config = random.choice(os_fingerprints)
            
            # Generate realistic session fingerprint
            session_id = f"{random.randint(100000, 999999)}-{random.randint(1000, 9999)}"
            
            user_agent = f'Mozilla/5.0 ({os_config["platform"]}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{chrome_version} Safari/537.36'
            
            return {
                'user_agent': user_agent,
                'chrome_version': chrome_version,
                'platform': os_config['platform'],
                'accept_language': os_config['accept_language'],
                'timezone': os_config['timezone'],
                'session_id': session_id,
                'viewport': f"{random.choice([1920, 1366, 1536, 1440])},{random.choice([1080, 768, 864, 900])}",
                'screen_depth': random.choice([24, 32]),
                'connection_type': random.choice(['4g', 'wifi', 'ethernet'])
            }
        
        # ADVANCED PROXY AND TUNNEL INTEGRATION
        async def get_bypass_infrastructure():
            """Get working proxies and tunnel endpoints for IP rotation"""
            try:
                # Try to import our tunnel proxy service
                from fake_useragent import UserAgent
                
                # Use fake-useragent for even more realistic browser simulation
                ua = UserAgent()
                realistic_agents = [
                    ua.chrome,
                    ua.firefox, 
                    ua.safari,
                    ua.edge
                ]
                
                # Get some working free proxies for testing
                # In production, you'd use premium residential proxies
                test_proxies = [
                    'http://103.149.162.194:80',
                    'http://103.145.113.78:80', 
                    'http://188.166.56.246:80',
                    'http://157.230.103.91:33554',
                    'http://165.154.243.154:80'
                ]
                
                # Test which proxies are working
                working_proxies = []
                for proxy in test_proxies[:3]:  # Test first 3
                    try:
                        response = requests.get('https://httpbin.org/ip', 
                                              proxies={'http': proxy, 'https': proxy}, 
                                              timeout=3)
                        if response.status_code == 200:
                            working_proxies.append(proxy)
                            logger.info(f"✅ Working proxy: {proxy}")
                    except:
                        pass
                
                return {
                    'proxies': working_proxies,
                    'user_agents': realistic_agents[:4]  # Get 4 realistic UAs
                }
                
            except Exception as e:
                logger.warning(f"⚠️ Advanced bypass setup failed: {e}")
                return {'proxies': [], 'user_agents': []}
        
        # Get bypass infrastructure
        try:
            import asyncio
            if hasattr(asyncio, 'run'):
                bypass_info = asyncio.run(get_bypass_infrastructure())
            else:
                bypass_info = {'proxies': [], 'user_agents': []}
        except:
            bypass_info = {'proxies': [], 'user_agents': []}
        
        fingerprint = generate_realistic_chrome_fingerprint()
        proxy_list = bypass_info.get('proxies', [])
        enhanced_user_agents = bypass_info.get('user_agents', [fingerprint['user_agent']])
        
        logger.info(f"🕵️ Generated Chrome fingerprint: {fingerprint['user_agent'][:60]}...")
        logger.info(f"🌐 Working proxies found: {len(proxy_list)}")
        logger.info(f"🎭 Enhanced user agents: {len(enhanced_user_agents)}")
        
        # Use enhanced user agent if available
        if enhanced_user_agents:
            fingerprint['user_agent'] = random.choice(enhanced_user_agents)
        
        # REALISTIC CHROME BROWSER SIMULATION
        def create_realistic_chrome_headers(fingerprint):
            """Create realistic Chrome browser headers with proper fingerprinting"""
            return {
                'User-Agent': fingerprint['user_agent'],
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1',
                'Sec-CH-UA': f'"Not_A Brand";v="8", "Chromium";v="{fingerprint["chrome_version"].split(".")[0]}", "Google Chrome";v="{fingerprint["chrome_version"].split(".")[0]}"',
                'Sec-CH-UA-Mobile': '?0',
                'Sec-CH-UA-Platform': f'"{fingerprint["platform"].split(";")[0] if "Windows" in fingerprint["platform"] else "macOS"}"',
                'Cache-Control': 'max-age=0',
                'Referer': 'https://www.google.com/',
                'Origin': 'https://www.youtube.com',
            }
        
        chrome_headers = create_realistic_chrome_headers(fingerprint)
        
        # PROXY CONFIGURATION
        def get_proxy_config():
            """Get proxy configuration for requests"""
            if proxy_list:
                proxy = random.choice(proxy_list)
                logger.info(f"🌐 Using proxy: {proxy}")
                return {
                    'http': proxy,
                    'https': proxy
                }
            return None
        
        proxy_config = get_proxy_config()
        
        # Advanced bypass options with realistic Chrome simulation
        base_opts = {
            'format': 'bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best',
            'extractaudio': True,
            'audioformat': 'mp3',
            'outtmpl': f'{TEMP_DIR}/%(id)s.%(ext)s',
            'quiet': False,  # Enable logging to see what's happening
            'no_warnings': False,
            'socket_timeout': 45,
            'retries': 10,
            'fragment_retries': 10,
            'ignoreerrors': False,
            
            # REALISTIC CHROME BROWSER SIMULATION
            'user_agent': fingerprint['user_agent'],
            'http_headers': chrome_headers,
            
            # BYPASS STRATEGY 3: Geo and Network Simulation  
            'geo_bypass': True,
            'geo_bypass_country': random.choice(['US', 'CA', 'GB', 'AU', 'DE']),
            
            # PROXY SUPPORT (if available)
            **(({'proxy': random.choice(proxy_list)} if proxy_list else {})),
            
            # BYPASS STRATEGY 4: Rate Limiting
            'sleep_interval': 2,
            'max_sleep_interval': 5,
            'sleep_interval_requests': 1,
            
            # BYPASS STRATEGY 5: Format Selection (avoid detection)
            'prefer_free_formats': True,
            'youtube_include_dash_manifest': False,
            
            # BYPASS STRATEGY 6: Extractor Arguments
            'extractor_args': {
                'youtube': {
                    'skip': ['dash', 'hls'],
                    'player_skip': ['configs'],
                    'comment_sort': ['top'],
                    'max_comments': [0],
                }
            },
            
            # BYPASS STRATEGY 7: Additional Options
            'no_check_certificate': True,
            'prefer_insecure': True,
            'call_home': False,
            'no_color': True,
        }
        
        # 🥷 ULTIMATE DISGUISED TERMINAL BYPASS STRATEGIES
        # Generate stealth session for this request
        stealth_session = generate_stealth_browser_session()
        logger.info(f"🎭 Generated stealth session: {stealth_session['session_id']}")
        logger.info(f"🖥️ Disguised as: {stealth_session['platform']} - {stealth_session['chrome_version']}")
        
        # Advanced headers that mimic real browser behavior
        stealth_headers = {
            'User-Agent': stealth_session['user_agent'],
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': stealth_session['accept_language'],
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
            'sec-ch-ua': f'"Chromium";v="{stealth_session["chrome_version"].split(".")[0]}", "Google Chrome";v="{stealth_session["chrome_version"].split(".")[0]}", "Not=A?Brand";v="99"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': f'"{stealth_session["platform"].split(";")[0]}"'
        }
        
        strategies = [
            # 🎯 Strategy 1: STEALTH DESKTOP - Complete browser mimicry
            {
                'format': 'worstaudio[filesize<30M]/worstaudio/worst',
                'outtmpl': f'{TEMP_DIR}/%(id)s.%(ext)s',
                'quiet': True,
                'no_warnings': True,
                'extractaudio': True,
                'audioformat': 'mp3',
                'http_headers': stealth_headers,
                'extractor_args': {
                    'youtube': {
                        'player_client': ['web', 'mweb'],
                        'skip': ['dash', 'hls'],
                    }
                },
                'socket_timeout': 45,
                'retries': 3,
                'fragment_retries': 5,
                'sleep_interval_requests': random.uniform(1, 3),
                'geo_bypass': True,
                'no_check_certificate': True,
            },
            
            # 🎯 Strategy 2: MOBILE STEALTH - iOS Safari mimicry
            {
                'format': 'worstaudio[filesize<25M]/worstaudio/worst',
                'outtmpl': f'{TEMP_DIR}/%(id)s.%(ext)s',
                'quiet': True,
                'no_warnings': True,
                'extractaudio': True,
                'audioformat': 'mp3',
                'http_headers': {
                    'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_2_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate, br',
                },
                'extractor_args': {
                    'youtube': {
                        'player_client': ['ios', 'mweb'],
                        'skip': ['dash'],
                    }
                },
                'socket_timeout': 30,
                'retries': 2,
            },
            
            # 🎯 Strategy 3: ANDROID STEALTH - Native app mimicry
            {
                'format': 'worstaudio[filesize<20M]/worst',
                'outtmpl': f'{TEMP_DIR}/%(id)s.%(ext)s',
                'quiet': True,
                'no_warnings': True,
                'extractaudio': True,
                'audioformat': 'mp3',
                'http_headers': {
                    'User-Agent': f'com.google.android.youtube/{random.choice(["19.09.37", "19.10.35", "19.11.43"])} (Linux; U; Android {random.choice(["13", "14"]); SM-G998B) gzip',
                    'X-YouTube-Client-Name': '3',
                    'X-YouTube-Client-Version': '19.09.37',
                },
                'extractor_args': {
                    'youtube': {
                        'player_client': ['android', 'android_music'],
                        'skip': ['dash'],
                    }
                },
                'socket_timeout': 25,
                'retries': 1,
            },
            
            # 🎯 Strategy 4: EMBED STEALTH - Bypass main page detection
            {
                'format': 'worst[ext=mp4]/worst',
                'outtmpl': f'{TEMP_DIR}/%(id)s.%(ext)s',
                'quiet': True,
                'no_warnings': True,
                'extractaudio': True,
                'audioformat': 'mp3',
                'http_headers': stealth_headers,
                'extractor_args': {
                    'youtube': {
                        'player_client': ['web'],
                        'skip': ['webpage', 'dash'],
                    }
                },
                'no_check_certificate': True,
                'ignoreerrors': True,
            },
            
            # 🎯 Strategy 5: ULTIMATE FALLBACK - Minimal detection footprint
            {
                'format': 'worst',
                'outtmpl': f'{TEMP_DIR}/%(id)s.%(ext)s',
                'quiet': True,
                'no_warnings': True,
                'extractaudio': True,
                'audioformat': 'mp3',
                'http_headers': {
                    'User-Agent': 'yt-dlp/2023.12.30',
                },
                'extractor_args': {
                    'youtube': {
                        'player_client': ['web'],
                    }
                },
            }
        ]
        
        info = None
        last_error = None
        
        for i, strategy in enumerate(strategies, 1):
            try:
                strategy_name = [
                    "STEALTH DESKTOP", "MOBILE STEALTH", "ANDROID STEALTH", 
                    "EMBED STEALTH", "ULTIMATE FALLBACK"
                ][i-1] if i <= 5 else f"STRATEGY {i}"
                
                logger.info(f"🎯 Attempting {strategy_name} (#{i}/{len(strategies)})...")
                logger.info(f"🥷 Using disguised terminal with stealth fingerprinting...")
                
                # Add intelligent delay based on previous failures
                if i > 1:
                    delay = random.uniform(2, 5) * i  # Increasing delay for later attempts
                    logger.info(f"⏳ Strategic delay: {delay:.1f}s to avoid detection...")
                    import time
                    time.sleep(delay)
                
                # Execute with stealth configuration
                with yt_dlp.YoutubeDL(strategy) as ydl:
                    logger.info("🚀 Initiating stealth audio extraction...")
                    info = ydl.extract_info(video_url, download=True)
                    if info:
                        logger.info(f"🎉 SUCCESS! {strategy_name} bypassed YouTube blocking!")
                        logger.info("🎵 Real audio extraction successful - proceeding to Whisper transcription")
                        break
                        
            except Exception as e:
                last_error = str(e)
                error_preview = str(e)[:150] + "..." if len(str(e)) > 150 else str(e)
                logger.warning(f"❌ {strategy_name} blocked: {error_preview}")
                
                # Check if it's a rate limiting error
                if "429" in str(e) or "rate limit" in str(e).lower():
                    logger.info("🛑 Rate limiting detected - extending delay for next attempt")
                    time.sleep(random.uniform(5, 10))
                elif "blocked" in str(e).lower() or "forbidden" in str(e).lower():
                    logger.info("🚫 IP/fingerprint detected - rotating to next strategy")
                    
                if i < len(strategies):
                    logger.info(f"🔄 Rotating to next stealth strategy...")
                continue
        
        if not info:
            logger.error(f"🚫 All audio extraction strategies failed: {last_error}")
            logger.info("🔄 YouTube is blocking audio extraction - this is expected behavior")
            logger.info("💡 The system will gracefully fall back to existing captions")
            return {
                'success': False,
                'error': f'YouTube blocked audio extraction: {last_error}',
                'fallback_reason': 'youtube_blocking',
                'video_id': video_id if 'video_id' in locals() else None
            }
            
        video_id = info.get('id')
        title = info.get('title', 'Unknown Video')
        duration = info.get('duration', 0)
        
        logger.info(f"🎬 Successfully extracted video info: {title} ({duration}s)")
        
        # Find audio file
        audio_path = None
        for ext in ['mp3', 'm4a', 'webm', 'opus']:
            test_path = f"{TEMP_DIR}/{video_id}.{ext}"
            if os.path.exists(test_path):
                audio_path = test_path
                logger.info(f"🎵 Found audio file: {audio_path}")
                break
        
        if not audio_path:
            logger.error("❌ Audio file not found after successful extraction")
            return {
                'success': False,
                'error': 'Audio file not found after extraction - this may indicate YouTube blocking',
                'fallback_reason': 'file_not_found',
                'video_id': video_id
            }
        
        # Convert to base64
        with open(audio_path, 'rb') as f:
            audio_data = f.read()
        
        audio_size_mb = len(audio_data) / (1024 * 1024)
        logger.info(f"🎵 Audio file size: {audio_size_mb:.2f} MB")
        
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        # Clean up audio file
        try:
            os.remove(audio_path)
            logger.info(f"🗑️ Cleaned up audio file: {audio_path}")
        except Exception as e:
            logger.warning(f"⚠️ Could not clean up audio file: {e}")
        
        # 🎤 REAL WHISPER TRANSCRIPTION - This is the actual audio-to-text conversion!
        logger.info("🚀 SUCCESS: Starting REAL Whisper API transcription of extracted audio!")
        logger.info("🎯 This will generate a fresh transcription from the actual audio, not YouTube captions")
        
        whisper_result = await transcribe_with_whisper_api(
            audio_base64, 'audio/mp3', title, duration, language, video_id
        )
        
        if whisper_result.get('success'):
            logger.info("✅ REAL WHISPER TRANSCRIPTION COMPLETED SUCCESSFULLY!")
            logger.info(f"📝 Generated {len(whisper_result.get('transcription', {}).get('segments', []))} segments from real audio")
            # Mark this as a real transcription, not YouTube captions
            whisper_result['transcription_source'] = 'whisper_real_audio'
            whisper_result['is_real_transcription'] = True
        
        return whisper_result
            
    except Exception as e:
        logger.error(f"❌ Audio extraction failed: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

async def save_transcription_to_supabase(video_id: str, video_url: str, title: str, channel_name: str, duration: int, transcription_data: Dict, user_id: str = 'demo-user') -> bool:
    """Save transcription results to Supabase transcription_history table"""
    try:
        logger.info(f"💾 Saving transcription to Supabase for video: {video_id}")
        
        # Format captions for storage
        captions = []
        if transcription_data.get('segments'):
            captions = [
                {
                    'start': seg['start'],
                    'end': seg['end'],
                    'text': seg['text'],
                    'id': f"caption-{i}"
                }
                for i, seg in enumerate(transcription_data['segments'])
            ]
        
        # Insert into transcription_history table
        transcription_record = {
            'user_id': user_id,
            'video_id': video_id,
            'video_url': video_url,
            'video_title': title,
            'video_thumbnail': f'https://img.youtube.com/vi/{video_id}/hqdefault.jpg',
            'channel_name': channel_name,
            'duration': str(duration),
            'transcription_type': 'api',  # Whisper API transcription
            'captions': captions,
            'cost_charged': 0.02,  # Whisper API cost estimate
            'ai_confidence': 0.95,  # High confidence for Whisper
            'processing_time_ms': 30000  # Estimate
        }
        
        result = supabase.table('transcription_history').insert(transcription_record).execute()
        
        if result.data:
            logger.info(f"✅ Transcription saved to Supabase successfully")
            return True
        else:
            logger.error(f"❌ Failed to save transcription to Supabase")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error saving transcription to Supabase: {str(e)}")
        return False

async def transcribe_with_whisper_api(audio_base64: str, mime_type: str, title: str, duration: int, language: str, video_id: str) -> Dict:
    """Transcribe audio using OpenAI Whisper API"""
    if not OPENAI_API_KEY:
        return {
            'success': False,
            'error': 'OpenAI API key not configured'
        }
    
    try:
        # Decode and save audio
        audio_data = base64.b64decode(audio_base64)
        temp_path = f"{TEMP_DIR}/whisper_{uuid.uuid4().hex}.mp3"
        
        with open(temp_path, 'wb') as f:
            f.write(audio_data)
        
        # Call Whisper API
        with open(temp_path, 'rb') as audio_file:
            response = requests.post(
                'https://api.openai.com/v1/audio/transcriptions',
                headers={
                    'Authorization': f'Bearer {OPENAI_API_KEY}'
                },
                files={
                    'file': audio_file
                },
                data={
                    'model': 'whisper-1',
                    'language': language,
                    'response_format': 'verbose_json'
                },
                timeout=300
            )
        
        # Clean up temp file
        try:
            os.remove(temp_path)
        except:
            pass
        
        if response.status_code == 200:
            whisper_result = response.json()
            
            # Convert to our format
            segments = []
            if 'segments' in whisper_result:
                segments = [
                    {
                        'start': seg.get('start', 0),
                        'end': seg.get('end', 0),
                        'text': seg.get('text', '').strip()
                    }
                    for seg in whisper_result['segments']
                    if seg.get('text', '').strip()
                ]
            
            transcription_data = {
                'text': whisper_result.get('text', ''),
                'language': whisper_result.get('language', language),
                'segments': segments
            }
            
            # 🔥 SAVE TO SUPABASE - This is the key missing piece!
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            await save_transcription_to_supabase(
                video_id=video_id,
                video_url=video_url,
                title=title,
                channel_name='Unknown Channel',  # Could be extracted from video metadata
                duration=duration,
                transcription_data=transcription_data,
                user_id='demo-user'
            )
            
            return {
                'success': True,
                'transcription': transcription_data,
                'metadata': {
                    'title': title,
                    'duration': duration,
                    'channelName': 'Unknown Channel',
                    'thumbnailUrl': f'https://img.youtube.com/vi/{video_id}/hqdefault.jpg'
                },
                'audioUrl': f'https://www.soundjay.com/misc/sounds/bell-ringing-05.mp3'  # Mock URL
            }
        else:
            return {
                'success': False,
                'error': f'Whisper API error: {response.status_code} - {response.text}'
            }
            
    except Exception as e:
        return {
            'success': False,
            'error': f'Transcription failed: {str(e)}'
        }

@app.post("/extract-captions", response_model=CaptionResponse)
async def extract_captions(request: CaptionRequest):
    """Extract captions with Whisper fallback"""
    try:
        logger.info(f"🎬 Processing caption request for: {request.video_url}")
        
        # Skip caption extraction and go directly to Whisper transcription
        # This ensures we always get real audio transcription instead of YouTube captions
        if request.fallback_to_whisper:
            logger.info("🎤 Using Whisper transcription for accurate audio-to-text conversion...")
            whisper_result = await extract_audio_and_transcribe_with_whisper(request.video_url, request.language)
            
            if whisper_result['success']:
                return CaptionResponse(
                    success=True,
                    transcription=whisper_result.get('transcription'),
                    metadata=whisper_result.get('metadata'),
                    captions=[]  # Empty since we're using transcription instead
                )
            else:
                return CaptionResponse(
                    success=False,
                    error=whisper_result.get('error')
                )
        
        # Fallback response if Whisper is not requested
        return CaptionResponse(
            success=False,
            error="Only Whisper transcription is supported in this version"
        )
        
    except Exception as e:
        logger.error(f"❌ Caption extraction failed: {str(e)}")
        return CaptionResponse(
            success=False,
            error=str(e)
        )

if __name__ == "__main__":
    # For local development
    uvicorn.run(
        "audio-converter-service:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=True
    ) 