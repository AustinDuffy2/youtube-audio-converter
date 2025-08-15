#!/bin/bash

# 🚂 Railway Deployment Script for Updated YouTube Audio Converter
# ================================================================
# This script deploys the updated service with /extract-captions endpoint
# and Whisper API integration for real audio transcription.

echo "🚂 Deploying Updated YouTube Audio Converter to Railway..."
echo "📦 Service includes:"
echo "   - /convert endpoint (original audio conversion)"
echo "   - /extract-captions endpoint (NEW - Whisper transcription)"
echo "   - yt-dlp for video audio extraction"
echo "   - OpenAI Whisper API integration"
echo ""

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found. Install it first:"
    echo "   npm install -g @railway/cli"
    echo "   or visit: https://railway.app/cli"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "audio-converter-service.py" ]; then
    echo "❌ audio-converter-service.py not found."
    echo "   Run this script from the services/youtube-audio-converter/ directory"
    exit 1
fi

echo "✅ Railway CLI found"
echo "✅ Service files found"
echo ""

# Login to Railway (if not already logged in)
echo "🔐 Checking Railway authentication..."
if ! railway whoami &> /dev/null; then
    echo "🔐 Please login to Railway:"
    railway login
fi

echo "✅ Railway authentication verified"
echo ""

# Deploy the service
echo "🚀 Deploying to Railway..."
echo "📋 Required environment variables:"
echo "   - SUPABASE_URL (for storing transcriptions)"
echo "   - SUPABASE_SERVICE_ROLE_KEY (for database access)"
echo "   - OPENAI_API_KEY (for Whisper transcription) ⭐ NEW"
echo "   - TEMP_DIR (optional, defaults to /tmp)"
echo ""

# Check if project is linked
if [ ! -f "railway.json" ] && [ ! -d ".railway" ]; then
    echo "🔗 No Railway project linked. Please link or create a project:"
    echo "   railway link    (to link existing project)"
    echo "   railway login   (to create new project)"
    echo ""
    read -p "Press Enter to continue once project is linked..."
fi

# Deploy
echo "🚀 Starting deployment..."
railway up

echo ""
echo "✅ Deployment complete!"
echo ""
echo "🔧 Next steps:"
echo "1. Set environment variables in Railway dashboard:"
echo "   - OPENAI_API_KEY (your OpenAI API key for Whisper)"
echo "   - SUPABASE_URL (your Supabase project URL)"
echo "   - SUPABASE_SERVICE_ROLE_KEY (your Supabase service role key)"
echo ""
echo "2. Test the endpoints:"
echo "   GET  /        - Health check"
echo "   POST /convert - Original audio conversion"
echo "   POST /extract-captions - NEW: Whisper transcription"
echo ""
echo "3. Update your app's service URL to point to the new Railway deployment"
echo ""
echo "📋 Example /extract-captions request:"
echo '   {'
echo '     "video_url": "https://www.youtube.com/watch?v=VIDEO_ID",'
echo '     "language": "en",'
echo '     "fallback_to_whisper": true'
echo '   }'
echo ""
echo "🎉 Your service now supports real audio extraction and Whisper transcription!"
