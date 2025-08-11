FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    wget \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && update-ca-certificates

# Install latest yt-dlp (force update to bypass YouTube blocks)
RUN pip install --no-cache-dir --upgrade yt-dlp

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements-converter.txt .
RUN pip install --no-cache-dir -r requirements-converter.txt

# Copy application code
COPY audio-converter-service.py .

# Expose port
EXPOSE 8000

# Set environment variables
ENV PORT=8000
ENV PYTHONHTTPSVERIFY=0
ENV SSL_VERIFY=false
ENV CURL_CA_BUNDLE=""
ENV REQUESTS_CA_BUNDLE=""

# Run the application
CMD ["sh", "-c", "uvicorn audio-converter-service:app --host 0.0.0.0 --port $PORT"] 