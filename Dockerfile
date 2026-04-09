FROM python:3.11-slim-bullseye

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DEBIAN_FRONTEND noninteractive

# Install system dependencies with stable mirrors and retries
RUN sed -i 's/deb.debian.org/ftp.us.debian.org/g' /etc/apt/sources.list && \
    apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    gcc \
    python3-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install dependencies
# We use --no-cache-dir to keep the image small
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Ensure directories exist
RUN mkdir -p static/uploads static/processed stubs

# Render uses the PORT environment variable
ENV PORT 10000
EXPOSE 10000

# Start the application using gunicorn
# Increased timeout for AI processing tasks
CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 600
