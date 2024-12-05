# Base image: Use Python 3.9 slim (lightweight version of Python)
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the Streamlit app and all necessary files to the container
COPY ./weights/ /app/.cxas/weights
COPY interactive_cxas_app.py /app/

# Set environment variables
ENV CXAS_PATH=/app/

# Install system-level dependencies (libGL for OpenCV and other dependencies)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    libsm6 \
    libxext6 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install setuptools wheel && \
    pip install torch torchvision torchaudio && \
    pip install cxas==0.0.15 streamlit opencv-python-headless pillow numpy

# Expose the default port for Streamlit (8501)
EXPOSE 8501

# Command to run the Streamlit app on container startup
CMD ["streamlit", "run", "interactive_cxas_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
