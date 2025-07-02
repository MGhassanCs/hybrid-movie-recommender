# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for pandas, matplotlib, seaborn, and surprise
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        libopenblas-dev \
        liblapack-dev \
        libatlas-base-dev \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Expose port for Streamlit (Hugging Face Spaces expects 7860)
EXPOSE 7860

# Default command: launch Streamlit app on the correct port
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"] 