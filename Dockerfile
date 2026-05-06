# Dockerfile
# Single container running both FastAPI and Streamlit

FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create jobs directory
RUN mkdir -p /app/jobs

# Expose ports
# 8000 = FastAPI
# 8501 = Streamlit
EXPOSE 8000 8501

# Run with supervisord
CMD ["supervisord","-c","supervisord.conf"]
