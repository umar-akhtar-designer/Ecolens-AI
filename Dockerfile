# EcoLens AI — container image for the Streamlit application.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ECOLENS_DEVICE=cpu

WORKDIR /app

# System deps for Pillow/OpenCV runtime (keep image lean).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-create storage so first run never fails on permissions.
RUN mkdir -p storage/uploads storage/explanations

EXPOSE 8501

HEALTHCHECK CMD python -c "import streamlit" || exit 1

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
