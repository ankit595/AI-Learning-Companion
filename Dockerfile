# =============================================================================
# Dockerfile — shared base image for both the FastAPI backend and the
# Streamlit frontend. Which process runs is decided by docker-compose's
# `command:` override for each service.
# =============================================================================

FROM python:3.11-slim

WORKDIR /app

# System deps needed by some document loaders (unstructured, pptx parsing, etc.)
# ca-certificates: fixes SSL verification failures when calling OpenAI API from container
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libmagic1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better layer caching on rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Directories that will be volume-mounted for persistence — created here too
# so the app doesn't fail if the volume is empty on first run.
RUN mkdir -p chroma_db notes data

EXPOSE 8000 8501
