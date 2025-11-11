FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# system deps for psycopg2/reportlab/pymupdf
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libc6-dev libpq-dev \
    libfreetype6-dev libjpeg62-turbo-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
WORKDIR /app
