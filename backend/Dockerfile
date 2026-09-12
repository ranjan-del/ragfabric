# Backend image for the RagFabric (FastAPI).
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source.
COPY . .

EXPOSE 8000

# Tables are created automatically on startup (see app.main.lifespan).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
