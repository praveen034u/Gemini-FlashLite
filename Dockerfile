FROM python:3.11-slim

# Set environment variable for Python not to buffer output (helps in debugging)
ENV PYTHONUNBUFFERED=1

# Cloud Run sets $PORT dynamically, default to 8080 for local testing
ENV PORT 8080

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy application code including model files (.pkl, .pth)
COPY . .

# Let Cloud Run know what port to expose (optional, just for readability)
EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]