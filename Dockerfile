FROM python:3.10-slim

# OS dependencies
RUN apt-get update && apt-get install -y libsndfile1 && rm -rf /var/lib/apt/lists/*

# Create app dir
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your FastAPI app
COPY . .

# Expose port
EXPOSE 10000

# Run server
CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=10000"]
