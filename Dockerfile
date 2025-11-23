FROM python:3.10-slim

# Soundfile kitabxanasının işləməsi üçün vacib olan sistem faylları
RUN apt-get update && apt-get install -y libsndfile1 && rm -rf /var/lib/apt/lists/*

# App qovluğunu yaradırıq
WORKDIR /app

# Kitabxanaları yükləyirik
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bütün faylları kopyalayırıq
COPY . .

# Render üçün portu açırıq
EXPOSE 10000

# Serveri 10000 portunda işə salırıq
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
