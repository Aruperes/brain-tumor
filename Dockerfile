# Gunakan base image Python yang stabil
FROM python:3.11-slim

# Set working directory di dalam container
WORKDIR /app

# Instal dependensi sistem yang diperlukan oleh opencv-python
# apt-get update && apt-get install -y <paket>
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Salin file requirements terlebih dahulu untuk caching
COPY requirements.txt .

# Instal dependensi Python
RUN pip install --no-cache-dir -r requirements.txt

# Salin sisa kode aplikasi
COPY . .

# Perintah untuk menjalankan aplikasi (diambil dari Procfile)
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:$PORT"]