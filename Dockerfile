# Gunakan base image Python yang slim
FROM python:3.11-slim

# Tetapkan direktori kerja
WORKDIR /app

# =================================================================
# PERBAIKAN: Menggunakan nama paket 'libgl1' yang lebih modern
# =================================================================
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Salin file requirements terlebih dahulu untuk caching
COPY requirements.txt .

# Instal dependensi Python
RUN pip install --no-cache-dir -r requirements.txt

# Salin sisa kode aplikasi
COPY . .

# Jalankan aplikasi menggunakan Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "app:app"]