# Tahap 1: Gunakan image Python 3.11 yang ramping sebagai dasar
FROM python:3.11-slim

# Atur direktori kerja di dalam kontainer
WORKDIR /app

# Atur variabel lingkungan agar Python tidak melakukan buffering output
ENV PYTHONUNBUFFERED 1

# Salin file requirements.txt terlebih dahulu untuk memanfaatkan cache Docker
COPY requirements.txt .

# Instal dependensi menggunakan versi CPU-only dari torch untuk build yang lebih cepat
# Ini adalah langkah kunci untuk menghindari masalah path dan timeout
RUN pip install --no-cache-dir -r requirements.txt

# Salin sisa kode aplikasi ke dalam direktori kerja
COPY . .

# Beri tahu Docker port mana yang akan diekspos oleh aplikasi
# Railway akan secara otomatis menggunakan variabel PORT-nya di sini
EXPOSE 8080

# Perintah untuk menjalankan aplikasi saat kontainer dimulai
# Railway akan mengganti $PORT dengan port yang benar
CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 0 app:app
