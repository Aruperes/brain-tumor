# Tahap 1: Gunakan image Python 3.11 yang ramping sebagai dasar
FROM python:3.11-slim

# Atur direktori kerja di dalam kontainer
WORKDIR /app

# Atur variabel lingkungan agar Python tidak melakukan buffering output
ENV PYTHONUNBUFFERED 1

# Salin file requirements.txt terlebih dahulu untuk memanfaatkan cache Docker
COPY requirements.txt .

# Langkah 1: Instal torch & torchvision versi CPU-only secara terpisah untuk build yang lebih cepat dan stabil.
# Ini mencegah pip mencoba mengunduh versi CUDA yang besar.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision

# Langkah 2: Instal sisa dependensi dari requirements.txt
RUN pip install --no-cache-dir -r requirements.txt --no-deps ultralytics

# Salin sisa kode aplikasi ke dalam direktori kerja
COPY . .

# Beri tahu Docker port mana yang akan diekspos oleh aplikasi
# Railway akan secara otomatis menggunakan variabel PORT-nya di sini
EXPOSE 8080

# Perintah untuk menjalankan aplikasi saat kontainer dimulai
# Railway akan mengganti $PORT dengan port yang benar
CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 0 app:app
