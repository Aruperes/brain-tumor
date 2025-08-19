# ===================================================
# Tahap 1: Builder - Menginstal semua dependensi
# ===================================================
FROM python:3.11-slim AS builder

WORKDIR /app

# Buat virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Instal dependensi sistem yang dibutuhkan
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Instal dependensi Python ke dalam virtual environment
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ===================================================
# Tahap 2: Final - Image akhir yang bersih dan kecil
# ===================================================
FROM python:3.11-slim

WORKDIR /app

# Salin virtual environment yang sudah jadi dari tahap builder
COPY --from=builder /opt/venv /opt/venv

# Salin kode aplikasi Anda
COPY . .

# Aktifkan virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Instal dependensi sistem (tetap dibutuhkan saat runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Perintah untuk menjalankan aplikasi
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "app:app"]