FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Instal dependensi Python
RUN pip install --no-cache-dir -r requirements.txt

# Salin sisa kode aplikasi
COPY . .

# Jalankan aplikasi menggunakan Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "app:app"]