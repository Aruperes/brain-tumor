FROM python:3.11-slim

WORKDIR /app

# Install dependencies untuk OpenCV, TensorFlow, dsb
RUN apt-get update && apt-get install -y \
    python3-dev \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Jalankan langsung app.py
CMD ["python", "app.py"]
