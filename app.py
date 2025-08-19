import os
import uuid
import numpy as np
import google.generativeai as genai
import tensorflow as tf
import cv2
import matplotlib
import markdown
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from ultralytics import YOLO
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from PIL import Image
from werkzeug.utils import secure_filename

# Konfigurasi awal
matplotlib.use("Agg")
load_dotenv()
app = Flask(__name__)

# Konfigurasi Kunci API dan Konstanta
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY tidak ditemukan di environment variables.")
genai.configure(api_key=GEMINI_API_KEY)

LABELS = ["Glioma Tumor", "Normal", "Meningioma Tumor", "Pituitary Tumor"]
IMAGE_SIZE = 224
HF_REPO_ID = "Revando/EfficientNet"

# ======== Pra-muat Semua Model Saat Aplikasi Dimulai ========
print("Memuat model, ini mungkin memakan waktu...")
try:
    CLASSIFICATION_MODELS = {
        "efficientnet": {
            "model": load_model(hf_hub_download(repo_id=HF_REPO_ID, filename="model/effnet.h5")),
            "last_conv": "top_conv",
        },
        "resnet": {
            "model": load_model(hf_hub_download(repo_id=HF_REPO_ID, filename="model/resnet.h5")),
            "last_conv": "conv5_block3_out",
        },
        "vgg": {
            "model": load_model(hf_hub_download(repo_id=HF_REPO_ID, filename="model/vgg.h5")),
            "last_conv": "block5_conv2",
        },
        "densenet": {
            "model": load_model(hf_hub_download(repo_id=HF_REPO_ID, filename="model/densenet.h5")),
            "last_conv": "conv4_block24_concat",
        },
    }
    print("✅ Model klasifikasi berhasil dimuat.")

    yolo_path = hf_hub_download(repo_id=HF_REPO_ID, filename="model/yolo.pt")
    YOLO_MODEL = YOLO(yolo_path)
    print("✅ Model YOLO berhasil dimuat. Aplikasi siap.")
except Exception as e:
    print(f"❌ Gagal memuat model: {e}")
    CLASSIFICATION_MODELS = {}
    YOLO_MODEL = None

# ======== Fungsi Utilitas ========
def get_gemini_explanation(prompt_text, image_path):
    """Mengirim teks dan gambar ke Gemini untuk mendapatkan penjelasan."""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        img = Image.open(image_path)
        response = model.generate_content([prompt_text, img])
        return response.text
    except Exception as e:
        return f"Gagal menghasilkan penjelasan dari Gemini: {str(e)}"


def get_gradcam_heatmap(model, img_array, last_conv_layer_name):
    """Membuat heatmap Grad-CAM."""
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# ======== Rute Aplikasi ========
@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/classify", methods=["GET", "POST"])
def classify():
    if request.method == "POST":
        selected_model_name = request.form.get("model")
        img_file = request.files.get("image")

        if not selected_model_name or not img_file or img_file.filename == "":
            return render_template("classify.html", error="Harap pilih model dan unggah gambar.")

        # Buat nama file yang aman dan unik
        filename = secure_filename(img_file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        img_path = os.path.join("static", unique_filename)
        img_file.save(img_path)

        # Pra-pemrosesan gambar
        img = cv2.imread(img_path)
        img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img_array = np.expand_dims(img_resized, axis=0)

        # Dapatkan model yang sudah dimuat
        model_data = CLASSIFICATION_MODELS.get(selected_model_name)
        if not model_data:
            return render_template("classify.html", error=f"Model '{selected_model_name}' tidak ditemukan.")
        
        model = model_data["model"]
        last_conv = model_data["last_conv"]

        # Prediksi
        preds = model.predict(img_array)
        prediction = LABELS[np.argmax(preds)]

        # Buat Grad-CAM
        heatmap = get_gradcam_heatmap(model, img_array, last_conv)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        gradcam_filename = f"gradcam_{unique_filename}"
        gradcam_path = os.path.join("static", gradcam_filename)
        superimposed_img = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)
        cv2.imwrite(gradcam_path, superimposed_img)

        # Dapatkan penjelasan dari Gemini
        prompt = f"Analisis gambar MRI otak berikut, yang telah diklasifikasikan sebagai '{prediction}'. Fokus pada area yang disorot oleh heatmap Grad-CAM untuk menjelaskan kemungkinan lokasi dan karakteristik tumor. Berikan penjelasan ilmiah yang mudah dipahami dalam format Markdown."
        explanation_md = get_gemini_explanation(prompt, gradcam_path)
        explanation_html = markdown.markdown(explanation_md)

        return render_template(
            "classify.html",
            prediction=prediction,
            gradcam_path=gradcam_filename,
            input_path=unique_filename,
            selected_model=selected_model_name,
            explanation=explanation_html,
        )
    return render_template("classify.html")


@app.route("/segment", methods=["GET", "POST"])
def segment():
    if request.method == "POST":
        img_file = request.files.get("image")
        if not img_file or img_file.filename == "":
            return render_template("segment.html", error="Harap unggah gambar.")

        # Buat nama file yang aman dan unik
        filename = secure_filename(img_file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        img_path = os.path.join("static", unique_filename)
        img_file.save(img_path)

        # Segmentasi YOLO
        results = YOLO_MODEL(img_path)
        segmented_filename = f"hasil_{unique_filename}"
        result_img_path = os.path.join("static", segmented_filename)
        results[0].save(filename=result_img_path)

        # Dapatkan penjelasan dari Gemini
        prompt = "Analisis hasil segmentasi tumor otak pada gambar MRI berikut. Jelaskan lokasi, bentuk, dan karakteristik area yang tersegmentasi. Berikan penjelasan ilmiah yang mudah dipahami dalam format Markdown."
        explanation_md = get_gemini_explanation(prompt, result_img_path)
        explanation_html = markdown.markdown(explanation_md)

        return render_template(
            "segment.html",
            segmented_path=segmented_filename,
            input_path=unique_filename,
            explanation=explanation_html,
        )
    return render_template("segment.html")

# Jalankan aplikasi (hanya untuk local, Gunicorn akan menangani di produksi)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)