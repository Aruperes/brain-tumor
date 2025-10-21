from flask import Flask, render_template, request
import os
import numpy as np
from ultralytics import YOLO
import google.generativeai as genai
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import matplotlib
import markdown
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
import base64
from bson import ObjectId
from flask import redirect, url_for

matplotlib.use("Agg")
from dotenv import load_dotenv

from pymongo import MongoClient
from datetime import datetime


MONGODB_URI = os.getenv("MONGODB_URI")
mongo_client = MongoClient(MONGODB_URI)
db = mongo_client["braintumor"]
history_collection = db["history"]

# Load environment variables

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

HF_REPO_ID = "Revando/EfficientNet"

# Utility functions

def get_gemini_explanation(prompt):
    try:
        model = genai.GenerativeModel("gemini-2.5-pro")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gagal menghasilkan penjelasan AI: {str(e)}"


LABELS = ["Glioma Tumor", "Normal", "Meningioma Tumor", "Pituitary Tumor"]
IMAGE_SIZE = 224


def get_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Load models
MODEL_PATHS = {
    "efficientnet": ("model/effnetb2.h5", "top_conv"),
    #"resnet": ("model/resnet101.h5", "conv5_block3_out"),
    #"vgg": ("model/vgg.h5", "block5_conv2"),
    #"densenet": ("model/densenet201.h5", "conv4_block24_concat"),
}

MODELS = {}
for name, (filename, last_conv) in MODEL_PATHS.items():
    path = hf_hub_download(repo_id=HF_REPO_ID, filename=filename)
    MODELS[name] = {
        "model": load_model(path),
        "last_conv": last_conv
    }

yolo_model_path = hf_hub_download(repo_id=HF_REPO_ID, filename="model/yolo.pt")
yolo_model = YOLO(yolo_model_path)

def load_selected_model(model_name):
    if model_name not in MODELS:
        raise ValueError("Unknown model")
    return MODELS[model_name]["model"], MODELS[model_name]["last_conv"]

def validate_with_gemini(image_path):
    try:
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        prompt = (
            "Anda adalah validator untuk aplikasi medis. "
            "Tugas Anda adalah memeriksa apakah gambar berikut adalah citra MRI otak manusia. "
            "Jika gambar adalah MRI otak (dengan atau tanpa tumor), jawab 'VALID'. "
            "Jika bukan (misalnya foto wajah, hewan, CT scan, X-ray, atau objek lain), jawab 'INVALID'. "
            "Jawaban hanya satu kata: VALID atau INVALID."
        )

        model = genai.GenerativeModel("gemini-2.5-pro")
        response = model.generate_content(
            [
                prompt,
                {"mime_type": "image/jpeg", "data": img_bytes}
            ]
        )

        return response.text.strip().upper() == "VALID"
    except Exception as e:
        print("Error Gemini Validation:", e)
        return False

# Flask app setup

app = Flask(__name__)

# Model performance metrics
MODEL_PERFORMANCE = {
    "efficientnet": {
        "cm_image": "cm_efficientnet.png",
        "accuracy": 98.47,
        "precision": 98.26,
        "recall": 98.66,
        "f1_score": 98.45,
        "specificity": 99.50,
    },
    "resnet": {
        "cm_image": "cm_resnet.png",
        "accuracy": 96.94,
        "precision": 96.66,
        "recall": 97.12,
        "f1_score": 96.87,
        "specificity": 98.99,
    },
    "vgg": {
        "cm_image": "cm_vgg.png",
        "accuracy": 0.92,
        "precision": 0.91,
        "recall": 0.90,
        "f1_score": 0.905,
        "specificity": 0.93,
    },
    "densenet": {
        "cm_image": "cm_densenet.png",
        "accuracy": 97.55,
        "precision": 97.61,
        "recall": 97.89,
        "f1_score": 97.73,
        "specificity": 99.17,
    },
}

# Routes

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/classify", methods=["GET", "POST"])
def classify():
    # Initialize all variables
    prediction = None
    gradcam_filename = None
    gradcam_only_filename = None
    selected_model = None
    classification_info = None
    explanation_html = None
    error_message = None
    gradcam_path = None
    gradcam_only_path = None
    input_b64 = None  # Initialize input_b64

    if request.method == "POST":
        username = request.form["username"] 
        selected_model = request.form["model"]
        img_file = request.files["image"]
        img_bytes = img_file.read()
        input_path = img_file.filename

        # Simpan file sementara di memori untuk validasi dan proses
        with open("temp_input.jpg", "wb") as f:
            f.write(img_bytes)

        # Input Validation
        if not validate_with_gemini("temp_input.jpg"):
            os.remove("temp_input.jpg")
            error_message = "Gambar Anda tidak dikenali sebagai citra MRI otak. Silakan unggah gambar MRI otak."
            return render_template("classify.html", error_message=error_message)

        # Preprocess image
        img_array = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        img_resized = cv2.resize(img_array, (IMAGE_SIZE, IMAGE_SIZE))
        img_input = np.expand_dims(img_resized, axis=0)

        # Load classification model
        model, last_conv = load_selected_model(selected_model)
        preds = model.predict(img_input)
        pred_class = LABELS[np.argmax(preds)]
        prediction = pred_class

        # Grad-CAM heatmap
        heatmap = get_gradcam_heatmap(model, img_input, last_conv)
        heatmap = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # Heatmap only (base64)
        _, heatmap_buf = cv2.imencode('.jpg', heatmap_color)
        heatmap_b64 = base64.b64encode(heatmap_buf).decode("utf-8")

        # GradCAM overlay (base64)
        superimposed_img = cv2.addWeighted(img_array, 0.6, heatmap_color, 0.4, 0)
        _, gradcam_buf = cv2.imencode('.jpg', superimposed_img)
        gradcam_b64 = base64.b64encode(gradcam_buf).decode("utf-8")

        # Input image (base64)
        input_b64 = base64.b64encode(img_bytes).decode("utf-8")

        os.remove("temp_input.jpg")

        # Gemini AI explanation
        prompt = (
            f"Saya mengirimkan gambar MRI otak yang telah diklasifikasikan sebagai: {prediction}.\n"
            "Saya juga melampirkan hasil visualisasi Grad-CAM pada gambar ini. "
            "Tolong analisis gambar Grad-CAM tersebut secara detail. "
            "Jelaskan secara spesifik di mana letak area yang paling disorot oleh Grad-CAM pada gambar, dan apakah area tersebut menunjukkan keberadaan tumor. "
            "Jika terdapat tumor, sebutkan secara jelas lokasi atau area pada otak yang terindikasi oleh Grad-CAM. "
            "Jangan hanya mengulang hasil prediksi atau penjelasan umum tentang Grad-CAM, tapi berikan analisis berdasarkan gambar Grad-CAM yang diberikan. "
            "Gunakan bahasa yang mudah dipahami namun tetap ilmiah. "
            "Sertakan potensi implikasi dan langkah selanjutnya secara umum, tanpa memberikan diagnosis pasti.\n"
            "Langsung berikan penjelasan tanpa kata-kata yang tidak relevan. "
            "Sertakan link ke sumber atau jurnal ilmiah untuk setiap pernyataan dalam penjelasan Anda mengenai hasil klasifikasi.\n"
            "Format penjelasan menggunakan Markdown dengan heading, poin-poin, dan penekanan teks agar mudah dibaca."
        )
        classification_info = get_gemini_explanation(prompt)
        if classification_info:
            explanation_html = markdown.markdown(
                classification_info, extensions=["fenced_code", "tables"]
            )

        # Simpan ke MongoDB
        if prediction:
            history_collection.insert_one({
                "type": "Classification",
                "filename": input_path,
                "result": prediction,
                "username": request.form["username"],  
                "timestamp": datetime.utcnow(),
                "input_b64": input_b64,
                "gradcam_b64": gradcam_b64,
                "heatmap_b64": heatmap_b64,
            })

        # Untuk preview di halaman
        gradcam_path = gradcam_b64
        gradcam_only_path = heatmap_b64
        input_path = input_b64

    return render_template(
        "classify.html",
        prediction=prediction,
        gradcam_path=gradcam_path,           
        gradcam_only_path=gradcam_only_path, 
        input_path=input_b64,               
        selected_model=selected_model,
        explanation=explanation_html,
        model_performance=MODEL_PERFORMANCE.get(selected_model) if selected_model else None,
        error_message=error_message,
    )

@app.route("/segment", methods=["GET", "POST"])
def segment():
    segmented_path = None
    segmentation_info = None
    error_message = None
    if request.method == "POST":
        username = request.form["username"]  
        image = request.files["image"]
        img_bytes = image.read()
        image_path = image.filename

        # Simpan file sementara di memori untuk validasi dan proses
        with open("temp_segment.jpg", "wb") as f:
            f.write(img_bytes)

        # Input validation 
        if not validate_with_gemini("temp_segment.jpg"):
            os.remove("temp_segment.jpg")
            error_message = "Gambar yang Anda unggah tidak dikenali sebagai citra MRI otak. Silakan unggah gambar MRI otak."
            return render_template("segment.html", error_message=error_message)

        # YOLO segmentasi
        results = yolo_model("temp_segment.jpg")
        result_img = results[0].plot()
        _, result_buf = cv2.imencode('.jpg', result_img)
        segmented_b64 = base64.b64encode(result_buf).decode("utf-8")
        input_b64 = base64.b64encode(img_bytes).decode("utf-8")

        os.remove("temp_segment.jpg")

        # Simpan ke MongoDB
        if segmented_b64:
            history_collection.insert_one({
                "type": "Segmentation",
                "filename": image_path,
                "result": "Segmented",
                "username": request.form["username"], 
                "timestamp": datetime.utcnow(),
                "input_b64": input_b64,
                "segmented_b64": segmented_b64,
            })

        segmented_path = segmented_b64

    return render_template(
        "segment.html",
        segmented_path=segmented_path,
        segmentation_info=segmentation_info,
        error_message=error_message,
    )

@app.route("/history")
def history():
    history_items = list(history_collection.find().sort("timestamp", -1))
    return render_template("history.html", history=history_items)

@app.route("/delete_history/<history_id>", methods=["POST"])
def delete_history(history_id):
    history_collection.delete_one({"_id": ObjectId(history_id)})
    return redirect(url_for("history"))

@app.route("/delete_all_history", methods=["POST"])
def delete_all_history():
    history_collection.delete_many({})
    return redirect(url_for("history"))

if __name__ == "__main__":
    app.run(debug=True)