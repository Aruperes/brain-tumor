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


# Load all models once at startup
MODEL_PATHS = {
    "efficientnet": ("model/effnet.h5", "top_conv"),
    "resnet": ("model/resnet.h5", "conv5_block3_out"),
    "vgg": ("model/vgg.h5", "block5_conv2"),
    "densenet": ("model/densenet.h5", "conv4_block24_concat"),
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
    prediction = None
    gradcam_filename = None
    gradcam_only_filename = None
    selected_model = None
    classification_info = None
    explanation_html = None
    input_path = None  
    error_message = None

    if request.method == "POST":
        selected_model = request.form["model"]
        img_file = request.files["image"]
        img_path = os.path.join("static", img_file.filename)
        img_file.save(img_path)
        input_path = img_file.filename

        # Input Validation
        if not validate_with_gemini(img_path):
         error_message = "Gambar Anda tidak dikenali sebagai citra MRI otak. Silakan unggah gambar MRI otak."
         return render_template("classify.html", error_message=error_message)

        # Preprocess image
        img = cv2.imread(img_path)
        img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img_array = np.expand_dims(img_resized, axis=0)

        # Load classification model
        model, last_conv = load_selected_model(selected_model)
        preds = model.predict(img_array)
        pred_class = LABELS[np.argmax(preds)]
        prediction = pred_class

        # Grad-CAM heatmap
        gradcam_filename = f"gradcam_{img_file.filename}"
        gradcam_only_filename = f"heatmap_{img_file.filename}"
        gradcam_path = os.path.join("static", gradcam_filename)
        heatmap = get_gradcam_heatmap(model, img_array, last_conv)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        #heatmap saja
        gradcam_only_filename = f"heatmap_{img_file.filename}"
        gradcam_only_path = os.path.join("static", gradcam_only_filename)
        cv2.imwrite(gradcam_only_path, heatmap_color)
        # Gabungan overlay
        superimposed_img = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)
        cv2.imwrite(gradcam_path, superimposed_img)

        # Gemini AI explanation with Markdown formatting request
        prompt = (
            f"Saya mengirimkan gambar MRI otak yang telah diklasifikasikan sebagai: {prediction}.\n"
            f"Saya juga melampirkan hasil visualisasi Grad-CAM pada gambar ini (file: {gradcam_filename}). "
            "Tolong analisis gambar Grad-CAM tersebut secara detail. "
            "Jelaskan secara spesifik di mana letak area yang paling disorot oleh Grad-CAM pada gambar, dan apakah area tersebut menunjukkan keberadaan tumor. "
            "Jika terdapat tumor, sebutkan secara jelas lokasi atau area pada otak yang terindikasi oleh Grad-CAM. "
            "Jangan hanya mengulang hasil prediksi atau penjelasan umum tentang Grad-CAM, tapi berikan analisis berdasarkan gambar Grad-CAM yang diberikan. "
            "Gunakan bahasa yang mudah dipahami namun tetap ilmiah. "
            "Sertakan potensi implikasi dan langkah selanjutnya secara umum, tanpa memberikan diagnosis pasti.\n"
            "Format penjelasan menggunakan Markdown dengan heading, poin-poin, dan penekanan teks agar mudah dibaca."
        )
        classification_info = get_gemini_explanation(prompt)

        # Convert Markdown to HTML 
        if classification_info:
            explanation_html = markdown.markdown(
                classification_info, extensions=["fenced_code", "tables"]
            )

        # Pada classify (setelah prediction berhasil)
        if prediction:
            history_collection.insert_one({
                "type": "Classification",
                "filename": input_path,
                "result": prediction,
                "model": selected_model,
                "timestamp": datetime.utcnow()
            })

    return render_template(
        "classify.html",
        prediction=prediction,
        gradcam_path=gradcam_filename,
        gradcam_only_path=gradcam_only_filename,
        input_path=input_path,
        selected_model=selected_model,
        explanation=explanation_html,
        model_performance=MODEL_PERFORMANCE.get(selected_model) if selected_model else None,
    )

@app.route("/segment", methods=["GET", "POST"])
def segment():
    segmented_path = None
    segmentation_info = None
    error_message = None
    if request.method == "POST":
        image = request.files["image"]
        image_path = os.path.join("static", image.filename)
        image.save(image_path)

        # Input validation 
        if not validate_with_gemini(image_path):
            error_message = "Gambar yang Anda unggah tidak dikenali sebagai citra MRI otak. Silakan unggah gambar MRI otak."
            return render_template("segment.html", error_message=error_message)

        results = yolo_model(image_path)
        result_img_path = f"static/hasil_{image.filename}"
        results[0].save(filename=result_img_path)
        segmented_path = f"hasil_{image.filename}"

        prompt = (
            f"Saya mengirimkan gambar MRI otak yang telah melalui proses segmentasi tumor "
            f"menggunakan YOLO (file: {segmented_path}). "
            "Area yang tersegmentasi menunjukkan kemungkinan keberadaan tumor.\n\n"
            "Tolong analisis hasil segmentasi ini secara detail:\n"
            "- Sebutkan lokasi area yang ditandai oleh segmentasi.\n"
            "- Deskripsikan jenis tumor, bentuk, ukuran relatif, dan karakteristik visual.\n"
            "- Jelaskan apakah area tersebut konsisten dengan ciri-ciri tumor otak.\n\n"
            "Gunakan bahasa yang mudah dipahami namun tetap ilmiah. "
            "Format penjelasan menggunakan Markdown."
        )
        segmentation_info = get_gemini_explanation(prompt)

        if segmentation_info:
            segmentation_info = markdown.markdown(
                segmentation_info, extensions=["fenced_code", "tables"]
            )

        # Pada segment (setelah segmented_path berhasil)
        if segmented_path:
            history_collection.insert_one({
                "type": "Segmentation",
                "filename": segmented_path,
                "result": "Segmented",
                "model": "YOLO",
                "timestamp": datetime.utcnow()
            })

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