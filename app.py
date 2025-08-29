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

matplotlib.use("Agg")
from dotenv import load_dotenv

# Load environment variables

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

HF_REPO_ID = "Revando/EfficientNet"

# Utility functions

def get_gemini_explanation(prompt):
    try:
        model = genai.GenerativeModel("gemini-2.5-pro")
        #model = genai.GenerativeModel("gemini-1.5-flash")
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


def load_selected_model(model_name):
    if model_name == "efficientnet":
        model_path = hf_hub_download(repo_id=HF_REPO_ID, filename="model/effnet.h5")
        model = load_model(model_path)
        last_conv = "top_conv"
    elif model_name == "resnet":
        model_path = hf_hub_download(repo_id=HF_REPO_ID, filename="model/resnet.h5")
        model = load_model(model_path)
        last_conv = "conv5_block3_out"
    elif model_name == "vgg":
        model_path = hf_hub_download(repo_id=HF_REPO_ID, filename="model/vgg.h5")
        model = load_model(model_path)
        last_conv = "block5_conv2"
    elif model_name == "densenet":
        model_path = hf_hub_download(repo_id=HF_REPO_ID, filename="model/densenet.h5")
        model = load_model(model_path)
        last_conv = "conv4_block24_concat"
    else:
        raise ValueError("Unknown model")
    return model, last_conv

yolo_model = YOLO("model/yolo.pt")

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

# Routes

@app.route("/")
def dashboard():
    return render_template("dashboard.html")


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

    return render_template(
        "classify.html",
        prediction=prediction,
        gradcam_path=gradcam_filename,
        gradcam_only_path=gradcam_only_filename,
        input_path=input_path,
        selected_model=selected_model,
        explanation=explanation_html,
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

    return render_template(
        "segment.html",
        segmented_path=segmented_path,
        segmentation_info=segmentation_info,
        error_message=error_message,
    )

if __name__ == "__main__":
    app.run(debug=True)