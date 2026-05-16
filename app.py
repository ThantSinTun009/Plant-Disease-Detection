from flask import Flask, render_template, request, flash, redirect, url_for
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import numpy as np
import os
import tensorflow as tf
from PIL import Image
import json

app = Flask(__name__)
app.secret_key = "supersecretkey"

# ==========================================
# EMAIL CONFIG
# ==========================================

EMAIL_ADDRESS = "thantsintun@parami.edu.mm"
EMAIL_PASSWORD = "bjaw mbcv rnnq gmuy"
RECEIVER_EMAIL = "thantsintun@parami.edu.mm"

# ==========================================
# LOAD MODEL + CLASS NAMES
# ==========================================

with open("class_names.json") as f:
    class_names = json.load(f)

model = tf.keras.models.load_model("plant_disease_model.keras", compile=False)

# ==========================================
# SAFE CLASS MAPPING (IMPORTANT FIX)
# ==========================================

if isinstance(class_names, dict):
    # convert {index: name}
    index_to_class = {int(k): v for k, v in class_names.items()}
else:
    # list format
    index_to_class = {i: name for i, name in enumerate(class_names)}

# ==========================================
# DISEASE INFO (23 CLASSES)
# ==========================================

disease_info = {
    "Apple___Apple_scab": {
        "solution": "Prune infected branches and apply sulfur-based fungicide.",
        "fertilizer": "Use balanced NPK fertilizer and organic compost."
    },
    "Apple___Black_rot": {
        "solution": "Remove infected fruits and apply copper fungicide.",
        "fertilizer": "Use potassium-rich fertilizer."
    },
    "Apple___Cedar_apple_rust": {
        "solution": "Remove cedar hosts and spray fungicide.",
        "fertilizer": "Balanced fertilizer recommended."
    },
    "Apple___healthy": {
        "solution": "Plant is healthy.",
        "fertilizer": "Maintain organic compost."
    },

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "solution": "Crop rotation and fungicide application.",
        "fertilizer": "Nitrogen-rich fertilizer."
    },
    "Corn_(maize)___Common_rust_": {
        "solution": "Apply fungicide and resistant seeds.",
        "fertilizer": "NPK fertilizer recommended."
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "solution": "Remove infected leaves and spray fungicide.",
        "fertilizer": "Balanced fertilizer."
    },
    "Corn_(maize)___healthy": {
        "solution": "Plant is healthy.",
        "fertilizer": "Maintain soil nutrients."
    },

    "Pepper__bell___Bacterial_spot": {
        "solution": "Use copper spray and avoid wet leaves.",
        "fertilizer": "Balanced fertilizer."
    },
    "Pepper__bell___healthy": {
        "solution": "Plant is healthy.",
        "fertilizer": "Maintain nutrients."
    },

    "Potato___Early_blight": {
        "solution": "Apply fungicide and remove infected leaves.",
        "fertilizer": "Phosphorus-rich fertilizer."
    },
    "Potato___Late_blight": {
        "solution": "Remove infected plants immediately.",
        "fertilizer": "Balanced NPK fertilizer."
    },
    "Potato___healthy": {
        "solution": "Plant is healthy.",
        "fertilizer": "Maintain compost."
    },

    "Tomato_Bacterial_spot": {
        "solution": "Use copper spray.",
        "fertilizer": "Balanced fertilizer."
    },
    "Tomato_Early_blight": {
        "solution": "Remove infected leaves.",
        "fertilizer": "Calcium + potassium fertilizer."
    },
    "Tomato_Late_blight": {
        "solution": "Apply copper fungicide.",
        "fertilizer": "Potassium-rich fertilizer."
    },
    "Tomato_Leaf_Mold": {
        "solution": "Improve ventilation.",
        "fertilizer": "Balanced NPK."
    },
    "Tomato_Septoria_leaf_spot": {
        "solution": "Remove infected leaves.",
        "fertilizer": "Balanced fertilizer."
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "solution": "Use neem oil spray.",
        "fertilizer": "Potassium support."
    },
    "Tomato__Target_Spot": {
        "solution": "Apply fungicide.",
        "fertilizer": "Balanced fertilizer."
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "solution": "Remove infected plants.",
        "fertilizer": "Avoid excess nitrogen."
    },
    "Tomato__Tomato_mosaic_virus": {
        "solution": "Remove infected plants immediately.",
        "fertilizer": "Maintain soil health."
    },
    "Tomato_healthy": {
        "solution": "Plant is healthy.",
        "fertilizer": "Maintain nutrients."
    }
}

# ==========================================
# CONFIG
# ==========================================

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

IMG_SIZE = (224, 224)

# ==========================================
# IMAGE PREPROCESSING
# ==========================================


def preprocess_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((224, 224))

    img = np.array(img, dtype=np.float32)
    img = np.expand_dims(img, axis=0)

    return img

def format_label(label):
    # 1. Replace underscores
    label = label.replace("_", " ")

    # 2. Fix common weird spacing issues
    label = " ".join(label.split())

    # 3. Split crop and disease (first "___" or "__" pattern already removed above)
    parts = label.split(" ", 1)

    if len(parts) == 2:
        crop = parts[0]
        disease = parts[1]

        # Clean crop name
        crop = crop.replace("(maize)", "Maize").title()

        # Clean disease text
        disease = disease.replace("Gray leaf spot", "Gray Leaf Spot")
        disease = disease.replace("Leaf blight", "Leaf Blight")
        disease = disease.replace("spider mites two spotted spider mite",
                                  "Spider Mites (Two-Spotted)")

        # Healthy case
        if "healthy" in disease.lower():
            return f"{crop} - Healthy 🌿"

        return f"{crop} - {disease} 🍂"

    return label.title()

# ==========================================
# HOME
# ==========================================

@app.route('/')
def home():
    return render_template("home.html")

# ==========================================
# PREDICTION
# ==========================================

@app.route("/predict", methods=["GET", "POST"])
def predict():

    prediction = None
    confidence = None
    image_url = None
    solution = None
    friendly_label=None
    fertilizer = None

    if request.method == "POST":
        try:
            file = request.files.get("file")

            if not file or file.filename == "":
                flash("No file selected", "danger")
                return redirect(url_for("home"))

            filename = file.filename
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(image_path)

            image_url = f"/static/uploads/{filename}"

            # =========================
            # PREDICTION
            # =========================

            img = preprocess_image(image_path)
            preds = model.predict(img, verbose=0)
            preds = np.squeeze(preds)

            idx = int(np.argmax(preds))

            prediction = index_to_class.get(idx, "Unknown")

            confidence = float(preds[idx]) * 100
            
            # raw label from mapping
            raw_label = index_to_class.get(idx, "Unknown")

            # user-friendly label
            friendly_label = format_label(raw_label)
            

            print("PREDICTED:", prediction)
            print("CONFIDENCE:", confidence)

            # =========================
            # DISEASE INFO
            # =========================

            info = disease_info.get(raw_label, {
                "solution": "Consult agriculture specialist.",
                "fertilizer": "Use balanced fertilizer."
                })

            solution = info["solution"]
            fertilizer = info["fertilizer"]

        except Exception as e:
            print("ERROR:", e)
            flash("Prediction failed.", "danger")

    return render_template(
        "predict.html",
        prediction=prediction,
        friendly_prediction=friendly_label,
        confidence=confidence,
        image_path=image_url,
        solution=solution,
        fertilizer=fertilizer
    )
    

# ==========================================
# CONTACT
# ==========================================
@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        full_name = request.form['name']
        email = request.form['email']
        message = request.form['message']

        try:
            msg_admin = MIMEMultipart()
            msg_admin['From'] = EMAIL_ADDRESS
            msg_admin['To'] = RECEIVER_EMAIL
            msg_admin['Subject'] = f"New Contact Message from {full_name}"
            msg_admin['Reply-To'] = email

            msg_admin.attach(MIMEText(message, 'plain'))

            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
                server.send_message(msg_admin)

            flash("Message sent successfully!", "success")

        except Exception as e:
            print("EMAIL ERROR:", e)
            flash("Something went wrong.", "danger")

        return redirect(url_for('contact'))

    return render_template('contact.html')

# ==========================================
# RUN
# ==========================================

if __name__ == '__main__':
    app.run(debug=True)