from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import pickle
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# =========================
# Load Models (مرة واحدة بس)
# =========================

image_model = load_model("soil_image_model.h5")

with open("soil_model_full.pkl", "rb") as f:
    model_data = pickle.load(f)

rf_model = model_data["model"]
le_color = model_data["le_color"]
le_type = model_data["le_type"]
le_quality = model_data["le_quality"]
le_crop = model_data["le_crop"]

with open("image_classes.json", "r") as f:
    class_dict = json.load(f)

classes = list(class_dict.keys())

# =========================
# Functions
# =========================

def predict_soil_type(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = image_model.predict(img_array)
    return classes[np.argmax(pred)]

def get_color(soil_type):
    if soil_type == "Clay":
        return "Black"
    elif soil_type == "Sand":
        return "Yellow"
    else:
        return "Brown"

# =========================
# API (واحد بس للحالتين)
# =========================

@app.route("/predict", methods=["POST"])
def predict():

    # لازم دايمًا موجودين
    ph = float(request.form["ph"])
    moisture = float(request.form["moisture"])

    # =========================
    # الحالة 1: فيه صورة
    # =========================
    if "image" in request.files:

        file = request.files["image"]

        filepath = "temp.jpg"
        file.save(filepath)

        soil_type = predict_soil_type(filepath)
        soil_color = get_color(soil_type)

        os.remove(filepath)

    # =========================
    # الحالة 2: manual (مفيش صورة)
    # =========================
    else:
        soil_color = request.form["color"]

    # =========================
    # باقي الموديل
    # =========================

    sample = pd.DataFrame({
        "Soil_Color": [soil_color],
        "pH": [ph],
        "SWC (%)": [moisture]
    })

    sample["Soil_Color"] = le_color.transform(sample["Soil_Color"])

    pred = rf_model.predict(sample)

    final_type = le_type.inverse_transform([pred[0][0]])[0]
    quality = le_quality.inverse_transform([pred[0][1]])[0]
    crop = le_crop.inverse_transform([pred[0][2]])[0]

    return jsonify({
        "soil_type": final_type,
        "soil_quality": quality,
        "crop": crop
    })

# =========================
# Run App
# =========================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)