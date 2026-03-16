from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import requests
from datetime import datetime

app = Flask(__name__)

model = YOLO("best.pt")

SUPABASE_URL = "https://kcknxkeccmfxjkxgkiro.supabase.co/functions/v1/add-detection"

@app.route("/predict", methods=["POST"])
def predict():

    image = request.files["image"]
    img = Image.open(image)

    results = model(img)

    detections = []

    for r in results:
        for box in r.boxes:

            label = model.names[int(box.cls)]
            conf = float(box.conf)

            data = {
                "damage_type": label,
                "confidence": conf,
                "latitude": 16.544,
                "longitude": 81.521,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "image_url": "uploaded_image"
            }

            requests.post(SUPABASE_URL, json=data)

            detections.append(data)

    return jsonify(detections)

app.run(host="0.0.0.0", port=10000)