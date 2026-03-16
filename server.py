import os
from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image

app = Flask(__name__)
model = YOLO("best.pt")

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
            detections.append({"damage_type": label, "confidence": conf})

    return jsonify(detections)

# IMPORTANT: use PORT env var (fallback to 10000)
port = int(os.environ.get("PORT", 10000))
app.run(host="0.0.0.0", port=port)
