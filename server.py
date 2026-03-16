from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import traceback

app = Flask(__name__)

model = YOLO("best.pt")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        image = request.files["image"]
        img = Image.open(image)

        results = model(img)

        detections = []

        for r in results:
            for box in r.boxes:
                label = model.names[int(box.cls)]
                conf = float(box.conf)

                detections.append({
                    "damage_type": label,
                    "confidence": conf
                })

        return jsonify(detections)

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)})
