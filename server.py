from flask import Flask, request, jsonify
from ultralytics import YOLO
import base64
import numpy as np
import cv2
import traceback

app = Flask(__name__)

# 🔥 IMPORTANT: use lightweight if memory issue
model = YOLO("yolov8n.pt")   # OR "best.pt" if stable

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # ✅ get base64 image
        img_b64 = data["image"]

        # decode
        img_bytes = base64.b64decode(img_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        results = model(img)

        detections = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                detections.append({
                    "label": model.names[cls],
                    "confidence": conf,
                    "x": x1,
                    "y": y1,
                    "w": x2 - x1,
                    "h": y2 - y1
                })

        return jsonify({"detections": detections})

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)})
