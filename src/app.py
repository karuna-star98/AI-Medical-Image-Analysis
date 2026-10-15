"""Flask API + tiny web page: upload an image, get a prediction.
Run:  python main.py serve --model models/cnn_best.keras
"""
import os

import cv2
import numpy as np
from flask import Flask, jsonify, render_template_string, request

from src.config import DISCLAIMER, MODELS_DIR
from src.predict import load_trained_model, predict_array

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024      # 10 MB upload limit
ALLOWED = {".jpg", ".jpeg", ".png", ".bmp"}

PAGE = """<!doctype html><title>AI Medical Image Analysis</title>
<body style="font-family:sans-serif;max-width:560px;margin:40px auto">
<h2>AI-Powered Medical Image Analysis</h2>
<p style="color:#b00"><b>{{ disclaimer }}</b></p>
<input type="file" id="f" accept="image/*"> <button onclick="go()">Analyse</button>
<pre id="out" style="background:#f4f4f4;padding:12px;margin-top:16px"></pre>
<script>
async function go(){
  const f=document.getElementById('f').files[0]; if(!f) return;
  const fd=new FormData(); fd.append('image',f);
  const r=await fetch('/predict',{method:'POST',body:fd});
  document.getElementById('out').textContent=JSON.stringify(await r.json(),null,2);
}
</script></body>"""


def get_model():
    return load_trained_model(os.environ.get("MODEL_PATH", str(MODELS_DIR / "cnn_best.keras")))


@app.get("/")
def index():
    return render_template_string(PAGE, disclaimer=DISCLAIMER)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify({"error": "Send an image file in the 'image' form field."}), 400
    if os.path.splitext(file.filename)[1].lower() not in ALLOWED:
        return jsonify({"error": f"Unsupported file type. Use one of {sorted(ALLOWED)}"}), 400

    gray = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return jsonify({"error": "Could not decode image."}), 400

    result = predict_array(get_model(), gray)
    result["disclaimer"] = DISCLAIMER
    return jsonify(result)
