# API для обеих моделей (stable/new)
import os
import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_VERSION = os.getenv("MODEL_VERSION", "v0.9.9")
MODEL_TYPE = os.getenv("MODEL_TYPE", "stable")
MODEL_PATH = os.getenv("MODEL_PATH", "models/stable_model.pkl")

model = None

def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Файл модели не найден: {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

load_model()

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "version": MODEL_VERSION
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True) or {}
        x = data.get("x", [1, 2, 3, 4])

        if len(x) != 4:
            return jsonify({"error": "Количество фич не соответствует датасету iris"}), 400

        pred = model.predict([x])[0]

        return jsonify({
            "status": "ok",
            "version": MODEL_VERSION,
            "model_type": MODEL_TYPE,
            "prediction": int(pred)
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
