import os
from flask import Flask, jsonify, request
from predictor import OCRPredictor

MODEL_PATH = os.getenv("MODEL_PATH", "models/final_finetuned_model.keras")

app = Flask(__name__)
predictor = OCRPredictor(model_path=MODEL_PATH)


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "model_loaded": predictor.model is not None,
            "model_path": predictor.model_path,
            "input_size": [predictor.img_height, predictor.img_width],
            "num_classes": predictor.num_classes,
            "charset": predictor.charset,
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return (
            jsonify(
                {
                    "error": "Missing image file. Send multipart/form-data with key 'image'."
                }
            ),
            400,
        )

    image_file = request.files["image"]
    if image_file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    try:
        result = predictor.predict(image_file)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
