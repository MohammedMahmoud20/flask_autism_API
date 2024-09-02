from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import joblib  # For loading the ML model
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load the models
dl_coloring_model = load_model("model/autism_binary_coloring_modelv.h5")
dl_handwriting_model = load_model("model/autism_binary_handwriting_modelv.h5")  # Load the handwriting model
ml_model = joblib.load("model/autism_ml_model.pkl")  # Load the ML model

# Load the scaler used during training for the ML model
scaler = joblib.load("model/ml_scaler.pkl")

# Helper function to predict using the deep learning models
def predict_image(model, img_path, target_size=(128, 128)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Rescale as done during training
    prediction = model.predict(img_array)
    return "Non-ASD" if prediction[0][0] > 0.5 else "ASD"

# Coloring Model API
@app.route('/coloring', methods=['POST'])
def predict_coloring():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        filepath = os.path.join("uploads", file.filename)
        file.save(filepath)
        prediction = predict_image(dl_coloring_model, filepath)
        os.remove(filepath)
        return jsonify({"prediction": prediction})

# Handwriting Model API
@app.route('/handwriting', methods=['POST'])
def predict_handwriting():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        filepath = os.path.join("uploads", file.filename)
        file.save(filepath)
        prediction = predict_image(dl_handwriting_model, filepath)
        os.remove(filepath)
        return jsonify({"prediction": prediction})

# Machine Learning Model API
@app.route('/ml', methods=['POST'])
def predict_ml():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400
        input_df = pd.DataFrame([data])
        input_scaled = scaler.transform(input_df)
        prediction = ml_model.predict(input_scaled)
        result = "Non-ASD" if prediction[0] == 0 else "ASD"
        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(host='0.0.0.0', port=3000, debug=False)
