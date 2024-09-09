from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import cv2

app = Flask(__name__)

# Load the models
dl_coloring_model = load_model("model/autism_binary_coloring_modelv.h5")
dl_handwriting_model = load_model("model/autism_binary_handwriting_modelv.h5")
dl_image_model = load_model("model/my_model.h5")
ml_model = load_model("model/autism_ml_model.h5")  # Load the ML model saved as a .h5 file



# Helper function to predict using the deep learning models
def predict_image(model, img_path, target_size=(128, 128)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Rescale as done during training
    prediction = model.predict(img_array)
    return "Non-ASD" if prediction[0][0] > 0.5 else "ASD"

# Helper function to predict using the newly added model
def predict_autism_image(model, img_path, target_size=(224, 224)):
    img = cv2.imread(img_path)
    img = cv2.resize(img, target_size)
    img_array = np.expand_dims(img, axis=0)
    prediction = model.predict(img_array)[0][0]
    return "Autistic" if prediction >= 0.5 else "Non_Autistic", float(prediction)

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

# Deep Learning Autism Image Model API
@app.route('/dl', methods=['POST'])
def predict_autism_image_api():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        filepath = os.path.join("uploads", file.filename)
        file.save(filepath)
        prediction, probability = predict_autism_image(dl_image_model, filepath)
        os.remove(filepath)
        return jsonify({"prediction": prediction, "probability": probability})


if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(host='0.0.0.0', port=3000, debug=False)
