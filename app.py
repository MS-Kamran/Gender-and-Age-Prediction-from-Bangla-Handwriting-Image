from flask import Flask, render_template, request, jsonify
from flask_cors import CORS  # Import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os
from PIL import Image

app = Flask(__name__)

# Enable CORS for the app
CORS(app)  # This will allow cross-origin requests

# Load the gender model
gender_model = load_model('model/AP-AksharNet_1024Gender_Trained.h5')

# Load the age model (replace with your actual age model path)
age_model = load_model('model/AP-AksharNet_1024Age_Trained.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_gender', methods=['POST'])
def predict_gender():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save the file to a temporary location
    filepath = os.path.join('static', file.filename)
    file.save(filepath)

    try:
        # Open the image
        external_image = Image.open(filepath)

        # Force the image to RGB
        if external_image.mode != 'RGB':
            external_image = external_image.convert('RGB')

        # Resize the image
        external_image_resized = external_image.resize((1024, 512))

        # Convert the image to a numpy array and normalize pixel values
        external_image_array = np.array(external_image_resized) / 255.0

        # Ensure the shape is (512, 1024, 3)
        if external_image_array.shape != (512, 1024, 3):
            return jsonify({'error': f'Image shape mismatch: {external_image_array.shape}'}), 400

        # Reshape the image to match the input shape of the model (add batch dimension)
        external_image_array = external_image_array.reshape(1, 512, 1024, 3)

        # Make gender prediction
        prediction = gender_model.predict(external_image_array)

        # Define gender class labels
        gender_labels = ["Female", "Male"]
        result = prediction.argmax(axis=1)[0]
        result_label = gender_labels[result]

        # Get gender prediction probabilities
        prediction_probs = prediction[0]
        gender_prediction_details = {
            gender_labels[i]: prediction_probs[i] * 100 for i in range(len(gender_labels))
        }

        return jsonify({
            'gender_result': result_label,
            'gender_probabilities': gender_prediction_details,
            'filename': filepath
        })

    except Exception as e:
        return jsonify({'error': f'Error processing image for gender prediction: {str(e)}'}), 500


@app.route('/predict_age', methods=['POST'])
def predict_age():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save the file to a temporary location
    filepath = os.path.join('static', file.filename)
    file.save(filepath)

    try:
        # Open the image
        external_image = Image.open(filepath)

        # Force the image to RGB
        if external_image.mode != 'RGB':
            external_image = external_image.convert('RGB')

        # Resize the image
        external_image_resized = external_image.resize((1024, 512))

        # Convert the image to a numpy array and normalize pixel values
        external_image_array = np.array(external_image_resized) / 255.0

        # Ensure the shape is (512, 1024, 3)
        if external_image_array.shape != (512, 1024, 3):
            return jsonify({'error': f'Image shape mismatch: {external_image_array.shape}'}), 400

        # Reshape the image to match the input shape of the age model (add batch dimension)
        external_image_array = external_image_array.reshape(1, 512, 1024, 3)

        # Make age prediction
        prediction = age_model.predict(external_image_array)

        # Define age range class labels
        age_labels = [
            "14-16 Years", "8-10 Years", "11-13 Years", "4-7 Years", "17-21 Years"
        ]

        # Get prediction probabilities for age ranges
        prediction_probs = prediction[0]
        age_prediction_details = {
            age_labels[i]: prediction_probs[i] * 100 for i in range(len(age_labels))
        }

        # Determine the predicted age range
        result = prediction.argmax(axis=1)[0]
        result_label = age_labels[result]

        return jsonify({
            'age_class_result': result_label,
            'age_class_probabilities': age_prediction_details,
            'filename': filepath
        })

    except Exception as e:
        return jsonify({'error': f'Error processing image for age prediction: {str(e)}'}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use PORT from environment or default to 5000
    app.run(host='0.0.0.0', port=port, debug=True)  # Bind to all interfaces