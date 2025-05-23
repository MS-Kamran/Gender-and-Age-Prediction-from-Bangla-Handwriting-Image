from flask import Flask, render_template, request, jsonify, send_from_directory, make_response
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os
import uuid
import time
from PIL import Image
from werkzeug.utils import secure_filename
from functools import lru_cache

app = Flask(__name__)
CORS(app)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load models only once when the app starts
print("Loading models...")
start_time = time.time()

# Configure TensorFlow for better performance
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

# Load models with optimization
try:
    # Load the gender model
    gender_model = load_model('model/AP-AksharNet_1024Gender_Trained.h5', compile=False)
    gender_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Load the age model
    age_model = load_model('model/AP-AksharNet_1024Age_Trained.h5', compile=False)
    age_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print(f"Models loaded successfully in {time.time() - start_time:.2f} seconds")
except Exception as e:
    print(f"Error loading models: {str(e)}")
    raise

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@lru_cache(maxsize=32)
def preprocess_image(file_path):
    """
    Preprocess image with caching for improved performance
    """
    try:
        # Open the image
        img = Image.open(file_path)

        # Force the image to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize the image
        img_resized = img.resize((1024, 512))

        # Convert to numpy array and normalize
        img_array = np.array(img_resized) / 255.0

        # Ensure the shape is correct
        if img_array.shape != (512, 1024, 3):
            return None, f'Image shape mismatch: {img_array.shape}'

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        return img_array, None
    except Exception as e:
        return None, f'Error preprocessing image: {str(e)}'

def save_uploaded_file(file):
    """
    Save uploaded file with a unique filename
    """
    if not file or not allowed_file(file.filename):
        return None, 'Invalid file or file type not allowed'
    
    # Generate a unique filename to prevent conflicts
    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4()}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    
    try:
        file.save(filepath)
        return filepath, None
    except Exception as e:
        return None, f'Error saving file: {str(e)}'

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    response = make_response(send_from_directory('static', path))
    
    # Set cache control headers
    if path.startswith('uploads/'):
        # Don't cache user uploads
        response.headers['Cache-Control'] = 'no-store'
    elif path.endswith(('.css', '.js')):
        # Cache CSS and JS files for 1 week
        response.headers['Cache-Control'] = 'public, max-age=604800'
    elif path.endswith(('.jpg', '.jpeg', '.png', '.gif', '.svg', '.ico')):
        # Cache images for 1 month
        response.headers['Cache-Control'] = 'public, max-age=2592000'
    else:
        # Default cache of 1 day for other static files
        response.headers['Cache-Control'] = 'public, max-age=86400'
    
    return response

@app.route('/robots.txt')
def serve_robots():
    return send_from_directory('static', 'robots.txt')

@app.route('/predict_gender', methods=['POST'])
def predict_gender():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save the uploaded file
    filepath, error = save_uploaded_file(file)
    if error:
        return jsonify({'error': error}), 400

    try:
        # Preprocess the image
        img_array, error = preprocess_image(filepath)
        if error:
            return jsonify({'error': error}), 400

        # Make prediction
        prediction = gender_model.predict(img_array, verbose=0)

        # Define gender class labels
        gender_labels = ["Female", "Male"]
        result = prediction.argmax(axis=1)[0]
        result_label = gender_labels[result]

        # Get probabilities
        prediction_probs = prediction[0]
        gender_prediction_details = {
            gender_labels[i]: float(prediction_probs[i] * 100) for i in range(len(gender_labels))
        }

        # Create the relative filepath for frontend display
        display_filepath = filepath.replace('\\', '/').split('static/')[-1]
        
        return jsonify({
            'gender_result': result_label,
            'gender_probabilities': gender_prediction_details,
            'filename': f'static/{display_filepath}'
        })

    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500


@app.route('/predict_age', methods=['POST'])
def predict_age():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save the uploaded file
    filepath, error = save_uploaded_file(file)
    if error:
        return jsonify({'error': error}), 400

    try:
        # Preprocess the image
        img_array, error = preprocess_image(filepath)
        if error:
            return jsonify({'error': error}), 400

        # Make prediction
        prediction = age_model.predict(img_array, verbose=0)

        # Define age class labels
        age_labels = [
            "14-16 Years", "8-10 Years", "11-13 Years", "4-7 Years", "17-21 Years"
        ]

        # Get prediction probabilities
        prediction_probs = prediction[0]
        age_prediction_details = {
            age_labels[i]: float(prediction_probs[i] * 100) for i in range(len(age_labels))
        }

        # Get the most likely age range
        result = prediction.argmax(axis=1)[0]
        result_label = age_labels[result]

        # Create the relative filepath for frontend display
        display_filepath = filepath.replace('\\', '/').split('static/')[-1]
        
        return jsonify({
            'age_class_result': result_label,
            'age_class_probabilities': age_prediction_details,
            'filename': f'static/{display_filepath}'
        })

    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
