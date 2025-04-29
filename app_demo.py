from flask import Flask, render_template, request, jsonify, send_from_directory, make_response
from flask_cors import CORS
import os
import uuid
import time
from PIL import Image
from werkzeug.utils import secure_filename
import random

app = Flask(__name__)
CORS(app)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

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
        # Simulate processing delay
        time.sleep(1)

        # Mock prediction results
        gender_labels = ["Female", "Male"]
        
        # Randomly select a gender for demo purposes
        random_index = random.randint(0, 1)
        result_label = gender_labels[random_index]
        
        # Generate mock probabilities
        if random_index == 0:  # Female
            female_prob = random.uniform(60, 95)
            male_prob = 100 - female_prob
        else:  # Male
            male_prob = random.uniform(60, 95)
            female_prob = 100 - male_prob
            
        gender_prediction_details = {
            "Female": float(female_prob),
            "Male": float(male_prob)
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
        # Simulate processing delay
        time.sleep(0.5)

        # Mock prediction results
        age_labels = [
            "14-16 Years", "8-10 Years", "11-13 Years", "4-7 Years", "17-21 Years"
        ]

        # Randomly select an age range for demo purposes
        random_index = random.randint(0, len(age_labels) - 1)
        result_label = age_labels[random_index]
        
        # Generate mock probabilities
        probabilities = [random.uniform(5, 20) for _ in range(len(age_labels))]
        # Make the selected age range have a higher probability
        probabilities[random_index] = random.uniform(40, 85)
        
        # Normalize probabilities to sum to 100
        total = sum(probabilities)
        probabilities = [p * 100 / total for p in probabilities]
        
        age_prediction_details = {
            age_labels[i]: float(probabilities[i]) for i in range(len(age_labels))
        }

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
    port = int(os.environ.get('PORT', 8080))
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=True) 