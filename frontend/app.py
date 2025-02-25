import os
import sys
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image

# Add the parent directory to the path so we can import from model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.neural_network import index_to_char

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variable to store the model
model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_model():
    """Load the trained model."""
    global model
    model_path = '../model/handwriting_model.h5'
    
    if os.path.exists(model_path):
        print("Loading existing model...")
        model = tf.keras.models.load_model(model_path)
        return True
    else:
        print("No model found. Please train the model first.")
        return False

def preprocess_image(image_path):
    """Preprocess the image for the model."""
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        return None, "Failed to read image"
    
    # Invert the image (black background to white background)
    img = cv2.bitwise_not(img)
    
    # Threshold to get binary image
    _, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    
    # Find contours to segment characters
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours from left to right
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    
    # Extract and process each character
    characters = []
    for contour in contours:
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter out very small contours (noise)
        if w < 10 or h < 10:
            continue
        
        # Extract the character
        char_img = img[y:y+h, x:x+w]
        
        # Resize to 28x28 with padding to maintain aspect ratio
        target_size = 28
        aspect_ratio = w / h
        
        if aspect_ratio > 1:
            # Width is larger
            new_w = target_size
            new_h = int(target_size / aspect_ratio)
            pad_top = (target_size - new_h) // 2
            pad_bottom = target_size - new_h - pad_top
            pad_left = pad_right = 0
        else:
            # Height is larger
            new_h = target_size
            new_w = int(target_size * aspect_ratio)
            pad_left = (target_size - new_w) // 2
            pad_right = target_size - new_w - pad_left
            pad_top = pad_bottom = 0
        
        # Resize
        char_img = cv2.resize(char_img, (new_w, new_h))
        
        # Pad
        char_img = cv2.copyMakeBorder(
            char_img, 
            pad_top, pad_bottom, pad_left, pad_right, 
            cv2.BORDER_CONSTANT, 
            value=0
        )
        
        # Normalize
        char_img = char_img.astype('float32') / 255.0
        
        # Add to list
        characters.append(char_img.reshape(28, 28, 1))
    
    if not characters:
        return None, "No characters detected in the image"
    
    return characters, None

def recognize_text(characters):
    """Recognize text from preprocessed characters."""
    if model is None:
        return "Model not loaded"
    
    text = ""
    for char_img in characters:
        # Make prediction
        prediction = model.predict(np.expand_dims(char_img, axis=0), verbose=0)
        predicted_idx = np.argmax(prediction)
        predicted_char = index_to_char(predicted_idx)
        text += predicted_char
    
    return text

@app.route('/')
def index():
    """Render the main page."""
    model_loaded = model is not None
    return render_template('index.html', model_loaded=model_loaded)

@app.route('/train', methods=['POST'])
def train_model():
    """Run the training script."""
    try:
        # Import the training script
        from train import main as train_main
        
        # Run the training
        train_main()
        
        # Load the trained model
        load_model()
        
        return jsonify({'success': True, 'message': 'Model trained successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error training model: {str(e)}'})

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload."""
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the image
        characters, error = preprocess_image(filepath)
        
        if error:
            return jsonify({'success': False, 'message': error})
        
        # Recognize text
        text = recognize_text(characters)
        
        return jsonify({'success': True, 'text': text})
    
    return jsonify({'success': False, 'message': 'Invalid file type'})

@app.route('/camera', methods=['POST'])
def process_camera():
    """Process image from camera."""
    if 'image' not in request.form:
        return jsonify({'success': False, 'message': 'No image data'})
    
    # Get the base64 image data
    image_data = request.form['image'].split(',')[1]
    
    # Decode the image
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    
    # Save the image
    filename = f"camera_{int(tf.timestamp())}.jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(filepath)
    
    # Process the image
    characters, error = preprocess_image(filepath)
    
    if error:
        return jsonify({'success': False, 'message': error})
    
    # Recognize text
    text = recognize_text(characters)
    
    return jsonify({'success': True, 'text': text})

if __name__ == '__main__':
    # Try to load the model at startup
    load_model()
    app.run(debug=True) 