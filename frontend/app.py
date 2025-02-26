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
import json

# Add the parent directory to the path so we can import from model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.neural_network import index_to_char

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'webp'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variable to store the model
model = None

# Try to import SocketIO, but provide a fallback if not available
try:
    from flask_socketio import SocketIO
    socketio = SocketIO(app, cors_allowed_origins="*")
    socketio_available = True
    print("SocketIO imported successfully")
except ImportError:
    socketio_available = False
    print("WARNING: flask_socketio not installed. Real-time progress updates will not be available.")
    # Create dummy functions to prevent errors
    class DummySocketIO:
        def emit(self, event, data):
            print(f"[DUMMY SOCKETIO] Would emit {event}: {data}")
    socketio = DummySocketIO()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_model():
    """Load the trained model."""
    global model
    
    # Use absolute path to ensure correct location
    import os
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(script_dir, 'model', 'handwriting_model.h5')
    
    print(f"Looking for model at: {model_path}")
    
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}...")
        try:
            model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    else:
        print(f"No model found at {model_path}. Please train the model first.")
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
        # Import the training script with callback support
        from train import train_with_callbacks
        
        # Define callback function to send progress updates
        def progress_callback(epoch, logs):
            # Calculate progress within epoch (0-100%)
            batch_size = logs.get('size', 1)
            total_samples = logs.get('total_samples', 1000)
            batch_index = logs.get('batch', 0)
            
            # Calculate progress percentage within the current epoch
            epoch_progress = min(100, (batch_index * batch_size * 100) // total_samples)
            
            # Send update via WebSocket (or print if not available)
            update_data = {
                'epoch': epoch,
                'total_epochs': logs.get('total_epochs', 10),
                'epoch_progress': epoch_progress,
                'accuracy': logs.get('accuracy', 0) * 100,  # Convert to percentage
                'loss': logs.get('loss', 0),
                'time_elapsed': logs.get('time_elapsed', 0),
                'time_remaining': logs.get('time_remaining', 0)
            }
            
            socketio.emit('training_update', update_data)
        
        # Run the training with progress callbacks
        success, message = train_with_callbacks(progress_callback)
        
        # Send final update to complete the progress bar
        socketio.emit('training_complete', {
            'success': success,
            'message': message
        })
        
        # Check if model file exists after training
        import os
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(script_dir, 'model', 'handwriting_model.h5')
        model_exists = os.path.isfile(model_path)
        
        print(f"After training, model file exists: {model_exists} at {model_path}")
        
        # Try to load the model regardless of the training result
        model_loaded = load_model()
        
        if success and model_loaded:
            return jsonify({'success': True, 'message': 'Model trained and loaded successfully'})
        elif success and model_exists:
            return jsonify({'success': True, 'message': 'Model trained successfully but could not be loaded'})
        elif success:
            return jsonify({'success': False, 'message': 'Model trained but file was not saved correctly'})
        else:
            return jsonify({'success': False, 'message': message})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error training model: {str(e)}'})

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and image processing."""
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'})
        
    if file and allowed_file(file.filename):
        try:
            # Read the image file
            img = Image.open(file)
            
            # Convert WEBP to PNG if necessary
            if file.filename.lower().endswith('.webp'):
                # Convert to RGB mode if necessary (in case of RGBA webp)
                if img.mode in ('RGBA', 'LA'):
                    background = Image.new('RGB', img.size, 'white')
                    background.paste(img, mask=img.split()[-1])
                    img = background
                
                # Create a byte buffer for the PNG
                buf = io.BytesIO()
                img.save(buf, format='PNG')
                buf.seek(0)
                
                # Create a new filename
                filename = secure_filename(file.filename.rsplit('.', 1)[0] + '.png')
            else:
                # For non-webp files, just get the original filename
                filename = secure_filename(file.filename)
                buf = file.stream
            
            # Save the processed image
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img.save(filepath)
            
            # Process the image for character recognition
            characters, error = preprocess_image(filepath)
            
            if error:
                return jsonify({'success': False, 'message': error})
            
            # Recognize text
            text = recognize_text(characters)
            
            return jsonify({'success': True, 'text': text})
            
        except Exception as e:
            return jsonify({'success': False, 'message': f'Error processing image: {str(e)}'})
            
    return jsonify({'success': False, 'message': 'File type not allowed'})

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

@app.route('/debug', methods=['GET'])
def debug_info():
    """Return debug information about the model and directories."""
    import os
    
    # Get absolute paths
    app_dir = os.path.abspath(os.path.dirname(__file__))
    parent_dir = os.path.dirname(app_dir)
    model_dir = os.path.join(parent_dir, 'model')
    model_path = os.path.join(model_dir, 'handwriting_model.h5')
    
    # Check if directories and files exist
    model_dir_exists = os.path.isdir(model_dir)
    model_file_exists = os.path.isfile(model_path)
    
    # List files in model directory if it exists
    model_dir_files = []
    if model_dir_exists:
        model_dir_files = os.listdir(model_dir)
    
    # Check if model is loaded in memory
    model_loaded_in_memory = model is not None
    
    # Return debug info
    debug_info = {
        'app_directory': app_dir,
        'parent_directory': parent_dir,
        'model_directory': model_dir,
        'model_path': model_path,
        'model_directory_exists': model_dir_exists,
        'model_file_exists': model_file_exists,
        'files_in_model_directory': model_dir_files,
        'model_loaded_in_memory': model_loaded_in_memory
    }
    
    return jsonify(debug_info)

@app.route('/socketio-check', methods=['GET'])
def socketio_check():
    """Check if SocketIO is available."""
    return jsonify({'available': socketio_available})

if __name__ == '__main__':
    # Try to load the model at startup
    load_model()
    app.run(debug=True) 