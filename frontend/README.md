# Handwriting Recognition Web Application

This is a simple web application that uses a machine learning model to recognize handwritten text from images.

## Features

- Train the handwriting recognition model directly from the web interface
- Capture images using your device's camera
- Upload images from your computer
- Process images to extract handwritten text
- View the recognized text in real-time

## Setup and Installation

1. Make sure you have Python 3.8+ installed
2. Install the required packages:
   ```
   pip install -r ../requirements.txt
   ```
3. Run the Flask application:
   ```
   python app.py
   ```
4. Open your browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

## Usage

1. **First-time setup**: Click the "Train Model" button to train the handwriting recognition model. This may take a few minutes.
2. **Using the camera**: 
   - Click the "Capture Image" button to take a photo
   - Click "Process Image" to recognize the text
3. **Uploading an image**:
   - Switch to the "Upload Image" tab
   - Select an image file containing handwritten text
   - Click "Upload & Process" to recognize the text
4. The recognized text will appear in the "Recognized Text" section

## Requirements

- Modern web browser with camera access (for camera functionality)
- Images should have clear, well-contrasted handwriting
- Works best with dark text on a light background

## Troubleshooting

- If the camera doesn't work, make sure you've granted camera permissions to the website
- If text recognition is poor, try using an image with better lighting and contrast
- Make sure the handwriting is clear and characters are separated 