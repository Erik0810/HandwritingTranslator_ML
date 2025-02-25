"""
Test script to verify all required packages are installed correctly.
"""
print("Testing imports...")

try:
    import numpy
    print("✓ NumPy imported successfully")
except ImportError as e:
    print(f"✗ Error importing NumPy: {e}")

# TensorFlow should work with the official Python distribution
print("Attempting to import TensorFlow...")
try:
    import tensorflow
    print(f"✓ TensorFlow imported successfully (version {tensorflow.__version__})")
except ImportError as e:
    print(f"✗ Error importing TensorFlow: {e}")
except Exception as e:
    print(f"✗ Error with TensorFlow: {type(e).__name__}: {e}")

try:
    import matplotlib
    print("✓ Matplotlib imported successfully")
except ImportError as e:
    print(f"✗ Error importing Matplotlib: {e}")

try:
    import cv2
    print(f"✓ OpenCV imported successfully (version {cv2.__version__})")
except ImportError as e:
    print(f"✗ Error importing OpenCV: {e}")

try:
    import flask
    print("✓ Flask imported successfully")
except ImportError as e:
    print(f"✗ Error importing Flask: {e}")

try:
    from PIL import Image
    print("✓ Pillow imported successfully")
except ImportError as e:
    print(f"✗ Error importing Pillow: {e}")

try:
    import sklearn
    print("✓ Scikit-learn imported successfully")
except ImportError as e:
    print(f"✗ Error importing Scikit-learn: {e}")

try:
    import werkzeug
    print("✓ Werkzeug imported successfully")
except ImportError as e:
    print(f"✗ Error importing Werkzeug: {e}")

print("\nImport test completed.") 