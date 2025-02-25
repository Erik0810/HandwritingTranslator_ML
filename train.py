import tensorflow as tf
import numpy as np
from model.neural_network import create_model, char_to_index, CHARACTERS, get_character_set
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Import sklearn for dataset loading
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def load_existing_model():
    """Try to load an existing model if it exists."""
    model_path = 'model/handwriting_model.h5'
    if os.path.exists(model_path):
        print("Loading existing model...")
        return tf.keras.models.load_model(model_path)
    print("No existing model found. Creating new model...")
    return create_model()

def prepare_custom_data(custom_images, custom_labels):
    """Prepare custom data for training."""
    # Convert to numpy arrays if they aren't already
    images = np.array(custom_images)
    labels = np.array([char_to_index(label) for label in custom_labels])
    
    # Ensure images are in the right format (28x28 pixels)
    if images.shape[1:] != (28, 28):
        raise ValueError("Images must be 28x28 pixels")
    
    # Reshape for CNN (no need to normalize as it's handled in the model)
    return images.reshape(-1, 28, 28, 1), labels

def load_training_data():
    """Load and combine multiple datasets for comprehensive character recognition."""
    print("Loading training datasets...")
    
    # Load MNIST dataset for digits using sklearn
    print("Loading MNIST dataset for digits...")
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    mnist_images = mnist.data.to_numpy().reshape(-1, 28, 28, 1) / 255.0
    mnist_labels = np.array([char_to_index(str(int(label))) for label in mnist.target])
    
    # Create synthetic data for letters and symbols
    print("Generating synthetic data for letters and symbols...")
    
    # Generate synthetic data for punctuation and symbols
    def generate_synthetic_symbols(num_samples=1000):
        synthetic_images = []
        synthetic_labels = []
        
        # Get indices of special characters in our character set
        special_chars_indices = [i for i, c in enumerate(CHARACTERS) 
                               if not c.isalnum()]  # Get indices of non-alphanumeric chars
        
        base_shapes = {
            '.': lambda: np.zeros((28, 28, 1)),  # Will add a small circle
            '!': lambda: np.zeros((28, 28, 1)),  # Will add a vertical line with dot
            '?': lambda: np.zeros((28, 28, 1)),  # Will add question mark shape
            # Add more base shapes for other symbols
        }
        
        print("Generating synthetic symbol data...")
        for _ in range(num_samples):
            # Randomly select a special character
            char_idx = np.random.choice(special_chars_indices)
            char = CHARACTERS[char_idx]
            
            # Create base image (either from predefined or random)
            if char in base_shapes:
                img = base_shapes[char]()
                # Add the specific symbol shape
                if char == '.':
                    rr, cc = np.ogrid[13:16, 13:16]
                    img[rr, cc] = 1.0  # Normalized to [0,1]
                elif char == '!':
                    img[5:20, 13:15] = 1.0
                    img[22:24, 13:15] = 1.0
                # Add more symbol patterns here
            else:
                # Create a random pattern for other symbols
                img = np.random.rand(28, 28, 1)  # Random values between 0 and 1
            
            synthetic_images.append(img)
            synthetic_labels.append(char_idx)
        
        return np.array(synthetic_images), np.array(synthetic_labels)
    
    # Generate synthetic data for letters (A-Z, a-z)
    def generate_synthetic_letters(num_samples=5000):
        synthetic_images = []
        synthetic_labels = []
        
        # Get indices of letter characters in our character set
        letter_indices = [i for i, c in enumerate(CHARACTERS) if c.isalpha()]
        
        print("Generating synthetic letter data...")
        for _ in range(num_samples):
            # Randomly select a letter
            char_idx = np.random.choice(letter_indices)
            char = CHARACTERS[char_idx]
            
            # Create a simple representation of the letter
            img = np.zeros((28, 28, 1))
            
            # Add some basic shapes based on the letter
            if char.lower() in ['a', 'b', 'c', 'd', 'e', 'o']:
                # Draw a circle-like shape for round letters
                center_x, center_y = 14, 14
                radius = 8
                for i in range(28):
                    for j in range(28):
                        if ((i - center_y) ** 2 + (j - center_x) ** 2) < radius ** 2:
                            img[i, j] = 0.8
                            
                # Add vertical line for b, d, etc.
                if char.lower() in ['b', 'd']:
                    line_x = 6 if char.lower() == 'b' else 22
                    img[5:23, line_x-1:line_x+1] = 1.0
            else:
                # For other letters, create more abstract patterns
                # Vertical line for letters like i, l, t
                if char.lower() in ['i', 'l', 't', 'j', 'f']:
                    img[5:23, 13:15] = 1.0
                # Horizontal lines for letters like e, f, t
                if char.lower() in ['e', 'f', 't']:
                    img[10:12, 8:20] = 1.0
                # Diagonal for letters like k, x, z
                if char.lower() in ['k', 'x', 'z']:
                    for i in range(10, 20):
                        img[i, i] = 1.0
                        img[i, 28-i] = 1.0
            
            # Add some noise to make it more realistic
            img += np.random.rand(28, 28, 1) * 0.2
            img = np.clip(img, 0, 1)  # Ensure values stay in [0,1]
            
            synthetic_images.append(img)
            synthetic_labels.append(char_idx)
        
        return np.array(synthetic_images), np.array(synthetic_labels)
    
    # Generate synthetic data for symbols and letters
    symbols_x, symbols_y = generate_synthetic_symbols()
    letters_x, letters_y = generate_synthetic_letters()
    
    # Combine MNIST and synthetic data
    x_all = np.concatenate([mnist_images, symbols_x, letters_x])
    y_all = np.concatenate([mnist_labels, symbols_y, letters_y])
    
    # Split into train, validation, and test sets
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x_all, y_all, test_size=0.2, random_state=42
    )
    
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val, y_train_val, test_size=0.2, random_state=42
    )
    
    print(f"Training data: {len(x_train)} samples")
    print(f"Validation data: {len(x_val)} samples")
    print(f"Test data: {len(x_test)} samples")
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def plot_training_history(history):
    """Plot training history."""
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def main(custom_images=None, custom_labels=None):
    # Print available characters
    print(f"Training model to recognize these {len(get_character_set())} characters:")
    print(get_character_set())
    
    # Load training data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_training_data()
    print(f"Training data shape: {x_train.shape}")
    print(f"Validation data shape: {x_val.shape}")
    print(f"Test data shape: {x_test.shape}")

    # If we have custom data, add it to the training set
    if custom_images is not None and custom_labels is not None:
        print("Adding custom training data...")
        custom_x, custom_y = prepare_custom_data(custom_images, custom_labels)
        x_train = np.concatenate([x_train, custom_x])
        y_train = np.concatenate([y_train, custom_y])
        print(f"Added {len(custom_images)} custom training examples")

    # Load or create the model
    model = load_existing_model()

    # Train the model with improved parameters
    print("Training model...")
    history = model.fit(
        x_train, 
        y_train,
        epochs=15,
        batch_size=64,
        validation_data=(x_val, y_val),
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            )
        ]
    )

    # Evaluate the model
    print("\nEvaluating model...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"\nTest accuracy: {test_accuracy:.4f}")

    # Plot training history
    plot_training_history(history)

    # Save the model
    print("\nSaving model...")
    model.save('model/handwriting_model.h5')
    print("Model saved successfully!")

    return history

if __name__ == "__main__":
    # Example of how to use custom data:
    # custom_images = [...] # Your additional 28x28 pixel images
    # custom_labels = ['A', '.', '5', '?', ...]  # Any character from the character set
    main() 