import tensorflow as tf
from tensorflow.keras import models, layers

# Define our character set with added punctuation
CHARACTERS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,;:?!-\'"`()[]{}@#$%&*+'
NUM_CLASSES = len(CHARACTERS)

def create_model():
    """
    Creates a neural network for handwritten text recognition.
    Enhanced to handle both letters, numbers, and punctuation marks.
    """
    model = models.Sequential([
        # Input layer
        layers.InputLayer(input_shape=(28, 28, 1)),
        # Replace experimental.preprocessing.Rescaling with Lambda layer
        layers.Lambda(lambda x: x / 255.0),
        
        # Convolutional layers with increased complexity
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        
        # Flatten the 2D features
        layers.Flatten(),
        
        # Dense layers for classification
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),  # Prevent overfitting
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Output layer with one neuron per character
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    # Compile the model with improved settings
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def char_to_index(char):
    """Convert a character to its index in our character set."""
    return CHARACTERS.find(char)

def index_to_char(index):
    """Convert an index to its corresponding character."""
    if 0 <= index < len(CHARACTERS):
        return CHARACTERS[index]
    return '?'

def get_character_set():
    """Return the current character set being used."""
    return CHARACTERS 