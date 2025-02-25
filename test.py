import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from model.neural_network import index_to_char, CHARACTERS

def load_and_predict():
    # Load the saved model
    print("Loading trained model...")
    model = tf.keras.models.load_model('model/handwriting_model.h5')

    # Load EMNIST test data
    print("Loading test data...")
    (_, _), (x_test, y_test) = tf.keras.datasets.emnist.load_data(type='byclass')
    
    # EMNIST images are rotated 90 degrees and flipped
    x_test = np.rot90(np.flip(x_test, axis=1), k=3, axes=(1,2))
    
    # Filter to keep only the characters we want
    test_mask = y_test < len(CHARACTERS)
    x_test = x_test[test_mask]
    y_test = y_test[test_mask]
    
    # Normalize and reshape
    x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

    # Select a random test image
    test_index = np.random.randint(0, len(x_test))
    test_image = x_test[test_index]
    true_label = index_to_char(y_test[test_index])

    # Make prediction
    prediction = model.predict(np.expand_dims(test_image, axis=0))
    predicted_char = index_to_char(np.argmax(prediction))

    # Get top 3 predictions
    top_3_indices = np.argsort(prediction[0])[-3:][::-1]
    top_3_chars = [index_to_char(idx) for idx in top_3_indices]
    top_3_probs = [prediction[0][idx] for idx in top_3_indices]

    # Display results
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(test_image.reshape(28, 28), cmap='gray')
    plt.title(f'Prediction: {predicted_char}\nTrue Label: {true_label}')
    plt.axis('off')

    # Add a bar chart of top 3 predictions
    plt.subplot(1, 2, 2)
    plt.bar(top_3_chars, top_3_probs)
    plt.title('Top 3 Predictions')
    plt.ylim(0, 1)
    plt.show()

if __name__ == "__main__":
    load_and_predict() 