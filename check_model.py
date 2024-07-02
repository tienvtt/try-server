import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model('mnist_cnn_model.h5')

# Example: Perform inference with the model
import numpy as np

# Example input for inference (adjust as needed)
example_input = np.random.rand(1, 28, 28, 1)  # Example input shape: (batch_size, height, width, channels)

# Perform inference
predictions = model.predict(example_input)

print("Example predictions:")
print(predictions)
