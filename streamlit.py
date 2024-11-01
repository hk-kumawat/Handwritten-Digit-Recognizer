import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import numpy as np
import cv2

# Load the pre-trained model
model = tf.keras.models.load_model('mnist_model_enhanced.h5')

# Title and instructions
st.title("Handwritten Digit Recognition")
st.write("Draw a digit in the box below, and the model will predict it.")

# Create a canvas for user input
canvas_result = st_canvas(
    stroke_width=10,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Check if there is valid data in the canvas
if canvas_result.image_data is not None:
    if np.sum(canvas_result.image_data) > 0:
        try:
            # Preprocess the image
            img = canvas_result.image_data.astype('uint8')  # Convert to uint8
            img = cv2.resize(img, (28, 28))  # Resize to 28x28
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            img = img / 255.0  # Normalize to [0,1]
            img = np.expand_dims(img, axis=(0, -1))  # Reshape to (1, 28, 28, 1)

            # Perform prediction
            prediction = model.predict(img)
            predicted_class = np.argmax(prediction, axis=1)[0]
            confidence = np.max(prediction) * 100

            # Display results
            st.write(f"**Predicted Digit:** {predicted_class}")
            st.write(f"**Confidence Level:** {confidence:.2f}%")
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
    else:
        st.warning("Please draw a digit on the canvas to get a prediction.")
else:
    st.warning("No input detected on the canvas. Please try drawing a digit.")
