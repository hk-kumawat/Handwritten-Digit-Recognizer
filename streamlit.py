import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the model without displaying success message
model_loaded = False
try:
    model = tf.keras.models.load_model('mnist_model_enhanced.h5')
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {e}")

# Proceed only if the model is loaded successfully
if model_loaded:
    # Custom CSS for styling
    st.markdown(
        """
        <style>
        .title {
            font-size: 45px;
            color: #FF6347;
            text-align: center;
            font-weight: bold;
            padding-top: 20px;
        }
        .subtitle {
            font-size: 20px;
            color: #FFFFFF;
            text-align: center;
            margin-bottom: 30px;
        }
        .st-canvas {
            border: 3px solid #FF6347;
            border-radius: 10px;
            margin: 0 auto;
        }
        .prediction-box {
            font-size: 35px;
            color: #32CD32;
            text-align: center;
            font-weight: bold;
            padding: 20px;
            background-color: #F0F8FF;
            border-radius: 10px;
            margin-top: 20px;
        }
        .confidence-text {
            font-size: 18px;
            color: #20B2AA;
            text-align: center;
            font-style: italic;
        }
        .footer {
            font-size: 15px;
            color: #FFFFFF;
            text-align: center;
            margin-top: 50px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Title and instructions with emojis
    st.markdown('<div class="title">🎨 Handwritten Digit Recognition 🔍</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Draw a digit below, and our AI will guess it with confidence! 🤖✨</div>', unsafe_allow_html=True)

    # Create a centered layout for the canvas and predictions
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
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

    # Check for image data on the canvas before proceeding
    if canvas_result.image_data is not None:
        # Check if the canvas is not completely blank
        if np.sum(canvas_result.image_data) > 0:
            try:
                # Preprocess the image from the canvas
                img = canvas_result.image_data
                img = cv2.resize(img, (28, 28))  # Resize to 28x28
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                img = img / 255.0  # Normalize
                img = np.expand_dims(img, axis=(0, -1))  # Reshape to (1, 28, 28, 1)

                st.markdown('<div class="plot-container"><h3>🖼️ Processed Input Image:</h3></div>', unsafe_allow_html=True)
                st.image(img.squeeze(), width=150)

                # Predict the digit
                prediction = model.predict(img)

                # Check if prediction is a valid output
                if prediction.size > 0:
                    predicted_class = np.argmax(prediction, axis=1)[0]
                    confidence = np.max(prediction) * 100

                    # Display the prediction with a custom style
                    st.markdown(f'<div class="prediction-box">Predicted Digit: {predicted_class}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="confidence-text">Confidence Level: {confidence:.2f}%</div>', unsafe_allow_html=True)

                    # Display the prediction probabilities bar chart
                    st.markdown("### 🔢 Prediction Probabilities")
                    fig, ax = plt.subplots()
                    bars = ax.bar(range(10), prediction[0], color="#4682B4", edgecolor="#4682B4")
                    bars[predicted_class].set_color("#32CD32")  # Highlight the predicted digit
                    ax.set_xticks(range(10))
                    ax.set_xlabel("Digit", fontsize=12, fontweight='bold')
                    ax.set_ylabel("Probability", fontsize=12, fontweight='bold')
                    ax.set_title("Model Confidence per Digit", fontsize=16, fontweight='bold', color="#4B0082")
                    st.pyplot(fig)
                else:
                    st.error("Prediction returned empty. Please check the model and input data.")

            except IndexError as e:
                st.error("An error occurred: Tried to access an index that doesn't exist.")
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")
        else:
            st.warning("Please draw a digit on the canvas to get a prediction.")

    # Footer with developer name
    st.markdown('<div class="footer">Developed by - Harshal Kumawat</div>', unsafe_allow_html=True)
