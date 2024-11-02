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
            if prediction is not None and prediction.size > 0:
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
