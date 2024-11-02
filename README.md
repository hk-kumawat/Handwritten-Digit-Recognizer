# Handwritten-Digit-Recognizer✍️

![Handwritten Digit Recognition](https://github.com/user-attachments/assets/handwritten_digit_recognizer.png)

## Overview

The Handwritten Digit Recognizer project uses a Convolutional Neural Network (CNN) model to classify handwritten digits (0–9) based on the popular MNIST dataset. With a user-friendly interface, users can draw digits and receive real-time predictions, making it suitable for applications like digitizing notes and automated number plate recognition.

## Live Demo

Try out the Handwritten Digit Recognizer! 👉🏻 [![Experience It! 🌟](https://img.shields.io/badge/Experience%20It!-blue)](your-streamlit-link)

<br>

- _Below is a preview of the Handwritten Digit Recognizer in action. Draw a digit to see its prediction!_ 👇🏻

![Screenshot](https://github.com/user-attachments/assets/handwritten_digit_recognizer_preview.png)

## Table of Contents

1. [Features](#features)
2. [Dataset](#dataset)
3. [Data Preprocessing](#data-preprocessing)
4. [Model Training](#model-training)
5. [Evaluation](#evaluation)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Technologies Used](#technologies-used)
9. [Results](#results)
10. [Conclusion](#conclusion)
11. [Contact](#contact)

<br>

## Features🌟

- Real-time digit recognition on a Streamlit interface with a drawable canvas.
- Provides predictions for handwritten digits (0–9).
- Utilizes a CNN model trained on 70,000 MNIST dataset images.

<br>

## Dataset📊

- **MNIST Dataset**: Contains 70,000 grayscale images of handwritten digits (0–9), each of size 28x28 pixels.
- **Data Splits**: The dataset is divided into 60,000 training images and 10,000 testing images.

<br>

## Data Preprocessing🛠

1. **Normalization**: Pixel values are scaled between 0 and 1 to optimize CNN performance.
2. **Data Augmentation**: Applied random transformations to enhance model generalization.

<br>

## Model Training🧠

- **Model Architecture**: Convolutional Neural Network (CNN) with layers to extract features from handwritten images.
- **Training**: Model is trained using the MNIST dataset, with data augmentation and early stopping to improve generalization.

<br>

## Evaluation📈

Here's a final version of the Evaluation section for your Handwritten Digit Recognizer README, incorporating training and validation metrics:

---

## Evaluation📈

You can definitely use the latest version provided, as it incorporates a clear explanation of the evaluation metrics along with a summary of the results, making it comprehensive and informative. Here’s a refined version, including the mention of the plots for accuracy and loss:

---

## Evaluation📈

The model is evaluated using the following metrics:

- **Accuracy**: Measures the overall performance of the model by calculating the ratio of correct predictions to the total number of predictions. The model achieved a training accuracy of approximately 97.06% and a validation accuracy of approximately 99.30%, indicating excellent performance on both the training and validation datasets.

- **Loss**: Tracks the model's error during training and testing, utilizing categorical cross-entropy for this multi-class classification task. The training loss decreased to around 0.0985, while the validation loss reached about 0.0221, suggesting that the model is effectively learning from the data and generalizing well to unseen examples.

### Training Summary:
- **Epochs**: 10
- **Final Training Accuracy**: 97.06%
- **Final Validation Accuracy**: 99.30%
- **Final Training Loss**: 0.0985
- **Final Validation Loss**: 0.0221

### Accuracy and Loss Plots
The plots below illustrate the training and validation accuracy and loss over the epochs, demonstrating the model's performance improvement throughout the training process.

- **Model Accuracy and Loss**:

![Screenshot 2024-11-02 205054](https://github.com/user-attachments/assets/e38c8e13-10a5-48eb-8b52-52f7ad9849e1)



<br>

## Installation🛠

1. **Clone the repository**:
   ```bash
   git clone https://github.com/hk-kumawat/handwrittendigitrecognizer.git
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

<br>

## Usage🚀

1. **Train the model**: Run the Jupyter Notebook to train the CNN model on the MNIST dataset and save the trained model as `mnist_model.h5`.
2. **Model Inference**:
   - Run the `streamlit.py` file to start the Streamlit app.
   - Draw a digit on the canvas and click "Predict" to see the model's prediction.

<br>

## Technologies Used💻

- Python
- Libraries: `pandas`, `numpy`, `tensorflow`, `keras`, `streamlit`
- Deployment: Streamlit for a web-based user interface

<br>

## Results🏆

- The model achieved high accuracy on the MNIST test dataset, effectively predicting handwritten digits with a robust CNN architecture.

<br>

## Conclusion📚

The Handwritten Digit Recognizer demonstrates the effectiveness of CNNs in image recognition tasks. With a simple user interface, the project showcases how AI can transform handwritten input into digital data, supporting applications from education to automated systems.

<br>

## Contact

### 📬 Get in Touch!
I’d love to hear from you! Feel free to reach out:

- [![GitHub](https://img.shields.io/badge/GitHub-hk--kumawat-blue?logo=github)](https://github.com/hk-kumawat) 💻 — Explore my projects and contributions.
- [![LinkedIn](https://img.shields.io/badge/LinkedIn-Harshal%20Kumawat-blue?logo=linkedin)](https://www.linkedin.com/in/harshal-kumawat/) 🌐 — Let’s connect professionally.
- [![Email](https://img.shields.io/badge/Email-harshalkumawat100@gmail.com-blue?logo=gmail)](mailto:harshalkumawat100@gmail.com) 📧 — Send me an email for any in-depth discussions.

---

Feel free to adjust this template as needed for your specific setup, especially the demo links, screenshots, and additional project insights!
