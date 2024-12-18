# Handwritten Digit Recognizer✍🏻🔢
![sddefault](https://github.com/user-attachments/assets/7319a7b4-0ccd-4af0-b577-942fbd389da9)


## Overview

The Handwritten Digit Recognizer project uses a Convolutional Neural Network (CNN) model to classify handwritten digits (0–9) based on the popular MNIST dataset. With a user-friendly interface, users can draw digits and receive real-time predictions, making it suitable for applications like digitizing notes and automated number plate recognition.

## Live Demo

Try out the Handwritten Digit Recognizer! 👉🏻 [![Experience It! 🌟](https://img.shields.io/badge/Experience%20It!-blue)](https://handwrittendigitpredictor.streamlit.app/)

<br>

 _Below is a preview of the Handwritten Digit Recognizer in action. Draw a digit to see its prediction!_ 👇🏻

<p align="center">
  <img src="https://github.com/user-attachments/assets/efda2ef5-790e-401b-abb0-b4322088a3f9" alt="house">
</p>

<br>

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
11. [License](#license)
12. [Contact](#contact)

<br>

## Features🌟

- Real-time digit recognition on a Streamlit interface with a drawable canvas.
- Provides predictions for handwritten digits (0–9).
- Utilizes a CNN model trained on 70,000 MNIST dataset images, ensuring reliable performance in digit classification.

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

- **Model Architecture**: The model is based on a Convolutional Neural Network (CNN), designed with multiple layers to effectively extract features from handwritten digit images.

- **Training Process**: The model is trained using the MNIST dataset, incorporating data augmentation techniques to enhance the diversity of training samples. Early stopping is also utilized to prevent overfitting, ensuring improved generalization to unseen data.

<br>

## Evaluation📈

The model is evaluated using the following metrics:

- **Accuracy**: Measures the overall performance of the model by calculating the ratio of correct predictions to the total number of predictions. The model achieved a training accuracy of approximately `97.06%` and a validation accuracy of approximately   `99.30%`, indicating excellent performance on both the training and validation datasets.

- **Loss**: Tracks the model's error during training and testing, utilizing categorical cross-entropy for this multi-class classification task. The training loss decreased to around `0.0985`, while the validation loss reached about `0.0221`, suggesting that the model is effectively learning from the data and generalizing well to unseen examples.

### Training Summary:
- **Epochs**: `10`
- **Final Training Accuracy**: `97.06%`
- **Final Validation Accuracy**: `99.30%`
- **Final Training Loss**: `0.0985`
- **Final Validation Loss**: `0.0221`

### Accuracy and Loss Plots
The plots below illustrate the training and validation accuracy and loss over the epochs, demonstrating the model's performance improvement throughout the training process.

- **Model Accuracy and Loss**:

![Screenshot 2024-11-02 205054](https://github.com/user-attachments/assets/e38c8e13-10a5-48eb-8b52-52f7ad9849e1)



<br>

## Installation🛠

1. **Clone the repository**:
   ```bash
   https://github.com/hk-kumawat/Handwritten-Digit-Recognizer.git
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   
<br>


## Usage🚀

1. **Train the Model**: 
   - Run the Jupyter Notebook to train the CNN model on the MNIST dataset.
   - The initial model is saved as `mnist_model.h5`, which achieved a certain accuracy.

2. **Enhanced Model**:
   - After further enhancements, a refined model is saved as `mnist_model_enhanced.h5`, yielding improved accuracy.

3. **Model Inference**:
   - Run the `streamlit.py` file to start the Streamlit app.
   - Draw a digit on the canvas and click "Predict" to see the model's prediction.


<br>


## Technologies Used💻

- **Programming Language**: Python
- **Libraries**: 
  - `pandas`
  - `numpy`
  - `tensorflow`
  - `keras`
  - `streamlit`
  - `streamlit_drawable_canvas`
  - `seaborn`
  - `opencv-python`
  - `matplotlib`
- **Deployment**: Streamlit for a web-based user interface


<br>


## Results🏆

The model demonstrated impressive accuracy on the **MNIST test dataset**, achieving approximately `99.30%` validation accuracy. By leveraging a robust **Convolutional Neural Network (CNN)** architecture, it effectively predicts handwritten digits with minimal error. The model's performance was validated through metrics such as **accuracy** and **loss**, showcasing its ability to generalize well to unseen data.

<br>

## Conclusion📚

The Handwritten Digit Recognizer demonstrates the effectiveness of CNNs in image recognition tasks. With a simple user interface, the project showcases how AI can transform handwritten input into digital data, supporting applications from education to automated systems.

<br> 

## License📝

This project is licensed under the **MIT License** - see the [LICENSE](./LICENSE) file for details.

<br>

## Contact

### 📬 Get in Touch!
I’d love to hear from you! Feel free to reach out:

- [![GitHub](https://img.shields.io/badge/GitHub-hk--kumawat-blue?logo=github)](https://github.com/hk-kumawat) 💻 — Explore my projects and contributions.
- [![LinkedIn](https://img.shields.io/badge/LinkedIn-Harshal%20Kumawat-blue?logo=linkedin)](https://www.linkedin.com/in/harshal-kumawat/) 🌐 — Let’s connect professionally.
- [![Email](https://img.shields.io/badge/Email-harshalkumawat100@gmail.com-blue?logo=gmail)](mailto:harshalkumawat100@gmail.com) 📧 — Send me an email for any in-depth discussions.

<br>

---


## Thanks for exploring this project! 🔢

> "Turning each stroke of creativity into a recognizable digit, one digit at a time." – Anonymous

