<a id="readme-top"></a>

# Handwritten Digit Recognizer✍🏻🔢
![sddefault](https://github.com/user-attachments/assets/7319a7b4-0ccd-4af0-b577-942fbd389da9)


## Overview

The Handwritten Digit Recognizer project uses a Convolutional Neural Network (CNN) model to classify handwritten digits (0–9) based on the popular MNIST dataset. The project also features an interactive **Streamlit** application where users can draw a digit and get a prediction from the enhanced model.

<br>

## Live Demo

Try out the Handwritten Digit Recognizer! 👉🏻 [![Experience It! 🌟](https://img.shields.io/badge/Experience%20It!-blue)](https://handwrittendigitpredictor.streamlit.app/)

<br>

 _Below is a preview of the Handwritten Digit Recognizer in action. Draw a digit and see the AI predict it in real-time!_ 👇🏻

<p align="center">
  <img src="https://github.com/user-attachments/assets/efda2ef5-790e-401b-abb0-b4322088a3f9" alt="house">
</p>


<br>

## Learning Journey 🗺️

This project represents a deep dive into the world of computer vision and deep learning. Here's the story behind it:

- **Inspiration:**  
  The MNIST dataset is often called the "Hello World" of machine learning. I wanted to take this classical problem and transform it into an interactive, real-world application that anyone could use and understand.

- **Why I Made It:**  
  Beyond learning the technical aspects of CNN architecture and image processing, I wanted to create something that bridges the gap between complex machine learning concepts and user-friendly applications.

- **Challenges Faced:**  
  - **Model Architecture:** Finding the right balance between model complexity and performance required extensive experimentation with different CNN architectures.
  - **Data Preprocessing:** Ensuring consistent input processing between training data and user-drawn digits was crucial for accurate predictions.
  - **Real-time Performance:** Optimizing the model to provide instant predictions while maintaining accuracy presented interesting technical challenges.
  - **UI/UX Design:** Creating an intuitive drawing interface that works across different devices required careful consideration of user experience.

- **What I Learned:**  
  - **Deep Learning:** Hands-on experience with CNN architecture design and training optimization
  - **Data Augmentation:** Techniques to improve model robustness using ImageDataGenerator
  - **Web Development:** Building interactive web applications with Streamlit
  - **UI/UX Design:** Creating an intuitive and responsive user interface
  - **Model Deployment:** Practical experience in deploying machine learning models in a web application

<br>


## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Technologies Used](#technologies-used)
5. [Dataset](#dataset)
6. [Data Preprocessing](#data-preprocessing)
7. [Model Training](#model-training)
8. [Evaluation](#evaluation)
9. [Results](#results)
10. [Directory Structure](#directory-structure)
11. [Contributing](#contributing)
12. [License](#license)
13. [Contact](#contact)

<br>

## Features🌟

- Real-time digit recognition on a Streamlit interface with a drawable canvas.
- Provides predictions for handwritten digits (0–9).
- Utilizes a CNN model trained on 70,000 MNIST dataset images, ensuring reliable performance in digit classification.

<br>

## Installation🛠

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/hk-kumawat/Handwritten-Digit-Recognizer.git
   cd Handwritten-Digit-Recognizer
   ```

2. **Create & Activate a Virtual Environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate       # On Windows: venv\Scripts\activate
   ```

3. **Install Required Packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Open in Dev Container:**
   - Use the provided `.devcontainer/devcontainer.json` for a pre-configured development environment.

<br>


## Usage🚀

### Running the Streamlit App

Start the interactive digit recognizer:
```bash
streamlit run streamlit.py
```
**Features include:**
- A drawable canvas for handwriting input.
- Real-time digit prediction with confidence levels.
- A bar chart displaying prediction probabilities.

<br>

### Running the Jupyter Notebook

Explore model training and evaluation:
1. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook "Handwritten_Digit_Recognition.ipynb"
   ```
2. **Execute cells** to follow the model building, training, evaluation, and visualization steps.

<br>


## Technologies Used💻

- **Programming Language:**  
  - `Python`

- **Deep Learning:**  
  - `TensorFlow`
  - `Keras`

- **Web Framework:**  
  - `Streamlit`

- **Data Handling & Visualization:**  
  - `NumPy`
  - `Pandas`
  - `Matplotlib`
  - `Seaborn`
  - `cv2` (OpenCV)

- **Utilities:**  
  - `streamlit_drawable_canvas`

<br>


## Dataset📊

The project uses the **MNIST Dataset**, which includes:
- `60,000` training images
- `10,000` test images
- 28x28 grayscale images
- 10 classes (digits 0-9)

<br>


## Data Preprocessing🔄

- **Normalization:**  
  Pixel values are scaled to the [0,1] range.
- **Reshaping:**  
  Data is reshaped to fit the CNN input requirements (28x28x1).
- **Augmentation:**  
  An `ImageDataGenerator` is used to apply rotations, shifts, and zooming to improve model robustness.

<br>


## Model Training🧠

- **Architecture:**  
  A CNN with two convolutional layers, max pooling, dropout, and dense layers.
- **Compilation:**  
  Optimized using `adam` with `sparse_categorical_crossentropy` loss.
- **Training Parameters:**  
  - Batch size: `60`  
  - Epochs: `10`  
  - Validation split: `30%`
- **Enhanced Training:**  
  Data augmentation is used along with early stopping to improve model performance further.

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

<br>

### Accuracy and Loss Plots
The plots below illustrate the training and validation accuracy and loss over the epochs, demonstrating the model's performance improvement throughout the training process.

![Screenshot 2024-11-02 205054](https://github.com/user-attachments/assets/e38c8e13-10a5-48eb-8b52-52f7ad9849e1)



<br>


## Results🏆

### Model Performance

- **Training Accuracy:** `97.06%`
- **Validation Accuracy:** `99.30%`
- **Test Accuracy:** `98.94%`

### System Performance

- **Average Prediction Time:** <1 second
- **Memory Usage:** ~500MB
- **Canvas Resolution:** 280x280 pixels
  
<br> 


## Directory Structure📁

```plaintext
hk-kumawat-handwritten-digit-recognizer/
├── README.md                      # Project documentation
├── Handwritten_Digit_Recognition.ipynb  # Notebook for model exploration & training
├── LICENSE                        # License information
├── mnist_model.h5                 # Saved base model
├── mnist_model_enhanced.h5        # Saved enhanced model with data augmentation
├── requirements.txt               # List of dependencies
├── streamlit.py                   # Streamlit app for interactive digit prediction
└── .devcontainer/
    └── devcontainer.json          # Configuration for development container
```

<br>


## Contributing🤝
Contributions make the open source community such an amazing place to learn, inspire, and create. 🙌 Any contributions you make are greatly appreciated! 😊

Have an idea to improve this project? Go ahead and fork the repo to create a pull request, or open an issue with the tag **"enhancement"**. Don't forget to give the project a star! ⭐ Thanks again! 🙏

<br>

1. **Fork** the repository.

2. **Create** a new branch:
   ```bash
   git checkout -b feature/YourFeatureName
   ```

3. **Commit** your changes with a descriptive message.

4. **Push** to your branch:
   ```bash
   git push origin feature/YourFeatureName
   ```

5. **Open** a Pull Request detailing your enhancements or bug fixes.

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


## Thanks for exploring—happy predicting! 🔢

> "In the world of AI, every digit tells a story, and every prediction opens a new chapter." – Anonymous

<p align="right">
  (<a href="#readme-top">back to top</a>)
</p>
