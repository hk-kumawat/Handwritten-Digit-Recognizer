<div align="center">

# Handwritten Digit Recognizer

### Deep Learning-Powered Digit Recognition with Interactive Web Interface

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Live Demo](https://img.shields.io/badge/Demo-Live-success.svg)](https://handwrittendigitpredictor.streamlit.app/)

[Live Demo](https://handwrittendigitpredictor.streamlit.app/) • [Report Bug](https://github.com/hk-kumawat/Handwritten-Digit-Recognizer/issues) • [Request Feature](https://github.com/hk-kumawat/Handwritten-Digit-Recognizer/issues)

<img src="https://github.com/user-attachments/assets/7319a7b4-0ccd-4af0-b577-942fbd389da9" alt="Handwritten Digit Recognizer Banner" width="600"/>

</div>

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Live Demo](#live-demo)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
  - [Web Application](#web-application)
  - [Jupyter Notebook](#jupyter-notebook)
- [Technical Documentation](#technical-documentation)
  - [Model Architecture](#model-architecture)
  - [Dataset](#dataset)
  - [Data Preprocessing](#data-preprocessing)
  - [Training Configuration](#training-configuration)
- [Performance Metrics](#performance-metrics)
- [Project Structure](#project-structure)
- [Technologies Stack](#technologies-stack)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

---

## Overview

The **Handwritten Digit Recognizer** is a production-ready deep learning application that leverages Convolutional Neural Networks (CNN) to classify handwritten digits (0-9) with **99.30% validation accuracy**. Built on the renowned MNIST dataset, this project demonstrates end-to-end machine learning pipeline implementation—from data preprocessing and model training to deployment via an interactive web interface.

### What Makes This Project Special

- **High Accuracy**: Achieves 99.30% validation accuracy through optimized CNN architecture
- **Real-Time Predictions**: Instant digit recognition with sub-second response time
- **Interactive Interface**: Draw digits directly in your browser with Streamlit-powered UI
- **Production Ready**: Complete with model persistence, deployment configuration, and containerization support
- **Educational Value**: Comprehensive documentation and Jupyter notebooks for learning

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Interactive Canvas** | Browser-based drawing interface with 280×280 pixel resolution |
| **Real-Time Recognition** | Instant predictions with confidence scores and probability distributions |
| **Dual Models** | Base model and enhanced model with data augmentation |
| **Visualization** | Dynamic bar charts showing prediction probabilities for all digits |
| **Deployment Ready** | Includes Docker configuration and Streamlit deployment setup |
| **Comprehensive Metrics** | Detailed accuracy, loss tracking, and confusion matrix analysis |

---

## Live Demo

Experience the application in action: **[Launch Live Demo](https://handwrittendigitpredictor.streamlit.app/)**

<div align="center">
  <img src="https://github.com/user-attachments/assets/efda2ef5-790e-401b-abb0-b4322088a3f9" alt="Application Demo" width="700"/>
  <p><i>Interactive digit drawing interface with real-time prediction</i></p>
</div>

---

## Quick Start

Get up and running in under 3 minutes:

```bash
# Clone the repository
git clone https://github.com/hk-kumawat/Handwritten-Digit-Recognizer.git
cd Handwritten-Digit-Recognizer

# Install dependencies
pip install -r requirements.txt

# Launch the application
streamlit run streamlit.py
```

Open your browser at `http://localhost:8501` and start drawing digits!

---

## Installation

### Prerequisites

Ensure you have the following installed on your system:

- **Python**: Version 3.8 or higher
- **pip**: Latest version recommended
- **Git**: For cloning the repository
- **Virtual Environment** (recommended): `venv` or `conda`

### Step-by-Step Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/hk-kumawat/Handwritten-Digit-Recognizer.git
cd Handwritten-Digit-Recognizer
```

#### 2. Create Virtual Environment (Recommended)

**Using venv:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Using conda:**
```bash
conda create -n digit-recognizer python=3.8
conda activate digit-recognizer
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Verify Installation

```bash
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
```

### Alternative: Development Container

For a pre-configured development environment, use the included Dev Container:

```bash
# Open in VS Code with Dev Containers extension
code .
# Select "Reopen in Container" when prompted
```

---

## Usage

### Web Application

Launch the interactive Streamlit application:

```bash
streamlit run streamlit.py
```

**Application Features:**
- **Drawing Canvas**: Use your mouse or touchscreen to draw digits
- **Prediction Display**: View the predicted digit with confidence percentage
- **Probability Chart**: Visual representation of prediction probabilities for all 10 digits
- **Clear Function**: Reset the canvas to try again

**Tips for Best Results:**
- Draw digits centered in the canvas
- Use bold, continuous strokes
- Ensure adequate size (not too small)
- Draw digits similar to handwritten style (not printed)

### Jupyter Notebook

Explore the complete model development process:

```bash
jupyter notebook Handwritten_Digit_Recognition.ipynb
```

**Notebook Contents:**
1. Data loading and exploration
2. Data preprocessing and augmentation
3. Model architecture design
4. Training with visualization
5. Performance evaluation
6. Prediction testing
7. Model export

---

## Technical Documentation

### Model Architecture

The CNN architecture employs a proven design optimized for image classification:

```
Input Layer (28×28×1)
    ↓
Conv2D (32 filters, 3×3, ReLU)
    ↓
MaxPooling2D (2×2)
    ↓
Conv2D (64 filters, 3×3, ReLU)
    ↓
MaxPooling2D (2×2)
    ↓
Flatten
    ↓
Dense (128 units, ReLU)
    ↓
Dropout (0.5)
    ↓
Dense (10 units, Softmax)
    ↓
Output (10 classes)
```

**Architecture Specifications:**

| Layer | Output Shape | Parameters | Activation |
|-------|-------------|------------|------------|
| Conv2D-1 | (26, 26, 32) | 320 | ReLU |
| MaxPooling2D-1 | (13, 13, 32) | 0 | - |
| Conv2D-2 | (11, 11, 64) | 18,496 | ReLU |
| MaxPooling2D-2 | (5, 5, 64) | 0 | - |
| Flatten | (1600) | 0 | - |
| Dense-1 | (128) | 204,928 | ReLU |
| Dropout | (128) | 0 | - |
| Dense-2 | (10) | 1,290 | Softmax |

**Total Parameters**: ~225,000

### Dataset

**MNIST Database of Handwritten Digits**

| Attribute | Specification |
|-----------|--------------|
| **Training Samples** | 60,000 images |
| **Test Samples** | 10,000 images |
| **Image Dimensions** | 28×28 pixels |
| **Color Space** | Grayscale (1 channel) |
| **Classes** | 10 (digits 0-9) |
| **Format** | Normalized pixel values [0, 1] |

### Data Preprocessing

**Preprocessing Pipeline:**

1. **Normalization**
   ```python
   X = X / 255.0  # Scale pixel values to [0, 1]
   ```

2. **Reshaping**
   ```python
   X = X.reshape(-1, 28, 28, 1)  # Add channel dimension
   ```

3. **Data Augmentation** (Enhanced Model)
   - Rotation: ±10 degrees
   - Width/Height Shift: ±10%
   - Zoom: ±10%
   - Horizontal Flip: Disabled (preserves digit orientation)

### Training Configuration

**Base Model:**

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Loss Function | Sparse Categorical Crossentropy |
| Batch Size | 60 |
| Epochs | 10 |
| Validation Split | 30% |
| Learning Rate | Default (0.001) |

**Enhanced Model** (with augmentation):

| Parameter | Value |
|-----------|-------|
| Data Augmentation | Enabled |
| Early Stopping | Enabled (patience: 3) |
| Model Checkpoint | Best validation loss |
| Additional Regularization | Dropout (0.5) |

---

## Performance Metrics

### Model Performance

#### Accuracy Metrics

| Metric | Base Model | Enhanced Model |
|--------|-----------|----------------|
| **Training Accuracy** | 97.06% | 98.20% |
| **Validation Accuracy** | 99.30% | 99.30% |
| **Test Accuracy** | 98.94% | 98.94% |

#### Loss Metrics

| Metric | Base Model | Enhanced Model |
|--------|-----------|----------------|
| **Training Loss** | 0.0985 | 0.0654 |
| **Validation Loss** | 0.0221 | 0.0221 |

### System Performance

| Metric | Value |
|--------|-------|
| **Prediction Time** | <100ms |
| **Model Size** | 2.8 MB |
| **Memory Usage** | ~500 MB |
| **Inference FPS** | 10+ predictions/second |

### Training Progress

<div align="center">
  <img src="https://github.com/user-attachments/assets/e38c8e13-10a5-48eb-8b52-52f7ad9849e1" alt="Training Metrics" width="700"/>
  <p><i>Model accuracy and loss progression across training epochs</i></p>
</div>

**Key Observations:**
- Rapid convergence within first 3 epochs
- Minimal overfitting (validation > training accuracy)
- Stable performance after epoch 5
- Excellent generalization to unseen data

---

## Project Structure

```
Handwritten-Digit-Recognizer/
│
├── README.md                              # Comprehensive project documentation
├── LICENSE                                # MIT License
├── requirements.txt                       # Python dependencies
│
├── streamlit.py                           # Streamlit web application
├── Handwritten_Digit_Recognition.ipynb   # Jupyter notebook with training pipeline
│
├── mnist_model.h5                         # Trained base model (Keras format)
├── mnist_model_enhanced.h5                # Enhanced model with augmentation
│
└── .devcontainer/
    └── devcontainer.json                  # VS Code dev container configuration
```

### File Descriptions

| File | Purpose |
|------|---------|
| `streamlit.py` | Main application interface with drawing canvas and prediction logic |
| `Handwritten_Digit_Recognition.ipynb` | Complete training pipeline, EDA, and model evaluation |
| `mnist_model.h5` | Saved weights for base CNN model |
| `mnist_model_enhanced.h5` | Improved model with data augmentation training |
| `requirements.txt` | Package dependencies with version specifications |
| `.devcontainer/` | Docker-based development environment configuration |

---

## Technologies Stack

### Core Technologies

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.8+ |
| **Deep Learning** | TensorFlow 2.x, Keras |
| **Web Framework** | Streamlit |
| **Numerical Computing** | NumPy, Pandas |
| **Visualization** | Matplotlib, Seaborn |
| **Computer Vision** | OpenCV (cv2) |
| **UI Components** | streamlit-drawable-canvas |

### Development Tools

- **Jupyter Notebook**: Interactive model development
- **Git**: Version control
- **Docker**: Containerization (Dev Container)
- **VS Code**: Recommended IDE

### Dependencies

```
tensorflow>=2.8.0
streamlit>=1.20.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
opencv-python>=4.5.0
streamlit-drawable-canvas>=0.9.0
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue: Import Errors

**Problem**: `ModuleNotFoundError: No module named 'tensorflow'`

**Solution**:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Issue: Model File Not Found

**Problem**: `OSError: Unable to open file (unable to open file: name = 'mnist_model_enhanced.h5')`

**Solution**:
Ensure you're running commands from the project root directory:
```bash
cd Handwritten-Digit-Recognizer
streamlit run streamlit.py
```

#### Issue: Canvas Not Displaying

**Problem**: Drawing canvas doesn't appear in Streamlit

**Solution**:
```bash
pip install --upgrade streamlit streamlit-drawable-canvas
streamlit cache clear
```

#### Issue: Low Prediction Accuracy

**Problem**: Model predictions are inconsistent

**Solution**:
- Draw digits centered in the canvas
- Use bold, continuous strokes
- Ensure digit size is adequate (not too small)
- Clear the canvas completely between attempts

#### Issue: TensorFlow GPU Not Detected

**Problem**: Model training is slow, GPU not utilized

**Solution**:
```bash
# Verify GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Install GPU-enabled TensorFlow
pip install tensorflow-gpu==2.8.0
```

### Performance Optimization

If experiencing slow performance:

1. **Reduce Image Size**: Lower canvas resolution in `streamlit.py`
2. **Use CPU Optimized TensorFlow**: Install `tensorflow-cpu`
3. **Close Background Applications**: Free up system memory
4. **Enable Model Quantization**: Reduce model size (advanced)

---

## FAQ

### General Questions

**Q: What is the minimum hardware requirement?**
A:
- Processor: Dual-core CPU (2.0 GHz+)
- RAM: 4 GB minimum, 8 GB recommended
- Storage: 500 MB free space
- GPU: Optional (for training acceleration)

**Q: Can I retrain the model with custom data?**
A: Yes! Use the Jupyter notebook (`Handwritten_Digit_Recognition.ipynb`) and replace the MNIST dataset with your custom images. Ensure images are 28×28 grayscale format.

**Q: How long does training take?**
A:
- CPU: ~10-15 minutes (10 epochs)
- GPU: ~2-3 minutes (10 epochs)

**Q: Is this suitable for production deployment?**
A: Yes, the Streamlit app can be deployed on platforms like Streamlit Cloud, Heroku, AWS, or Google Cloud. The live demo runs on Streamlit Cloud.

### Technical Questions

**Q: Why are there two model files?**
A:
- `mnist_model.h5`: Base model without augmentation
- `mnist_model_enhanced.h5`: Improved model trained with data augmentation for better generalization

**Q: Can I use this for recognizing handwritten text?**
A: This model is specifically trained for single digits (0-9). For text recognition, you'd need an OCR model like Tesseract or a sequence-to-sequence model.

**Q: How does the model handle rotated digits?**
A: The enhanced model includes rotation augmentation (±10°) during training, providing some rotation invariance. Extreme rotations may reduce accuracy.

**Q: What's the difference between validation and test accuracy?**
A:
- **Validation Accuracy**: Performance on validation set during training (used for hyperparameter tuning)
- **Test Accuracy**: Final performance on completely unseen test data (true generalization metric)

---

## Contributing

Contributions are what make the open-source community an incredible place to learn, innovate, and create. All contributions are **greatly appreciated**!

### How to Contribute

1. **Fork the Repository**
   ```bash
   # Click "Fork" on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/Handwritten-Digit-Recognizer.git
   ```

2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```

3. **Make Your Changes**
   - Write clean, documented code
   - Follow existing code style
   - Add tests if applicable

4. **Commit Your Changes**
   ```bash
   git commit -m "Add: Brief description of your feature"
   ```

5. **Push to Your Branch**
   ```bash
   git push origin feature/AmazingFeature
   ```

6. **Open a Pull Request**
   - Provide a clear description of changes
   - Reference any related issues
   - Wait for code review

### Contribution Guidelines

- **Code Quality**: Follow PEP 8 style guidelines
- **Documentation**: Update README and docstrings
- **Testing**: Ensure existing tests pass
- **Commit Messages**: Use clear, descriptive messages
- **Issues**: Check existing issues before creating new ones

### Areas for Contribution

- Model improvements (architecture, hyperparameters)
- UI/UX enhancements
- Additional features (batch prediction, model comparison)
- Documentation improvements
- Bug fixes and optimizations
- Test coverage expansion

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](./LICENSE) file for complete details.

**Summary**: You are free to use, modify, distribute, and sell this software, provided the original copyright notice and permission notice are included.

---

## Acknowledgments

This project was made possible by:

- **[MNIST Database](http://yann.lecun.com/exdb/mnist/)**: Created by Yann LeCun, Corinna Cortes, and Christopher J.C. Burges
- **[TensorFlow Team](https://www.tensorflow.org/)**: For the outstanding deep learning framework
- **[Streamlit](https://streamlit.io/)**: For the intuitive web application framework
- **[Streamlit Drawable Canvas](https://github.com/andfanilo/streamlit-drawable-canvas)**: By Fanilo Andrianasolo
- **Open Source Community**: For continuous inspiration and support

### Inspiration

> "The MNIST dataset is the 'Hello World' of machine learning. This project transforms that foundational concept into an interactive, real-world application accessible to everyone."

---

## Contact

### Connect With Me

I'd love to hear your feedback, suggestions, or just connect!

<div align="center">

[![GitHub](https://img.shields.io/badge/GitHub-hk--kumawat-181717?style=for-the-badge&logo=github)](https://github.com/hk-kumawat)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Harshal%20Kumawat-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/harshal-kumawat/)
[![Email](https://img.shields.io/badge/Email-harshalkumawat100@gmail.com-EA4335?style=for-the-badge&logo=gmail)](mailto:harshalkumawat100@gmail.com)

</div>

---

<div align="center">

### Thank you for exploring this project!

**If you found this helpful, please consider giving it a ⭐**

<p align="center">
  <a href="#readme-top">↑ Back to Top</a>
</p>

</div>
