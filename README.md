# COVID-19 Detection from Chest X-rays using Deep Learning

## Project Overview

This project is an end-to-end Deep Learning application developed for automatic COVID-19 detection from Chest X-ray images using Convolutional Neural Networks (CNNs) and Transfer Learning techniques.

The objective of this project is to assist healthcare professionals by building an AI-powered diagnostic system capable of identifying COVID-19 infections from chest radiographs.

The project also includes a professional Streamlit web application for real-time image prediction and visualization.

---

# Dataset

Dataset Used:
Chest X-ray COVID-19 Pneumonia Dataset

Classes:

* COVID19
* NORMAL
* PNEUMONIA

Dataset Structure:

train/

* COVID19/
* NORMAL/
* PNEUMONIA/

test/

* COVID19/
* NORMAL/
* PNEUMONIA/

---

# Technologies Used

* Python
* TensorFlow
* Keras
* CNN
* Transfer Learning
* ResNet50
* Streamlit
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* PIL

---

# Models Implemented

## 1. Basic CNN

Architecture:

* Conv2D
* MaxPooling
* Flatten
* Dense Layers

## 2. Transfer Learning using ResNet50

* Pretrained ImageNet weights
* Fine-tuning last layers

## 3. Transfer Learning + Data Augmentation

* Rotation
* Zoom
* Horizontal Flip
* Width/Height Shift

---

# Data Preprocessing

* Image resizing to 224x224
* Pixel normalization
* Train-validation-test split
* Data augmentation
* Label encoding

---

# Model Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* ROC-AUC Score

---

# Final Deployed Model

## Transfer Learning + Data Augmentation Model

Performance:

* Test Accuracy: 91.69%
* F1 Score: 91.57%

Reason for Selection:

* Better generalization
* Stable predictions
* Reduced overfitting
* Suitable for deployment

---

# Streamlit Web Application

The project includes a professional Streamlit dashboard where users can:

* Upload Chest X-ray images
* Predict COVID19 / NORMAL / PNEUMONIA
* View confidence score
* Visualize probability distribution
* Analyze class-wise prediction probabilities

---

# Features

* AI Medical Dashboard UI
* Probability Visualization
* Real-time Predictions
* Interactive Charts
* Deep Learning based Classification
* Transfer Learning using ResNet50

---

# How to Run the Project

## Step 1: Clone Repository

```bash
git clone <repository-link>
```

---

## Step 2: Create Virtual Environment

### Windows

```bash
python -m venv venv
```

---

## Step 3: Activate Virtual Environment

```bash
venv\Scripts\activate
```

---

## Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 5: Run Streamlit Application

```bash
streamlit run app.py
```

---

# Project Structure

COVID19_Streamlit_App/

* app.py
* README.md
* requirements.txt
* best_augmentation_model.keras
* .streamlit/
* venv/

---

# Future Improvements

* Grad-CAM Heatmaps
* EfficientNet/DenseNet Models
* Cloud Deployment
* Multi-image Predictions
* Explainable AI Features
* Medical Report Generation

---

# Conclusion

This project demonstrates how Deep Learning and Transfer Learning can be applied in medical imaging to assist in fast and automated COVID-19 detection using Chest X-ray images.

The deployed AI system successfully performs multi-class chest disease classification and provides an interactive web interface using Streamlit.
