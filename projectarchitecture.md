# Project Architecture – AI Digit Classifier

This document describes the architecture of the AI Digit Classifier project and
explains how different components interact to perform handwritten digit
recognition using machine learning and deep learning models.

The architecture is designed to ensure clear separation between experimentation,
model storage, and application-level inference.

---

## High-Level Architecture Overview

The system follows a three-layer architectural design:

1. Experimentation and Training Layer  
2. Model Storage Layer  
3. Application (Inference) Layer  

Each layer is loosely coupled, enabling easier maintenance, scalability, and
reproducibility.

---

## 1. Experimentation and Training Layer

Location: `notebooks/`

This layer is responsible for:
- Data exploration and visualization
- Training and evaluation of models
- Hyperparameter tuning
- Comparative analysis of algorithms

### Components

- `AI_Digit_Classifier.ipynb`  
  Contains exploratory data analysis, visualization of MNIST samples, model
  training, and evaluation using metrics such as accuracy, precision, recall,
  and F1-score.

- `AI_Digit_Classifier.py`  
  Provides a clean, script-based implementation of the training logic and is
  used to save trained models for later inference.

Note: This layer is not accessed during application runtime.

---

## 2. Model Storage Layer

Location: `models/`

This layer stores pretrained models generated during training.

### Stored Models

- `svm_model.pkl` – Support Vector Machine classifier  
- `rf_model.pkl` – Random Forest classifier  
- `cnn_model.h5` – Convolutional Neural Network model  

These models act as reusable artifacts for deployment and inference.

---

## 3. Application (Inference) Layer

Location: `app.py`

This layer provides a user-facing interface and performs real-time inference
using pretrained models.

### Responsibilities

- Accept handwritten digit images uploaded by the user
- Preprocess images to match MNIST input format
- Load the selected machine learning or deep learning model
- Generate predictions and display results

### Inference Workflow

1. The user uploads an image via the Streamlit interface  
2. The image is converted to grayscale and resized to 28×28 pixels  
3. Preprocessing is applied based on the selected model:
   - Flattened feature vectors for SVM and Random Forest
   - Normalized tensor input for CNN
4. The selected model generates a prediction  
5. The predicted digit and confidence score (for CNN) are displayed  

---

## Data Flow Diagram (Conceptual)

## Workflow Overview

```text
MNIST Dataset
      ↓
Training and Evaluation
      ↓
Saved Models (.pkl / .h5)
      ↓
Streamlit Application
      ↓
User Input Image
      ↓
Prediction Output
```

---

## Supporting Directories

### assets/
Contains application screenshots and visual assets used for documentation.

### sample_inputs/
Includes sample handwritten digit images for testing and demonstration.

### venv_tf/
Local Python virtual environment used for dependency isolation.
This directory is excluded from version control.

---

## Design Principles

- Separation of concerns between training and inference
- Reproducibility through preserved training scripts and notebooks
- Modularity to allow easy replacement or extension of models
- Maintainability through clear folder structure and responsibilities

---

## Summary

The project architecture ensures a clean separation between experimentation,
model management, and deployment. This design reflects real-world machine
learning development practices and supports scalability, clarity, and
maintainability.
