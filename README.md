# AI Digit Classifier – Handwritten Digit Recognition

A machine learning–based handwritten digit recognition system built using the
MNIST dataset. This project compares classical machine learning models
(Support Vector Machine and Random Forest) with a deep learning model
(Convolutional Neural Network) and deploys the trained models using a
Streamlit web application for real-time inference.

---

## Project Overview

Handwritten digit recognition is a core problem in computer vision with
applications in document digitization, bank cheque processing, postal mail
sorting, and automated data entry systems.

This project aims to design, evaluate, and deploy an accurate handwritten digit
classification system capable of recognizing digits from **0 to 9** using
multiple machine learning approaches.

---

## Application Demo

Below is a screenshot of the Streamlit application demonstrating real-time
handwritten digit classification using pretrained models.

![Streamlit App Running](assets/sample_running.png)

---

## Dataset

- **Dataset:** MNIST Handwritten Digits
- **Total Samples:** 70,000 images  
  - 60,000 training images  
  - 10,000 testing images
- **Image Size:** 28 × 28 pixels
- **Type:** Grayscale
- **Classes:** Digits (0–9)

The dataset is a standard benchmark and is not included in this repository.

---

## Methodology

###  Data Preprocessing
- Grayscale conversion and normalization
- Image resizing to 28×28 pixels
- Pixel value scaling
- Flattening for SVM and Random Forest
- Tensor reshaping for CNN input

###  Models Implemented
- **Support Vector Machine (SVM):** Linear and RBF kernels
- **Random Forest:** Ensemble of decision trees
- **Convolutional Neural Network (CNN):** Convolution, pooling, and dense layers

---

##  Experimental Results

| Model           | Accuracy | Precision | Recall | F1-score |
|-----------------|----------|-----------|--------|----------|
| SVM             | 93.51%   | 93.51%    | 93.51% | 93.50%   |
| Random Forest   | 96.75%   | 96.75%    | 96.75% | 96.75%   |
| CNN             | 98.70%   | 98.50%    | 98.60% | 98.60%   |

**Observation:**  
CNN achieves the highest accuracy due to its ability to automatically learn
spatial features from handwritten digit images.

---

## Result Discussion

- **SVM:** Performs well for linearly separable data but struggles with complex
  handwritten variations.
- **Random Forest:** Improves accuracy through ensemble learning and feature
  randomness.
- **CNN:** Outperforms classical models by capturing spatial hierarchies in image
  data.

Common misclassifications occur between visually similar digits such as
`1 vs 7`, `3 vs 8`, and `5 vs 6`.

---
## Project Structure

```text
AI-Digit-Classifier/
│
├── app.py                     # Streamlit inference application
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation
├── projectarchitecture.md     # System architecture details
├── projectSummary.md          # High-level project summary
│
├── models/                    # Trained machine learning models
│   ├── cnn_model.h5           # CNN model
│   ├── rf_model.pkl           # Random Forest model
│   └── svm_model.pkl          # SVM model
│
├── notebooks/                 # Training and experimentation
│   ├── AI_Digit_Classifier.ipynb
│   └── AI_Digit_Classifier.py
│
├── sample_inputs/             # Sample handwritten digit images
├── assets/                    # UI assets and images
└── venv_tf/                   # Local virtual environment
```


---

## How to Run the Application

```bash
# Create and activate virtual environment (Python 3.10 recommended)
python -m venv venv_tf
venv_tf\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```
## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- TensorFlow / Keras
- OpenCV
- Streamlit
- Jupyter Notebook

## 📁 Project Structure

