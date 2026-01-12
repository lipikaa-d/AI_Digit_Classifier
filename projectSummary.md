# Project Summary – AI Digit Classifier

The **AI Digit Classifier** is a machine learning–based system developed to recognize handwritten digits from 0 to 9 using image data. The project focuses on building, evaluating, and deploying an accurate digit recognition solution using both classical machine learning techniques and deep learning models.

The system is trained and evaluated on the **MNIST handwritten digit dataset**, a widely used benchmark in computer vision research. Three different models are implemented and compared: **Support Vector Machine (SVM)**, **Random Forest**, and **Convolutional Neural Network (CNN)**. Each model represents a different learning approach, enabling a comprehensive comparison of traditional and deep learning methods for image classification tasks.

The project workflow begins with data preprocessing, which includes grayscale normalization, image resizing to 28×28 pixels, and feature preparation based on the selected model. Classical models such as SVM and Random Forest operate on flattened pixel features, while the CNN model processes structured image tensors to automatically learn spatial patterns.

Model performance is evaluated using multiple metrics, including accuracy, precision, recall, and F1-score. Experimental results show that the CNN model achieves the highest classification accuracy due to its ability to extract hierarchical spatial features from handwritten digit images. Random Forest demonstrates strong performance through ensemble learning, while SVM provides reliable baseline results for linear separability.

To demonstrate real-world applicability, the trained models are deployed using a **Streamlit-based web application**. The application allows users to upload handwritten digit images, select a classification model, and obtain predictions in real time. This deployment-focused design highlights the practical use of machine learning models beyond experimentation.

Overall, the project provides a complete end-to-end implementation of a handwritten digit recognition system, covering data preprocessing, model training, evaluation, and deployment. It reflects real-world machine learning development practices and demonstrates the comparative strengths of classical and deep learning approaches in computer vision tasks.
