# 🍎 Rotten Apple Detection using VGG16 Transfer Learning

An end-to-end Computer Vision and Deep Learning project that classifies apples as either **Good Apple** or **Rotten Apple** using Transfer Learning with the **VGG16** architecture in TensorFlow/Keras.

---

## 📋 Table of Contents
* [Project Overview](#-project-overview)
* [Dataset](#-dataset)
* [Model Architecture](#-model-architecture)
* [Project Structure](#-project-structure)
* [Installation & Usage](#-installation--usage)
* [Training & Results](#-training--results)
* [Inference & Visualization](#-inference--visualization)

---

## 🔍 Project Overview
Early detection of spoiled or rotten produce is vital for quality control in agricultural supply chains and automated sorting systems. This project utilizes a pre-trained **VGG16** convolutional neural network to extract robust visual features and fine-tunes a custom classification head to accurately distinguish between fresh and rotten apples.

---

## 📂 Dataset
* **Source:** Downloaded programmatically via the Roboflow API (`rotten-apples-detection` dataset, Version 5).
* **Classes:** 
  1. `Good Apple`
  2. `Rotten Apple`
* **Splits:** Divided into Training and Validation directories.
* **Preprocessing:** Images are resized to `224x224` pixels and augmented using rotation, zoom, and horizontal flips to prevent overfitting.

---

## 🧠 Model Architecture
* **Base Model:** VGG16 (pre-trained on ImageNet with frozen convolutional layers to retain lower-level feature extractors).
* **Custom Classification Head:**
  * `Flatten` Layer
  * `Dense` Layer (128 units, ReLU activation)
  * `Dropout` Layer (Rate: 0.5 for regularization)
  * `Dense` Output Layer (Softmax/Categorical activation corresponding to 2 classes)
* **Optimizer:** Adam
* **Loss Function:** Categorical Crossentropy

---

## 📁 Project Structure
```text
deep_learning/
│
├── VGG_16_Transfer_Learning_in_Keras.ipynb   # Main Google Colab Jupyter Notebook
└── README.md                                 # Project Documentation
