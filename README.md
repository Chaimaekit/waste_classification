# Waste & Garbage Classifier CNN 

A Deep Learning project that classifies images of waste items into six standard recycling categories using a Convolutional Neural Network (CNN) with **Transfer Learning (MobileNetV2)**.

---

## Project Overview
Proper waste sorting is essential for recycling efficiency and environmental sustainability. This project automates the classification of waste items into distinct categories using computer vision. By leveraging **MobileNetV2** pre-trained on ImageNet, the model achieves high accuracy with low computational overhead and fast training times.

### Target Categories (6 Classes)
* **Cardboard**
* **Glass**
* **Metal**
* **Paper**
* **Plastic**
* **Trash (General Waste)**

---

## Project Structure

```text
garbage_classifier/
│
├── garbage_dataset/           # Dataset containing image subdirectories
│   ├── cardboard/
│   ├── glass/
│   ├── metal/
│   ├── paper/
│   ├── plastic/
│   └── trash/
│
├── main.py                    # Main script for data pipeline & model training
├── requirements.txt           # Project dependencies
├── .gitignore                 # Files to ignore in Git version control
└── README.md                  # Project documentation
```

---

## 🛠️ Technology Stack
* **Language:** Python 3.x
* **Deep Learning Framework:** TensorFlow / Keras
* **Pre-trained Architecture:** MobileNetV2
* **Data & Plotting:** NumPy, Matplotlib

---

## Getting Started

### 1. Prerequisites & Installation
Ensure you have Python installed. Clone the repository and install the required packages:

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Setup
Place your dataset folder in the project root directory and name it `garbage_dataset`. Ensure it follows the subdirectory structure shown above.

### 3. Training the Model
Run the main script to train the model locally:

```bash
python main.py
```

---

## Model Pipeline & Architecture

1. **Data Preprocessing & Augmentation:**
   * Images are automatically resized to **$150 \times 150$ pixels**.
   * Batch size is set to **32**.
   * Data is split into **80% Training** and **20% Validation**.
   * Performance optimizations (`cache` and `prefetch`) are applied using `tf.data`.

2. **Model Architecture:**
   * **Rescaling Layer:** Scales pixel values from $[0, 255]$ to $[-1, 1]$.
   * **Base Model:** Pre-trained `MobileNetV2` with frozen weights to preserve pre-learned feature extraction capabilities.
   * **Pooling Layer:** `GlobalAveragePooling2D` to reduce dimensionality.
   * **Regularization:** `Dropout(0.2)` to prevent overfitting.
   * **Output Layer:** `Dense(6, activation='softmax')` for multi-class classification probabilities.

3. **Compilation:**
   * **Optimizer:** Adam
   * **Loss Function:** `sparse_categorical_crossentropy`
   * **Metrics:** Accuracy

---

## Results & Evaluation
After running `main.py`, training history plots (Accuracy and Loss over epochs) are automatically displayed using Matplotlib to monitor convergence and check for overfitting.