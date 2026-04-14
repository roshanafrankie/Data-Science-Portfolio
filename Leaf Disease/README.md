# 🌿 Plant Leaf Disease Detection

This project implements a **Deep Learning** solution to identify 25 different health states across various crops, including Apple, Corn, Grape, Potato, and Tomato.

## 🚀 Project Overview
The core of this project is a **Convolutional Neural Network (CNN)** trained to recognize specific visual patterns caused by fungal, bacterial, and viral pathogens. This automation can assist in early disease detection, reducing crop loss and pesticide overuse.

## 🛠️ Tech Stack
- **Deep Learning:** TensorFlow / Keras
- **Image Processing:** OpenCV, PIL
- **Data Analysis:** NumPy, Matplotlib, Seaborn
- **Model Storage:** JSON (Architecture) & H5 (Weights)

## ⚙️ How It Works
1. **Preprocessing**: Images are resized to `128x128` pixels to match the model's input layer.
2. **Normalization**: Pixel values are scaled to a range of `[0, 1]` for better numerical stability.
3. **Inference**: The model outputs a probability distribution across 25 classes using a Softmax activation.
4. **Classification**: The script uses `np.argmax()` to select the class with the highest confidence score.

## 📊 Evaluation
The model's performance was validated using a confusion matrix to track precision and recall across all categories.
![Confusion Matrix](Confusion_Matrix.png)
