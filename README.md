# CIFAR-10 Image Classifier

This project is an end-to-end image classifier using the CIFAR-10 dataset, trained using TensorFlow/Keras, and deployed as a Streamlit app with live webcam and image upload functionality.

---

## 🚀 Features

- ✅ Train a Convolutional Neural Network (CNN) on CIFAR-10 dataset.
- ✅ Save and load trained model.
- ✅ Streamlit app to make live predictions.
- ✅ Upload image from your device to classify.
- ✅ Take a photo directly from webcam and classify it.
- ✅ Deployed-ready version compatible with Streamlit Cloud.

---

## 📦 Project Structure

.
├── model_training.py # Script to train and save the model
├── cifar10_model/ # Saved model directory (SavedModel format)
├── streamlit_app.py # Streamlit app for prediction
├── requirements.txt # Python dependencies
├── README.md # This file

The [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) contains 60,000 32x32 color images in 10 classes:

- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

---

## ⚙ Model Architecture

We trained a simple CNN using TensorFlow:

- Input: 32x32 color images
- Layers:
  - Conv2D + ReLU + MaxPooling
  - Dropout (to reduce overfitting)
  - Flatten + Dense layers
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy

Training was performed using:

```bash
python model_training.py