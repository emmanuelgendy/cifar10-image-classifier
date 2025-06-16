# CIFAR-10 Image Classifier (with Streamlit App)

This is an end-to-end deep learning project that classifies images from the CIFAR-10 dataset using a Convolutional Neural Network (CNN), with a live **Streamlit app** to upload and test your own images.

---

## Project Overview

Over the past few days, this project has gone from a basic CNN to a live, interactive image classification app:

- Built & trained a CNN model on CIFAR-10
- Improved accuracy with Dropout & tuning
- Visualized training history (accuracy/loss)
- Created a webcam-ready Streamlit interface
- Prepared for cloud deployment via Streamlit Cloud

---

## Project Structure

```bash
.
├── streamlit_app.py           # Streamlit interface to classify images
├── cifar10_cnn.py             # Model training script
├── plot_training.py           # Training accuracy/loss visualization
├── training_history.pkl       # Saved training history for plotting
├── cifar10_model.keras        # Trained CNN model
├── requirements.txt           # Python dependencies
├── runtime.txt                # Python version for Streamlit Cloud
└── README.md                  # Project overview
