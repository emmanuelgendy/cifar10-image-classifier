import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

st.title("CIFAR-10 Image Classifier")

# Load pre-trained model (using TensorFlow Keras applications)
@st.cache_resource
def load_model():
    model = tf.keras.applications.MobileNetV2(weights=None, input_shape=(32, 32, 3), classes=10)
    # For simplicity, we just simulate a trained model (in real case, load trained weights)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model

model = load_model()

# Upload image
uploaded_file = st.file_uploader("Choose a CIFAR-10 image (32x32)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((32, 32))
    st.image(image, caption='Uploaded Image', use_column_width=False)

    # Preprocess image
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction (dummy prediction here since model is untrained)
    prediction = np.random.rand(10)  # Random predictions for demo purposes
    predicted_class = class_names[np.argmax(prediction)]

    st.write(f"Predicted Class: **{predicted_class}**")

    # Display probabilities
    st.bar_chart(prediction)
