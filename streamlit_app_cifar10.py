import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# CIFAR-10 class labels
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

st.set_page_config(page_title="CIFAR-10 Image Classifier", layout="centered")
st.title("🧠 CIFAR-10 Image Classifier")
st.markdown("Upload an image and get the predicted CIFAR-10 class.")

# Load or define the model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("cifar10_model.h5")
    return model

model = load_model()

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    img = image.resize((32, 32))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = 100 * np.max(prediction)

    st.markdown(f"### 🎯 Prediction: **{predicted_class}**")
    st.markdown(f"Confidence: `{confidence:.2f}%`")

else:
    st.info("Upload an image to classify.")

