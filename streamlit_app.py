import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("cifar10_model")  # SavedModel format folder
    return model

model = load_model()

st.title("CIFAR-10 Image Classifier")

# User selects input method
option = st.radio("Choose input method:", ("Upload Image", "Use Webcam"))

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

elif option == "Use Webcam":
    image_data = st.camera_input("Take a picture")
    if image_data is not None:
        image = Image.open(image_data)
        st.image(image, caption='Webcam Image', use_column_width=True)

# Run prediction if image exists
if 'image' in locals():
    # Preprocess image
    image = image.resize((32, 32))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]

    st.write(f"Prediction: **{predicted_class}**")
