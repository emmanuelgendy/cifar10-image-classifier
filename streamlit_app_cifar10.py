import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cifar10_model.h5")

model = load_model()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

st.title("CIFAR-10 Image Classifier")

option = st.radio("Choose input method:", ["Upload image", "Use webcam"])

if option == "Upload image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
elif option == "Use webcam":
    picture = st.camera_input("Take a picture")
    if picture:
        image = Image.open(picture)

if 'image' in locals():
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    img = image.resize((32, 32))
    img = np.array(img) / 255.0
    if img.shape[-1] == 4:
        img = img[:, :, :3]
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    top_5 = np.argsort(prediction[0])[::-1][:5]

    st.subheader("Predictions:")
    for i in top_5:
        st.write(f"{class_names[i]}: {prediction[0][i] * 100:.2f}%")