import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Class names for CIFAR-10
class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("cifar10_model.h5")
    return tf.keras.Sequential([model, tf.keras.layers.Softmax()])

model = load_model()

st.title("🖼️ CIFAR-10 Image Classifier")
st.write("Upload an image and the model will try to guess which CIFAR-10 class it belongs to.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess: Resize to 32x32 and normalize
    image = image.resize((32, 32))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 32, 32, 3)

    # Predict
    predictions = model.predict(img_array)[0]
    top_3_indices = predictions.argsort()[-3:][::-1]  # Top 3 in descending order

    st.subheader("Top 3 Predictions:")

    for i in top_3_indices:
        st.write(f"{class_names[i]}: {predictions[i] * 100:.2f}%")

    st.subheader("Prediction:")
    st.markdown(f"**{class_names[pred_index]}** with **{confidence:.2f}%** confidence")
