import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="CIFAR-100 Classifier", layout="wide")

# CIFAR-100 labels
labels = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle',
    'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon',
    'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail',
    'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone',
    'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree',
    'wolf', 'woman', 'worm'
]

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('cifar100_model_improved.keras')
    return model

model = load_model()

st.title("CIFAR-100 Image Classifier")

option = st.radio("Choose input method", ['Upload Image', 'Use Webcam'])

def predict_image(image):
    image = image.resize((32, 32))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    predictions = model.predict(image)
    top_indices = predictions[0].argsort()[-5:][::-1]
    top_labels = [(labels[i], predictions[0][i]*100) for i in top_indices]
    return top_labels

if option == 'Upload Image':
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        top_predictions = predict_image(image)
        st.subheader("Top Predictions:")
        for label, confidence in top_predictions:
            st.write(f"{label}: {confidence:.2f}%")

elif option == 'Use Webcam':
    camera = st.camera_input("Take a picture")
    if camera is not None:
        image = Image.open(camera)
        st.image(image, caption='Captured Image', use_column_width=True)
        top_predictions = predict_image(image)
        st.subheader("Top Predictions:")
        for label, confidence in top_predictions:
            st.write(f"{label}: {confidence:.2f}%")
