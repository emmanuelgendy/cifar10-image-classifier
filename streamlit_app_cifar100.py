import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cifar100_model_improved.keras")


model = load_model()

class_names = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee',
    'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus',
    'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch',
    'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant',
    'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo',
    'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard',
    'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree',
    'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy',
    'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
    'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper',
    'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower',
    'sweet_pepper', 'table', 'tank', 'telephone', 'television',
    'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe',
    'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

st.title("CIFAR-100 Image Classifier")

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
