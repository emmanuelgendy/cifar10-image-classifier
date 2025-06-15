import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import matplotlib.pyplot as plt
import pickle


def plot_training_history():
    try:
        with open("training_history.pkl", "rb") as f:
            history = pickle.load(f)

        acc = history['accuracy']
        val_acc = history['val_accuracy']
        loss = history['loss']
        val_loss = history['val_loss']
        epochs = range(1, len(acc) + 1)

        st.subheader("📈 Training History")

        # Accuracy Plot
        fig1, ax1 = plt.subplots()
        ax1.plot(epochs, acc, label='Train Accuracy')
        ax1.plot(epochs, val_acc, label='Val Accuracy')
        ax1.set_title("Accuracy over Epochs")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.legend()
        st.pyplot(fig1)

        # Loss Plot
        fig2, ax2 = plt.subplots()
        ax2.plot(epochs, loss, label='Train Loss')
        ax2.plot(epochs, val_loss, label='Val Loss')
        ax2.set_title("Loss over Epochs")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()
        st.pyplot(fig2)

    except FileNotFoundError:
        st.warning("Training history file not found. Please train and save the model first.")

if st.checkbox("📊 Show Training History"):
    plot_training_history()


class CIFARCam(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_resized = cv2.resize(img, (32, 32))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb / 255.0
        img_input = np.expand_dims(img_normalized, axis=0)

        preds = model.predict(img_input)[0]
        top = preds.argsort()[-1]
        label = f"{class_names[top]}: {preds[top]*100:.2f}%"

        # Draw label on image
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        return img


# Class names for CIFAR-10
class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# Load the trained model
@st.cache_resource
def load_model():
    #model = tf.keras.models.load_model("cifar10_model.h5")
    model = tf.keras.models.load_model("cifar10_model_improved.keras")
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


st.subheader("📷 Or use your webcam:")
webrtc_streamer(
    key="cifar-webcam",
    video_transformer_factory=CIFARCam
)

