import streamlit as st
from utils import preprocess_image, CLASS_NAMES
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Brain Tumor Detection", layout="centered")
st.title("🧠 Brain Tumor Classification (Multi-Class)")

uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI", use_column_width=True)
    st.write("Classifying...")

    model = tf.keras.models.load_model("model.h5")
    img_array = preprocess_image(image)
    preds = model.predict(img_array)
    label = CLASS_NAMES[np.argmax(preds)]
    confidence = np.max(preds)

    st.success(f"Prediction: {label}")
    st.info(f"Confidence: {confidence * 100:.2f}%")
