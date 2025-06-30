import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Page setup
st.set_page_config(page_title="Cat vs Dog AI Classifier", page_icon="🐾", layout="centered")

# Hide Streamlit footer and menu
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cats_vs_dogs_model.h5")

model = load_model()

# UI
st.markdown("<h1 style='text-align: center;'>🐱🐶 Cat vs Dog Classifier</h1>", unsafe_allow_html=True)
st.write("Upload a pet image and let the AI guess whether it's a cat or a dog!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    image = image.resize((128, 128))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Predict with spinner
    with st.spinner("Analyzing image... 🧠"):
        prediction = model.predict(image_array)[0][0]
        label = "🐶 Dog" if prediction > 0.5 else "🐱 Cat"
        confidence = prediction if prediction > 0.5 else 1 - prediction

    st.markdown(f"### Prediction: {label}")
    st.progress(float(confidence))
    st.markdown(f"**Confidence:** {confidence:.2%}")
