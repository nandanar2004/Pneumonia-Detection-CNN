import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🫁",
    layout="centered"
)

# Project title
st.title("🫁 Pneumonia Detection System")

st.write(
    "A CNN-based system for classifying chest X-ray images "
    "as NORMAL or PNEUMONIA."
)

st.info(
    "Upload a chest X-ray image to get a model prediction."
)

st.warning(
    "Educational project only. This application is not a "
    "medical diagnostic tool."
)

# Load the trained CNN model
@st.cache_resource
def load_cnn_model():
    model_path = Path(__file__).parent / "models" / "model2_final.keras"
    model = tf.keras.models.load_model(model_path)
    return model

model = load_cnn_model()

st.success("CNN model loaded successfully!")
# Image upload
st.subheader("Upload Chest X-ray Image")

uploaded_file = st.file_uploader(
    "Choose a chest X-ray image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Chest X-ray",
        use_container_width=True
    )

    # Predict button
    if st.button("Predict Pneumonia", type="primary"):

        # Preprocess image
        resized_image = image.resize((224, 224))

        img_array = np.array(resized_image)
        img_array = np.expand_dims(img_array, axis=0)

        # Model prediction
        probability = model.predict(
            img_array,
            verbose=0
        )[0][0]

        # Classification threshold
        threshold = 0.3

        if probability >= threshold:
            prediction = "PNEUMONIA"
            st.error("Prediction: PNEUMONIA")
        else:
            prediction = "NORMAL"
            st.success("Prediction: NORMAL")

        # Display probability
        st.write(
            f"Pneumonia Probability: {probability * 100:.2f}%"
        )