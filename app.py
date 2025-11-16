import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
import io

# Load your saved model
MODEL_PATH = "alzheimer_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Define stage names
class_names = {
    0: "No Alzheimer",
    1: "Very Mild Dementia",
    2: "Mild Dementia",
    3: "Moderate Dementia"
}

# Define detailed stage messages and color codes
stage_details = {
    0: {
        "emoji": "🟢",
        "color": "success",
        "message": "🎉 **Great news!** There's no sign of Alzheimer's disease. Keep maintaining a healthy lifestyle with regular exercise, brain activities like puzzles, and a balanced diet."
    },
    1: {
        "emoji": "🟡",
        "color": "warning",
        "message": "🙂 **Very mild signs detected.** It’s manageable. Stay mentally active, follow a brain-healthy diet (like Mediterranean), sleep well, and consult your doctor for early advice."
    },
    2: {
        "emoji": "🟡",
        "color": "warning",
        "message": "😊 **Mild symptoms detected.** Routine support from family, regular memory activities, and prescribed medication can slow down the progress effectively."
    },
    3: {
        "emoji": "🔴",
        "color": "error",
        "message": "⚠️ **Moderate signs observed.** Immediate consultation with a neurologist is advised. Structured routines, support systems, and memory care plans will help manage it efficiently."
    }
}

def is_probable_mri(image_file):
    try:
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        img_color = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        image_file.seek(0)  # Reset file pointer

        if img_gray is None or img_gray.shape[0] < 64 or img_gray.shape[1] < 64:
            return False

        # 1. Aspect Ratio check
        h, w = img_gray.shape
        aspect_ratio = max(h, w) / min(h, w)
        if aspect_ratio > 1.5:
            return False

        # 2. Check if mostly grayscale (MRI should be grayscale)
        # Calculate color variance: if color channels differ a lot, it's likely not grayscale MRI
        b, g, r = cv2.split(img_color)
        mean_diff = (np.mean(np.abs(b - g)) + np.mean(np.abs(g - r)) + np.mean(np.abs(r - b))) / 3
        if mean_diff > 15:  # threshold for color variation, tweak if needed
            return False  # image is colored (likely photo or document)

        # 3. Brightness check (mean grayscale intensity)
        mean_intensity = np.mean(img_gray)
        if mean_intensity < 40 or mean_intensity > 220:
            return False

        # 4. Edge density check - MRIs have moderate edges
        edges = cv2.Canny(img_gray, 100, 200)
        edge_density = np.sum(edges > 0) / (h * w)
        if edge_density < 0.01 or edge_density > 0.3:
            return False

        # 5. Texture check (blurriness measure)
        laplacian_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()
        if laplacian_var < 50:
            return False

        return True
    except:
        return False

# Preprocess function
def preprocess_image(uploaded_image):
    image = Image.open(uploaded_image).convert('L')  # Convert to grayscale
    image = image.resize((128, 128))
    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=-1)  # Add channel dimension
    image_array = np.expand_dims(image_array, axis=0)   # Add batch dimension
    return image_array

# Page configuration
st.set_page_config(page_title="Alzheimer MRI Classifier", layout="centered")

# Title and Description
st.title("🧠 Alzheimer MRI Classifier")
st.markdown("""
Alzheimer’s disease is a progressive neurological disorder that primarily affects memory, thinking, and behavior.
Early detection plays a crucial role in managing and slowing the progression of the disease.

This tool uses a deep learning model to classify the stage of Alzheimer’s based on MRI images.
Please upload a brain MRI image to receive a prediction and guidance.
""")

# File uploader
uploaded_file = st.file_uploader("📤 Upload an MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="🖼️ Uploaded Image", use_container_width=True)
    st.markdown("---")

    with st.spinner("🔎 Verifying the image..."):
        # Copy image bytes for processing (Streamlit resets after read)
        file_bytes = uploaded_file.read()
        uploaded_file.seek(0)

        if not is_probable_mri(io.BytesIO(file_bytes)):
            st.error("⚠️ The uploaded image does not appear to be a brain MRI ")
        else:
            with st.spinner("🧪 Analyzing the MRI..."):
                preprocessed_image = preprocess_image(io.BytesIO(file_bytes))
                prediction = model.predict(preprocessed_image)
                predicted_class = int(np.argmax(prediction))
                stage = class_names[predicted_class]
                stage_info = stage_details[predicted_class]

                st.markdown(f"### {stage_info['emoji']} **Prediction: {stage}**")
                getattr(st, stage_info["color"])(stage_info["message"])

    st.markdown("---")
    st.markdown("🔄 **You can upload another MRI for a different prediction.**")
