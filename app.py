import streamlit as st
#import joblib
from PIL import Image
import tempfile
import cv2
import os
# Load model
#model = torch.load("model.pth")

#import logo
# Path to logo in the same folder as app.py
logo_path = "logo/when_ai_sees_litter_logo.png"
import base64

# Read and convert to base64
with open(logo_path, "rb") as f:
    logo_bytes = f.read()
logo_b64 = base64.b64encode(logo_bytes).decode("utf-8")

# Inject HTML + CSS to place it top-right
st.markdown(
    f"""
    <style>
    .top-right {{
        position: absolute;
        top: 10px;
        right: 10px;
    }}
    </style>
    <div class="top-right">
        <img src="data:image/png;base64,{logo_b64}" width="120">
    </div>
    """,
    unsafe_allow_html=True
)

# Main content
st.title("When AI Sees Litter")
st.write("An app to help beginners identify and handle trash responsibly 🌱")
st.subheader("📸 Identify Your Litter Now!")

# Dropdown selection
option = st.selectbox(
    "Choose input type:",
    ("Capture Photo (Webcam)", "Upload Image", "Upload Video")
)

# --- Option 1: Webcam Capture ---
if option == "Capture Photo (Webcam)":
    camera_photo = st.camera_input("Take a picture")
    if camera_photo is not None:
        img = Image.open(camera_photo)
        st.image(img, caption="Captured Photo", use_container_width=True)
        st.success("✅ Photo captured successfully!")

# --- Option 2: Upload Image ---
elif option == "Upload Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        img = Image.open(uploaded_image)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        st.success("✅ Image uploaded successfully!")

# --- Option 3: Upload Video ---
elif option == "Upload Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
    if uploaded_video is not None:
        # Display video player
        st.video(uploaded_video)

        # Save video to a temp file so OpenCV can read it
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        # Extract and show the first frame
        cap = cv2.VideoCapture(tfile.name)
        ret, frame = cap.read()
        if ret:
            st.image(frame, caption="First Frame from Video", channels="BGR")
        cap.release()

#if st.button("Predict"):
#prediction = model.predict([[float(user_input)]])
#st.write(f"Prediction: {prediction}")
