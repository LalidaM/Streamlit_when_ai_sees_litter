import streamlit as st
#import joblib


# Load model
#model = torch.load("model.pth")

st.title("When AI Sees Litter App")

# Example: simple input
user_input = st.text_input("Enter some text or number:")

from PIL import Image

st.title("📸 Upload a Photo")

# Upload photo (accept only images)
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image with PIL
    image = Image.open(uploaded_file)

    # Show the image in app
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.success("✅ Image uploaded successfully!")

#if st.button("Predict"):
#prediction = model.predict([[float(user_input)]])
#st.write(f"Prediction: {prediction}")
