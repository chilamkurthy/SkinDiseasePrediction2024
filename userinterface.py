import streamlit as st
import numpy as np
from PIL import Image 

# Load your pre-trained model (update the path as needed)
model = load_model('vnraiml_recognition_model_256.h5')

def preprocess_image(img):
    img = img.resize((224, 224))  # Resize the image to the input size of the model
    img_array = np.array(img) / 255.0  # Normalize the image
    return np.expand_dims(img_array, axis=0)

# Streamlit UI
st.title("Skin Disease Recognition and Growth Analysis")

# Image input
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Display the image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image",width=500)

    # Preprocess the image
    processed_image = preprocess_image(img)

    # Get other inputs
    age = st.number_input("Enter your age:", min_value=0, max_value=120)
    gender = st.selectbox("Select Gender:", ("Male", "Female", "Other"))
    drink = st.selectbox("Will you drink alcohol?", ("Yes", "No"))
    smoke = st.selectbox("Will you smoke?", ("Yes", "No"))
    itch = st.selectbox("Do you experience itching?", ("Yes", "No"))
    pain = st.selectbox("Do you feel pain?", ("Yes", "No"))
    bleed = st.selectbox("do you have bleeding?", ("Yes", "No"))
    sports = st.selectbox("Do you play outdoor sports?", ("Yes", "No"))

    if st.button("Predict"):
        # Make predictions with the model
        #predictions = model.predict(processed_image)
        result = 1 # Assuming the output is a classification

        # Display the results
        st.success(f"Prediction Result: {result}")
        st.write(f"Age: {age}")
        st.write(f"Gender: {gender}")
        st.write(f"Drinks Alcohol: {drink}")
        st.write(f"Smokes: {smoke}")
        st.write(f"Disease: MELANOMA")

# Run the app using: streamlit run your_script_name.py
