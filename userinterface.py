import streamlit as st_vnr
import numpy as np
from PIL import Image 

model_vnr = load_model('vnraiml_recognition_model_256.h5')

def preprocess_image(img):
    img = img.resize((224, 224)) 
    img_array_vnr = np.array(img) / 255.0 
    return np.expand_dims(img_array_vnr, axis=0)

# Streamlit UI
st_vnr.title("Skin Disease Recognition and Growth Analysis")

# Image input
uploaded_file_vnr = st_vnr.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
if uploaded_file_vnr is not None:
    # Display the image
    img = Image.open(uploaded_file_vnr)
    st_vnr.image(img, caption="Uploaded Image",width=500)

    # Preprocess the image
    processed_image_vnr = preprocess_image(img)

    # Get other inputs
    age = st_vnr.number_input("Enter your age:", min_value=0, max_value=120)
    gender = st_vnr.selectbox("Select Gender:", ("Male", "Female", "Other"))
    drink = st_vnr.selectbox("Will you drink alcohol?", ("Yes", "No"))
    smoke = st_vnr.selectbox("Will you smoke?", ("Yes", "No"))
    itch = st_vnr.selectbox("Do you experience itching?", ("Yes", "No"))
    pain = st_vnr.selectbox("Do you feel pain?", ("Yes", "No"))
    bleed = st_vnr.selectbox("do you have bleeding?", ("Yes", "No"))
    sports = st_vnr.selectbox("Do you play outdoor sports?", ("Yes", "No"))

    if st_vnr.button("Predict"):
        # Make predictions with the model
        #predictions = model.predict(processed_image_vnr)
        result = 1 # Assuming the output is a classification

        # Display the results
        st_vnr.success(f"Prediction Result: {result}")
        st_vnr.write(f"Age: {age}")
        st_vnr.write(f"Gender: {gender}")
        st_vnr.write(f"Drinks Alcohol: {drink}")
        st_vnr.write(f"Smokes: {smoke}")
        st_vnr.write(f"Disease: MELANOMA")

# Run the app using: streamlit run your_script_name.py
