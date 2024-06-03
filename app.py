import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

# Load the pre-trained VGG16 model
model = load_model('model_vgg16.h5')

st.markdown(
    """
    <style>
    .centered {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title of the web app
st.markdown('<h1 class="centered">Cat Vs Dog Classifier</h1>', unsafe_allow_html=True)

# Add some blank lines for spacing
st.text("\n\n")

# Image uploader widget
uploaded_file = st.sidebar.file_uploader("Choose an image (cat/dog)", type=["jpg", "jpeg", "png"])

# Check if an image has been uploaded
if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image for prediction
    img = image.load_img(uploaded_file, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    img_data = preprocess_input(x)

    # Prediction button
    if st.button("Predict"):
        # Predict the class of the uploaded image
        result = np.argmax(model.predict(img_data), axis=1)
        prediction = 'Dog' if result[0] == 1 else 'Cat'
        st.title(f"Prediction: {prediction}")
