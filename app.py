import streamlit as st
import matplotlib.pyplot as plt
from keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
model = load_model('my_model.keras')

# Define class names
class_names = ['5_miles_speedlimit', 'No_entry', 'speedlimit_15']

# Streamlit UI
st.set_page_config(page_title="Traffic Sign Classification", layout="centered")
st.title("Traffic Sign Classification")
st.write("Upload an image to classify the traffic sign.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load and preprocess the image
    img = Image.open(uploaded_file)
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = img_array.reshape(1, 224, 224, 3)
    
    # Predict the label
    label = model.predict(img_array)
    predicted_class_index = np.argmax(label)
    predicted_class = class_names[predicted_class_index]
    
    # Display image and prediction
    st.image(img, caption=f"Predicted Class: {predicted_class}", use_column_width=True)
    st.success(f"Prediction: {predicted_class}")
