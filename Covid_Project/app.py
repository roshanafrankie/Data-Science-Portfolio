import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Set page configuration
st.set_page_config(page_title="COVID-19 X-Ray Detector", page_icon="🛡️")

# 1. Load the model you saved from your notebook
@st.cache_resource
def load_my_model():
    # This loads the .keras file created by cnn.save()
    return tf.keras.models.load_model('covid_model.keras')

try:
    model = load_my_model()
except Exception as e:
    st.error(f"Error loading model: {e}. Make sure 'covid_model.keras' is in the same folder.")

# 2. Web Page User Interface
st.title("🛡️ COVID-19 Chest X-Ray Classifier")
st.write("This tool uses a Convolutional Neural Network (CNN) to analyze chest X-rays for signs of COVID-19.")
st.markdown("---")

# File uploader widget
uploaded_file = st.file_uploader("Upload a Chest X-ray (JPG, JPEG, or PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 3. Image Display & Processing
    # Convert to RGB to ensure 3 channels, matching model input_shape [128, 128, 3]
    img = Image.open(uploaded_file).convert('RGB')
    
    # Show the image on the web page
    st.image(img, caption='Uploaded X-ray', use_container_width=True)
    
    with st.spinner('AI is analyzing the image...'):
        # 4. Preprocessing (Must match your Training logic)
        # Resize to 128x128 as defined in your target_size
        img_resized = img.resize((128, 128))
        img_array = image.img_to_array(img_resized)
        
        # Add batch dimension: (1, 128, 128, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Rescale pixels to [0, 1] matching your ImageDataGenerator(rescale=1./255)
        img_array /= 255.0 

        # 5. Prediction
        prediction = model.predict(img_array)
        
        # 6. Output Logic
        # Based on your training_set.class_indices: {'Covid': 0, 'Normal': 1}
        st.markdown("### Analysis Result:")
        
        # Using 0.5 threshold for the Sigmoid output
        if prediction[0][0] > 0.5:
            confidence = prediction[0][0] * 100
            st.success(f"**PREDICTION: NORMAL**")
            st.info(f"Confidence: {confidence:.2f}%")
        else:
            confidence = (1 - prediction[0][0]) * 100
            st.error(f"**PREDICTION: COVID-19 DETECTED**")
            st.info(f"Confidence: {confidence:.2f}%")

st.markdown("---")
st.caption("Note: This is a demonstration tool and should not be used for actual medical diagnosis.")