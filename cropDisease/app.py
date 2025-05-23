import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from advices import advices

#trained model
model = tf.keras.models.load_model("cropDisease/model.h5")

#class names
class_names = [
    'Rice_Bacterial Blight',
    'Rice_Blast',
    'Rice_Brown Spot',
    'Rice_Tungro',
    'Wheat_Brown Rust',
    'Wheat_Loose Smut',
    'Wheat_Septoria',
    'Wheat_Healthy',
    'Wheat_Yellow Rust'
]

# Function to get local advice
def get_disease_measures(disease_name):
    return advices.get(disease_name.lower(), "No advice available for this disease.")

# Streamlit UI
st.set_page_config(page_title="Crop Disease Detector", page_icon="🌿")
st.title("🌾 Crop Disease Detection and Farmer Guidance")

# Crop type input
crop_type = st.selectbox("🌱 Select the crop type", options=["Rice", "Wheat"])

uploaded_file = st.file_uploader("📷 Upload a crop leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_names[predicted_class_index]
    confidence = prediction[0][predicted_class_index]

    # Check if prediction matches selected crop
    if predicted_class.lower().startswith(crop_type.lower()):
        st.success(f"🧪 Predicted Disease: **{predicted_class}** (Confidence: {confidence:.2f})")

        with st.spinner("🌱 Retrieving expert advice..."):
            advice = get_disease_measures(predicted_class)
            st.subheader("✅ Recommended Management Practices:")
            st.write(advice)
    else:
        st.warning(f"The predicted disease (**{predicted_class}**) does not match the selected crop (**{crop_type}**). Please check the input.")

    st.markdown("---")
