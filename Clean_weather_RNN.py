import io
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import streamlit as st
from PIL import Image
from keras.applications.vgg16 import VGG16
# Load model and define class labels
model = load_model("my_model.keras",compile=False)
class_names = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']

st.title("🧠 Brain Tumor Detection")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if not uploaded_file:
    st.stop()

# Read and preprocess image
image_bytes = uploaded_file.getvalue()
img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((150, 150))
st.image(img, caption="Uploaded Image", use_column_width=True)

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = x / 255.0

# Predict
final_prediction = model.predict(x)[0]
predicted_class_index = np.argmax(final_prediction)
confidence = final_prediction[predicted_class_index]
predicted_label = class_names[predicted_class_index]

# Display result
st.subheader(f"🩺 Prediction: **{predicted_label}**")
st.write(f"Confidence: `{confidence:.2f}`")
st.write("Raw output (softmax):", final_prediction)
