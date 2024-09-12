import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load the MobileNetV2 model
model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Create a Streamlit app
st.title("Image Classification App")
st.write("Upload an image to classify:")

# Create a file uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

# Load and preprocess the uploaded image
if uploaded_file is not None:
    img = load_img(uploaded_file, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = img_array.reshape((1, 224, 224, 3))

    # Make predictions using the MobileNetV2 model
    predictions = model.predict(img_array)

    # Get the top 5 predicted classes
    top_5_pred = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=5)

    # Display the predicted classes
    st.write("Predicted classes:")
    for pred in top_5_pred[0]:
        st.write(f"{pred[1]}: {pred[2]:.2f}%")