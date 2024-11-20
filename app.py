import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Define affine transformation functions
def translate_image(image, tx, ty):
    rows, cols = image.shape[:2]
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    translated_image = cv2.warpAffine(image, translation_matrix, (cols, rows))
    return translated_image

def rotate_image(image, angle):
    rows, cols = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
    return rotated_image

def scale_image(image, scale_factor):
    rows, cols = image.shape[:2]
    scaled_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    return scaled_image

def shear_image(image, shear_factor):
    rows, cols = image.shape[:2]
    shear_matrix = np.float32([[1, shear_factor, 0], [0, 1, 0]])
    sheared_image = cv2.warpAffine(image, shear_matrix, (cols + int(shear_factor * rows), rows))
    return sheared_image

# Streamlit app
st.title("Affine Transformations with Streamlit")

# Sidebar menu for selecting transformation
st.sidebar.title("Affine Transformations Menu")
uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = np.array(Image.open(uploaded_file))
    st.image(image, caption="Original Image", use_column_width=True)

    st.sidebar.subheader("Select Transformation")
    transformation = st.sidebar.selectbox(
        "Choose an affine transformation",
        ["Translation", "Rotation", "Scaling", "Shearing"]
    )

    if transformation == "Translation":
        tx = st.sidebar.slider("Translate X", -100, 100, 0)
        ty = st.sidebar.slider("Translate Y", -100, 100, 0)
        result_image = translate_image(image, tx, ty)
    elif transformation == "Rotation":
        angle = st.sidebar.slider("Rotation Angle", -180, 180, 0)
        result_image = rotate_image(image, angle)
    elif transformation == "Scaling":
        scale_factor = st.sidebar.slider("Scale Factor", 0.1, 3.0, 1.0)
        result_image = scale_image(image, scale_factor)
    elif transformation == "Shearing":
        shear_factor = st.sidebar.slider("Shear Factor", -1.0, 1.0, 0.0)
        result_image = shear_image(image, shear_factor)

    if 'result_image' in locals():
        st.image(result_image, caption="Transformed Image", use_column_width=True)
else:
    st.sidebar.info("Please upload an image to get started.")
