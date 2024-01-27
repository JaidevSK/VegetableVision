# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st

# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="VegetableVision",
    page_icon=":tomato:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Vegetable Detection using YOLOv8")

# Sidebar
st.sidebar.header("Select the Configuration")

# # Model Options
# model_type = st.sidebar.radio(
#     "Select Task", ['Detection', 'Segmentation'])

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100

# # Selecting Detection Or Segmentation
# if model_type == 'Detection':
#     model_path = Path(settings.DETECTION_MODEL)
# elif model_type == 'Segmentation':
#     model_path = Path(settings.SEGMENTATION_MODEL)
model_path = Path(settings.DETECTION_MODEL)


# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)


source_img = None
# If image is selected
source_img = st.sidebar.file_uploader(
    "Choose an image...", type=("jpg", "jpeg", "png"))

col1, col2 = st.columns(2)

with col1:
    try:
        if source_img is None:
            st.write("No image is uploaded yet!")
        else:
            uploaded_image = PIL.Image.open(source_img)
            st.image(source_img, caption="Uploaded Image",
                        use_column_width=True)
    except Exception as ex:
        st.error("Error occurred while opening the image.")
        st.error(ex)

with col2:
    if source_img is None:
        st.write("Upload an image and click Detect!")
    else:
        if st.sidebar.button('Detect Objects'):
            res = model.predict(uploaded_image,
                                conf=confidence
                                )
            boxes = res[0].boxes
            res_plotted = res[0].plot()[:, :, ::-1]
            st.image(res_plotted, caption='Detected Image',
                        use_column_width=True)
            try:
                with st.expander("Detection Results"):
                    for box in boxes:
                        st.write(box.data)
            except Exception as ex:
                # st.write(ex)
                st.write("No image is uploaded yet!")
