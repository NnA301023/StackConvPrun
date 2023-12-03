import os
from PIL import Image
import streamlit as st
from src.loader import download_model
from src.inference import load, predict


@st.cache(allow_output_mutation=True)
def load_model():
    model_path = "./pruned_models/model_concatenation_large_params_pruned.h5"
    model = load(model_path=model_path)
    return model


def interface_prediction():
    upload = st.file_uploader(
        "Insert Image", 
        type=["png", "jpg", "jpeg", "PNG", "JPG", "JPEG"]
    )
    if upload is not None:
        image = Image.open(upload)
        cls, conf = predict(model, image.resize((224, 224)))
        st.image(image, use_column_width="always")
        st.success(f"Predicted image: {cls} with confidence score: {conf}")
    else:
        st.warning("Please Upload Image...")

def about_me():
    pass

if __name__ == "__main__":
    st.title("Peat Land Image Classification from UAV Images")
    tab_predict, tab_about = st.tabs(['Inference', 'About Me'])
    
    model_path = "./pruned_models/model_concatenation_large_params_pruned.h5"
    if not os.path.exists(model_path):
        with st.spinner(text="Download Model..."):
            download_model(model_path)

    model = load_model()
    with tab_predict:
        interface_prediction()
    with tab_about:
        st.markdown(
            """
            Hi Everyone, im ... !
            
            """
        )