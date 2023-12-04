import os
import pandas as pd
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
        score_conf, cls, conf = predict(model, image.resize((224, 224)))
        st.image(image, use_column_width="always")
        st.success(f"Predicted image: {cls} with confidence score: {conf}")
        min_score = score_conf.min()
        max_score = score_conf.max()
        score_conf = ((score_conf - min_score) / (max_score - min_score)) * 100
        df_score = pd.DataFrame(score_conf, columns=["Bare", "Sedang", "Tinggi"])
        # df_score['Conf. Score'] = df_score['Conf. Score'].apply(lambda i: round(i, 2))
        st.bar_chart(df_score)
    else:
        st.warning("Please Upload Image...")

def about_me():
    st.markdown(
        """
        Hi Everyone, im ... !
        
        """
    )

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
        about_me()