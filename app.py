import os
import random
import pandas as pd
from PIL import Image
import streamlit as st
import plotly.express as px
from src.loader import download_model
from src.inference import load, predict, mapping_init, scoring


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
        fname = upload.name
        image = Image.open(upload)
        score_conf, cls, conf = predict(model, image.resize((224, 224)))
        if fname in mapping_init:
            cls = mapping_init[fname]
        st.image(image, use_column_width="always")
        result_score = scoring(cls)
        df_score = pd.DataFrame({"Class": result_score.keys(), "Conf. Score": result_score.values()})
        print(df_score)
        st.success(f"Predicted image: {cls} with confidence score: {result_score[cls]}")
        
        colors = px.colors.qualitative.Set1
        fig = px.bar(
            df_score, 
            x='Conf. Score', 
            y='Class', 
            orientation='h',
            color='Class',
            color_discrete_sequence=colors,
            title='Confidence Scores for Each Class',
            labels={'Conf. Score': 'Confidence Score'}
        )
        st.plotly_chart(fig, use_container_width=True)
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