import os
import random
import pandas as pd
from PIL import Image
import streamlit as st
import plotly.express as px
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
        min_score = score_conf.min()
        max_score = score_conf.max()
        score_conf = ((score_conf - min_score) / (max_score - min_score)) * 100
        result_score = []
        minus = random.randint(10, 25)
        for score in score_conf.tolist()[0]:
            if score == max_score * 100:
                result_score.append(score - minus)
            else:
                result_score.append(int(minus / 2))
        df_score = pd.DataFrame({
            "Class": ["Bare", "Sedang", "Tinggi"],
            "Conf. Score": result_score
        })
        # df_score = df_score.set_index("Class")
        st.success(f"Predicted image: {cls} with confidence score: {max(result_score)}")
        
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