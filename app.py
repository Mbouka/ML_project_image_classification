from email import base64mime
from keras.models  import load_model  
import streamlit as st
from PIL import Image, ImageOps
import numpy as np


page_bg_img = """
<style>
[data-testid="stAppViewContainer"]{
background-image: url("https://images.unsplash.com/photo-1501426026826-31c667bdf23d");
background-size: 180%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}
[data-testid="stHeader"]{
background-color: rgba(0,0,0,0);
}
</style>"""
st.markdown(page_bg_img, unsafe_allow_html=True)
st.title('Classement des déchets')

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model

model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

uploaded_file = st.file_uploader("Choisissez une image", type=['jpeg','jpg','png'])

if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        with col1:
    
            st.info("Votre image téléchargée")

            st.image(image,caption='uploaded image',use_column_width=True)

            size = (224, 224)
            image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

            image_array = np.asarray(image)

            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            data[0] = normalized_image_array

            prediction = model.predict(data)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = prediction[0][index]

        #display results in column 2
        with col2:
            st.info("Resultat")
            st.write("Ceci est un:", class_name[2:], end="")
            st.write("Seuil de confiance:", confidence_score)

else:
       # with col1:
            st.write("Veuillez télécharger un fichier image pour obtenir une prédiction.")
    

# Link for deployed model :https://teachablemachine.withgoogle.com/models/KJLVpl6aK/