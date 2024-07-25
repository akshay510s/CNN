import streamlit as st
import requests
from PIL import Image
import io

st.title("Face Expression Recognition")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    if st.button('Predict Expression'):
        buf = io.BytesIO()
        image.save(buf, format='JPEG')
        byte_im = buf.getvalue()
        response = requests.post('http://localhost:5000/predict', files={'image': byte_im})
        st.write('Predicted Expression:', response.json()['expression'])
