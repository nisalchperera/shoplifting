import os

import streamlit as st

from test import test


def app():
    st.header('Shoplifting Detection Web App')
    st.write('Welcome!')

    with st.form("my_form"):
        uploaded_file = st.file_uploader("Upload video", type=['mp4'])
        st.form_submit_button(label='Submit')
            
    if uploaded_file is not None: 
        input_path = os.path.join("data/validation/", uploaded_file.name)
        file_binary = uploaded_file.read()
        with open(input_path, "wb") as temp_file:
            temp_file.write(file_binary)
        
        with st.spinner('Processing video...'): 
            test([input_path])


if __name__ == "__main__":
    app()