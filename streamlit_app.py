import sys
sys.path.append ("/mount/src/shoplifting/")  

import os
import subprocess

import streamlit as st

from urllib.request import urlretrieve


def download_progress_hook(count, blocksize, totalsize):
    
    bytes_map = {
        0: "Kb",
        1: "Mb",
        2: "Gb",
        3: "Tb"
    }
    
    downloaded = count * blocksize
    if totalsize > 0:
        percentage = downloaded / totalsize * 100
        bar_length = 40
        block = int(bar_length * downloaded / totalsize)
        progress = '=' * block + '-' * (bar_length - block)

        down_unit = "bytes"
        total_unit = "bytes"
        
        print(f'\r|{progress}| {percentage:.2f}% ({downloaded} {down_unit}/{totalsize} {total_unit})', end='')
    if downloaded >= totalsize:
        print("Download successfull")

if not os.path.exists("models/shoplifting_detector_3d_cnn.pth"):
    link = 'https://shoplifting-analysis.s3.us-east-1.amazonaws.com/models/shoplifting_detector_3d_cnn.pth?response-content-disposition=inline&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEJT%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIQCtDUZsEmsq6165A2tM7U00TWbJD1sJbugwZXd3uNO1igIgeSYv4dgYCruSQ4EVxI1Fug%2Bx4yE10QnZpCwWa%2FC7NAAq8QIIzf%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARADGgw0NTQ4NDI1NjQzNzAiDDsqWiNa39o%2FKr4khyrFAjKD56lU28Q3lsfotxPXA3JAMu0Tc1mAlVY4DAgq68HHhdLcPMhY6vwn0bEikUQZzR0UAFMePm%2FULimDkLVivkq0l7KAAs4CY71f0Xu1Oxrivw7ICdjIp0T5YaZLGV6n2qEfpuJ91SiI6Cdgg7WsCi%2BpgFwOiQnVfH%2BktEKj%2Bxs2%2BDuycUPIy7niBK133fMJQ0qI7OaQo4ByjrOYgIiZUpMR%2BoFGir9S0rD4pZE013Zfk2dkidzOb%2BYvEgxv5%2FYGDxAdcKpauC6ku%2BJME9IqqQYUtfJA2x6c5KzTbvjoMeHDqNGk4xyeB664qcuDd%2F9D8LsKsdBXX09GskWchQ4vWByebXmMENSpstHf7Y9iu2MmdhiZi%2Bg%2BnBTdggJY2kUQBBemZ%2FmEqbQPpNFe2k30KgU7joYUoVQ5rQiMZWXfw9g6Z%2BGtvvYwkfDItwY6swLdTewjTNX91VlQLpq%2Fj8Kf%2Ft4uXRhm8sf%2BJG9vS9YpFmqJhPybIr5B9VX%2B7BkOgAcFSwZP7uMJlscyGAniqcdi2mBoJSaYIq3st4CDTNBjXn10TAPKotAxEjPGket%2FGJlfKd6dhd%2B3QlhDGmIw2z%2FFGqsDe7Od8JI9ONbnURJIrs1NdwqAbPAXhULu99YdVQPduR3RH7J3%2BtDZAMBKSrpLhUqKxfGZtlfZJISXwu1g%2Fqlp9MoseQRztS%2B6%2FD7pAW0c0R5HTa%2FaTCshu0XvW4nWNHiJdE4JOvzEViOzfnZOO8jZjcl%2BhyfEwBH%2BcvsVgExsS90Z5wozMB0sTpcz0Z%2FB66RbG1bn4ugs7s68g9j6WCEePBd4tAkjsH%2BA9sKXNWBsQKRC06TT0xmO5Qp%2FxW6IH6Ea&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240924T045552Z&X-Amz-SignedHeaders=host&X-Amz-Expires=43200&X-Amz-Credential=ASIAWTZV2X4JJZC65TYN%2F20240924%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=bc0dd35da8f5e4edb2084891c0ac6bdb1567a85c9b2dd2c022a82a0c9ebaccff'
    
    # command = ["wget", f"'{link}'", "-O", "./models/shoplifting_detector_3d_cnn.pth"]
    # print(f"Running Command: {' '.join(command)}")
    urlretrieve(link, "./models/shoplifting_detector_3d_cnn.pth", reporthook=download_progress_hook)

from test import test


def app():
    st.header('Shoplifting Detection Web App')
    st.write('Welcome!')

    with st.form("my_form"):
        uploaded_file = st.file_uploader("Upload video", type=['mp4'])
        st.form_submit_button(label='Submit')
            
    if uploaded_file is not None: 
        os.makedirs("data/validation/originals", exist_ok=True)
        os.makedirs("data/validation/predicted", exist_ok=True)
        
        input_path = os.path.join("data/validation/originals", uploaded_file.name)
        if not os.path.exists(os.path.join("data/validation/predicted", uploaded_file.name)):
            file_binary = uploaded_file.read()
            with open(input_path, "wb+") as temp_file:
                temp_file.write(file_binary)
            
            with st.spinner('Processing video...'): 
                videos = test([input_path])
            
            for output_path in videos:
                st.text(f"Visualizing {output_path}")
                with open(output_path, 'rb') as f:
                    st.download_button('Download Video', f, file_name=output_path.split("/")[-1])
        
        else:
            output_path = input_path.replace("originals", "predicted")
            with open(output_path, 'rb') as f:
                st.download_button('Download Video', f, file_name=output_path.split("/")[-1])


if __name__ == "__main__":
    app()