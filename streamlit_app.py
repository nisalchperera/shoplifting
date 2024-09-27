import sys
import shutil
import traceback
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
    link = 'https://shoplifting-analysis.s3.us-east-1.amazonaws.com/models/shoplifting_detector_3d_cnn.pth?response-content-disposition=inline&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEOD%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJIMEYCIQDUTzJI4jH2%2F4n2SmrrXaK8laxuHrv7oFZGvFNrg0N8xAIhAP0sr1l71JaSUeZgrulrfTQMCec1W1OKp9z1kkz1haofKugCCCkQAxoMNDU0ODQyNTY0MzcwIgyt8vdIgVaspkCmu3QqxQLBnEmMJiucL4ekXsqP52GNYvlLhYXcNuTOe6l6FY69J5w%2BegrXYwofhI9ZQamq3Qk88Qc2TEs3RonmKFR%2BNa79RTMCujGNwtf6Ke8KMhLuZt0ZnCntKUnMP2yxKLJdRIGEcZthWOrRLfsEh6PT%2FHiY%2BwmSwsH%2FBckk6aaIsdKD2quKYnTy0VX2fdTvL9INlzOJ2Zg5MhRCNmKSArUWBtKfujkLxxr4Ur6vQhBQSUSt1FtytB1R%2BJPs%2FjnueyVocNFb%2BHAW7%2BxpdnF4X4A4xbYAWsR7rWHbNngLOZ%2FMy2yMCpUVvxaw2bL3zkRTKxjyILx3f1j3y9H8aYJLLal9R2DEsFlphLOnnGbpwHjeG1e1MVlIWu9XR7zoChqnPs43j3m3pSWrR%2FE4yLtkwa%2B7zkGePoKF08wcyOzo0visgMPHq2YwvvP%2FMMa%2F2bcGOrICoLhu5zfzKVDr5fxP2T84YvOXwNFtAH54Ugou7U5k6iSi6xyppfYl6zknoDYqV1ipLdxLbNOaN7hBkb8Rfs8ZxkPI5LPJ1LU4T3WOGw5%2B9B29QBwjz%2Fbpfl9vtSwl0oeRs4G71pN6xjzaAQz9OgjCndKndutuQBWXbzI7emAQn4kQih%2Fzt0yi8Ue%2Fb3lpr%2FZznnsTcn31vkzE5EywghA5k0xrwV7mOnhRiJzEymvSlNyNpWW5DY%2FIQq9m%2BMZy9YlzsYFtAxyHwW4aJo2oEIDMHhbZ46XQatXOlso3y5%2FPWJQpCApPGqrfMmZRjkxV3dkKxiRJUPxu9Z4OGTAUY9NF1ZrwIyybBPzAaI61PxFFxeWJrQIYQ1xn%2FnEWtCGLcZ5r%2BB3acbD%2BeHXJVshrgcJ97FFo&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240927T073457Z&X-Amz-SignedHeaders=host&X-Amz-Expires=43200&X-Amz-Credential=ASIAWTZV2X4JEWNYZ4QV%2F20240927%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=31f552043e9c0a1252ef562f88daa2950bc1e0d9380c2b16ca7aeb2a3b5ea100'
    
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
    
    try:        
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
                    if os.path.exists(f"{output_path}_processed.mp4"):
                        st.video(f"{output_path}_processed.mp4")
                    else:
                        with st.spinner('Converting video...'):
                            subprocess.run(['ffmpeg', '-i', output_path, '-vcodec', 'h264', f"{output_path}_processed.mp4"])
                        
                        st.video(f"{output_path}_processed.mp4")
                    # with open(output_path, 'rb') as f:
                    #     st.download_button('Download Video', f, file_name=output_path.split("/")[-1])
            
            else:
                output_path = input_path.replace("originals", "predicted")
                if os.path.exists(f"{output_path}_processed.mp4"):
                    st.video(f"{output_path}_processed.mp4")
                else:
                    with st.spinner('Converting video...'):
                        subprocess.run(['ffmpeg', '-i', output_path, '-vcodec', 'h264', f"{output_path}_processed.mp4"])
                        
                    st.video(f"{output_path}_processed.mp4")
                # with open(output_path, 'rb') as f:
                #     st.download_button('Download Video', f, file_name=output_path.split("/")[-1])
                
            os.remove(output_path)
    except:
        st.error("An error occurred. Here's the full traceback:")
        st.code(traceback.format_exc())


if __name__ == "__main__":
    app()
