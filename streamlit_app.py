import sys
import shutil
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
    link = 'https://shoplifting-analysis.s3.us-east-1.amazonaws.com/models/shoplifting_detector_3d_cnn.pth?response-content-disposition=inline&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEMn%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIQD0TNOnr5%2BcB%2BM8jcE6L5FTACSI2Ai1662OFZEph0rPzwIgTjyFXCkJolgMU%2BK8tL9aJ4OlKDnIWlfm0YYeUd50ExIq6AIIEhADGgw0NTQ4NDI1NjQzNzAiDOzoLpj827dBBhO%2FEyrFAnPO%2FA8torvKqWeducD%2BfI8YOPyl4AnC1Towc9HeWqHCYvqUrBPbjYv%2FeSqKOWuT1oNpMF3Uo5xMFj7WeqEQDlp2u5ixSMlYBjLcD5mcbSgTzaRe7bb%2BmG8flm9aIrefUDPIx0nobuDHiIfFN4%2FesSVbUSdupbo%2F4TsJsLyIDVv4VMBkX1CjDblV7eVTENDiyrYIBT8AOJciymAKm9wtfn7ph6cu4pwjDfjT9eZWhrvToYIF3iow8a8wtqU5GUFbUynJkv2Tl8OnWF9LvCTkM9%2BRHN%2FhVTyRMiWut6Ao30D%2B01uhJsiZjrBxgNUJshVYrWmbmg1Ik1nyEExGyC3JcbixSsFIZiaeTGeZztXP%2F66B9iMSubL0FHSLb6bP66WO091PiDxphSV%2FPHspQsXvJGnVzfC47IKVkXAW91ZRy7%2BwOwqcdtkwru%2FTtwY6swL6jNFcE1x6tkZfWaBkpERfLVFx8V7wIx9Y2Pkejb4e3ItzeYb%2F7BnB0XTXgZkovmUdrqLX39YZCmRe6t2egj0h98ck8HjrokmMJKzJVsAyk039Xqj3deFPBA8PQAVV2%2BGJ00RZk0p4hKdMQu5Ir%2Fq5z%2BejtUPw0QFUMyS%2FbAe3lLR3YsuuBqSrm5Ioy6ZQ3eYJSE2CxC0YfYSXDSDVsqSrmtyrhEsxKYAOJdI5zlKSPleo66fP0gnHQ88rLShQXqIKmXcWfjOmHL2bZ903knUNt1WrTbjQFb93kaJYg3hg0B%2FHvSgnS5EiQMsZDyUD4ojQsKKJ4S5m%2FFnKlaswI2%2Fj53qS4WWyckcococJ%2Bvk0WWWv4UWF2V4ZgYSAuaLbQzBRvvaqtPhqY32xLxf5zGzo0rmR&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240926T090030Z&X-Amz-SignedHeaders=host&X-Amz-Expires=43200&X-Amz-Credential=ASIAWTZV2X4JKJQET5DC%2F20240926%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=13cd462654f6a83ff9d8f9977aafb66ccefc38ea583e4825af51d0cb3ced2a5c'
    
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
                if os.path.exists(f"{output_path}_processed.mp4"):
                    st.video(f"{output_path}_processed.mp4")
                else:
                    subprocess.run(['ffmpeg', '-i', output_path, '-vcodec', 'h264', f"{output_path}_processed.mp4"])
                    st.video(f"{output_path}_processed.mp4")
                # with open(output_path, 'rb') as f:
                #     st.download_button('Download Video', f, file_name=output_path.split("/")[-1])
        
        else:
            output_path = input_path.replace("originals", "predicted")
            if os.path.exists(f"{output_path}_processed.mp4"):
                st.video(f"{output_path}_processed.mp4")
            else:
                subprocess.run(['ffmpeg', '-i', output_path, '-vcodec', 'h264', f"{output_path}_processed.mp4"])
                st.video(f"{output_path}_processed.mp4")
            # with open(output_path, 'rb') as f:
            #     st.download_button('Download Video', f, file_name=output_path.split("/")[-1])
            
        os.remove(output_path)


if __name__ == "__main__":
    app()
