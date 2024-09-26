FROM nvidia/cuda:12.1.0-devel-ubuntu20.04

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV CUDNN_VERSION 8.9.3.28
ENV PYTHON_VERSION 3.9
ENV FORCE_CUDA="1"
ENV CUDA_CACHE_DISABLE="1"
ENV TORCH_CUDA_ARCH_LIST="Turing"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CUDA_VERSION_1 12.1
ENV CUDA_VERSION_2 121
ENV TORCH_VERSION_1 2.1.0
ENV TORCHVISION_VERSION 0.16.0
ENV DETECTRON_VERSION 0.6

LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

RUN rm /etc/apt/sources.list.d/cuda.list || true
RUN rm /etc/apt/sources.list.d/nvidia-ml.list || true
RUN apt-key del 7fa2af80
RUN apt-get update && apt-get install -y --no-install-recommends wget
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
RUN apt-get update && apt-get -y install tzdata

RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn8=$CUDNN_VERSION-1+cuda$CUDA_VERSION_1 \
    curl \
    software-properties-common \
    && apt-mark hold libcudnn8 && \
    rm -rf /var/lib/apt/lists/*


RUN add-apt-repository ppa:deadsnakes/ppa -y

RUN apt-get update && apt-get install -y --no-install-recommends\
    git\
    build-essential cmake pkg-config \
    # Python
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    python${PYTHON_VERSION}-distutils \
    # What is this for?
    libffi-dev \
    # Compilers
    cmake\
    gcc\
    g++\
    gfortran\
    ninja-build \
    #
    pkg-config \
    # Codecs
    ffmpeg \
    x264 \
    libx264-dev \
    libavcodec-dev \
    libavformat-dev \
    libxvidcore-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libjpeg-turbo8-dev \
    libwebp-dev \
    libtiff5-dev \
    libopenjp2-7-dev \
    libv4l-dev\
    libswscale-dev\
    libdvdnav4 gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
    # Why? this is a graphics library
    libgtk-3-dev\
    # Compression
    zlib1g-dev \
    unzip\
    # Why? Fonts
    libfreetype6-dev \
    # Why? Text rendering
    libfribidi-dev \
    libharfbuzz-dev \
    libraqm0 \
    # Why? Image processing?
    openexr \
    # Why? Linear algebra
    libatlas-base-dev\
    # For parallel programming
    libtbb2 \
    libtbb-dev \
    # Why? not sure why we need it
    libdc1394-22-dev \
    # SSH
    openssh-client \
    # text editor
    vim \
    screen \
    #
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Installing Python and its dependencies, updating pip to 24 breaks mmpycocotools
RUN python${PYTHON_VERSION} -m pip install --upgrade pip==21.3.1
RUN python${PYTHON_VERSION} -m pip install numpy==1.24.1

RUN mkdir -p /opt/program
RUN mkdir -p /opt/program/data
WORKDIR /opt/program

COPY requirements.txt /opt/program/requirements.txt

# Installing Jupyter
RUN apt-get update && \
    apt-get install -y --no-install-recommends\
    jupyter \
    jupyter-notebook \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python${PYTHON_VERSION} -m pip uninstall -y traitlets
RUN python${PYTHON_VERSION} -m pip install traitlets==5.9.0

# Installing PyTorch
RUN python${PYTHON_VERSION} -m pip install torch==${TORCH_VERSION_1}+cu${CUDA_VERSION_2} -f https://download.pytorch.org/whl/torch_stable.html
RUN python${PYTHON_VERSION} -m pip install torchvision==${TORCHVISION_VERSION}+cu${CUDA_VERSION_2} -f https://download.pytorch.org/whl/torch_stable.html

RUN python${PYTHON_VERSION} -m pip install -r /opt/program/requirements.txt \
    && rm /opt/program/requirements.txt

RUN python${PYTHON_VERSION} -m pip uninstall -y opencv-python opencv-python-headless

RUN git clone -b 4.x https://github.com/opencv/opencv.git
WORKDIR /opt/program/opencv
RUN mkdir build
WORKDIR /opt/program/opencv/build

RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D WITH_FFMPEG=ON \
      -D BUILD_EXAMPLES=OFF \
      -D INSTALL_PYTHON_EXAMPLES=OFF \
      -D INSTALL_C_EXAMPLES=OFF \
      -D BUILD_DOCS=OFF \
      -D BUILD_PERF_TESTS=OFF \
      -D BUILD_TESTS=OFF \
      -D WITH_TBB=ON \
      -D WITH_OPENMP=ON \
      -D WITH_IPP=ON \
      -D CPU_BASELINE=SSE4_2,AVX,AVX2 \
      ..

RUN make -j$(nproc)
RUN make install
RUN ldconfig

ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
ENV PYTHONPATH=/usr/local/lib/python3.x/site-packages:$PYTHONPATH
# Optional: Verify the installation
RUN python3 -c "import cv2; print(cv2.__version__)"

RUN python${PYTHON_VERSION} -m pip list

WORKDIR /opt/program/

COPY . /opt/program/

CMD ["streamlit", "run", "streamlit_app.py"]
