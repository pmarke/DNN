FROM nvidia/cuda:12.0.1-devel-ubuntu22.04 AS dnn_dev

LABEL author="Mark Petersen"
LABEL version="1.0"
LABEL description="This docker file is intended to create a container for practicing DNN. The main libraries that it contains are torch and opencv with cuda and python enabled."

ARG USERNAME=artemis
ARG USER_UID
ARG USER_GID
RUN echo "My UID : $USER_UID"
RUN echo "My GID : $USER_GID"

# Add the user
RUN groupadd --gid $USER_GID $USERNAME
RUN useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME && usermod -a -G dialout,plugdev $USERNAME
RUN mkdir -p /etc/sudoers.d
# Give the user root access without needing a password
RUN echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME
# RUN mkdir /home/$USERNAME/bin && chown $USERNAME:$USERNAME /home/$USERNAME/bin


RUN apt-get update && apt-get upgrade -y 

RUN apt-get install -yq \
    wget \
    unzip \
    git \
    cmake \ 
    gcc \ 
    g++ \
    # install opencv dependencies
    libavcodec-dev \
    libavutil-dev \
    libavformat-dev \
    libswscale-dev \
    libgstreamer-plugins-base1.0-dev \ 
    libgstreamer1.0-dev \
    libgtk-3-dev \
    libjpeg-dev \
    libpng-dev \
    libopenexr-dev \
    libtiff-dev \
    libtiff5-dev \
    libwebp-dev \
    zlib1g-dev \
    ffmpeg \ 
    libglew-dev \
    libpostproc-dev \
    cudnn9-cuda-12 \
    libeigen3-dev \
    libtbb-dev \
    libgtk2.0-dev \
    pkg-config \
    # install python
    python3 \
    python3-pip \
    python3-dev \
    && apt autoremove -y \
    && apt clean -y


RUN pip3 install --upgrade pip \ 
    numpy \
    scipy \
    matplotlib \
    pandas \
    torch \
    torchvision \
    torchaudio 

RUN apt-get -y install sudo

################################################
# Install OpenCV
################################################
# Version 4.9+ is needed to be compatible with cuda 12
ARG OPENCV_VERSION="4.10.0"  
ARG CUDA_VERSION="12.4"
ARG INSTALL_DIR=/usr/local

WORKDIR /opencv
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip \
    && wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip \
    && unzip opencv.zip \
    && unzip opencv_contrib.zip \
    && mv opencv-${OPENCV_VERSION} opencv \
    && mv opencv_contrib-${OPENCV_VERSION} opencv_contrib

RUN mkdir /opencv/opencv/build
WORKDIR /opencv/opencv/build

RUN cmake -D CMAKE_BUILD_TYPE=Release \
-D CMAKE_INSTALL_PREFIX=$INSTALL_DIR \
-D OPENCV_EXTRA_MODULES_PATH=/opencv/opencv_contrib/modules \
-D WITH_CUDA=ON \
-D ENABLE_FAST_MATH=1 \
-D CUDA_FAST_MATH=1 \
-D WITH_CUBLAS=1 \
-D BUILD_TESTS=OFF \
-D WITH_TBB=ON \
-D WITH_V4L=ON \
-D WITH_OPENGL=ON \
-D OPENCV_DNN_CUDA=ON \
-D PYTHON3_EXECUTABLE=$(which python3) \
-D PYTHON3_INCLUDE_DIR=$(python3 -c "from sysconfig import get_paths as gp; print(gp()['include'])") \
-D PYTHON3_LIBRARY=$(python3 -c "from distutils.sysconfig import get_config_var as gcv; print(gcv('LIBDIR'))") \
-D BUILD_EXAMPLES=ON .. \
    && make -j$(nproc) && make install && ldconfig

WORKDIR $APP_USER_HOME

RUN pip3 install torchvision
RUN pip3 install tensorboard 

# Qt 5 dependencies
RUN apt install -yq \
build-essential \ 
libgl1-mesa-dev \
libxkbcommon-x11-0 \
libxcb-image0 \
libxcb-keysyms1 \
libxcb-render-util0 \
libxcb-xinerama0 \
libxcb-icccm4 \
libxcb-cursor0

# matplotlib dependencies
RUN pip3 install tk \ 
    PyGObject \
    pycairo \
    Tornado \
    PyQt5

# flash attention
RUN pip3 install ninja
RUN MAKEFLAGS="-j6" pip3 install --no-build-isolation flash-attn
RUN pip3 install thop torchinfo pycocotools
RUN pip3 install tqdm

# sets the work directory for the subsequent instructions. If the
# directory does not exist, it will be created. 
WORKDIR /home/$USERNAME/DNN

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME