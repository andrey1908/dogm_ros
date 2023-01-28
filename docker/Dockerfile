FROM "nvidia/cuda:11.2.2-devel-ubuntu20.04"

ENV DEBIAN_FRONTEND noninteractive

ARG NUM_THREADS=1

# User setup
ARG USER=docker_dogm
ARG UID=1000
ARG GID=1000
ARG PW=user

# Add user and his password
RUN useradd -m ${USER} --uid=${UID} && \
    echo "${USER}:${PW}" | chpasswd && \
    usermod -s /bin/bash ${USER} && \
    usermod -a -G sudo ${USER}

COPY context/ /context
WORKDIR /context

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    lsb-release \
    cmake \
    pkg-config \
    unzip \
    yasm \
    checkinstall \
    git \
    tmux \
    tar \
    vim \
    curl \
    python3-pip \
    qt5-default \
    clang-format \
    clang-tidy \
    libglfw3-dev \
    libglew-dev \
    libglm-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libgtk-3-dev \
    libtbb-dev \
    libatlas-base-dev \
    gfortran && \
    sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
    apt-get update && apt-get install -y \
    ros-noetic-desktop \
    ros-noetic-tf2-eigen && \
    rm -rf /var/lib/apt/lists/*

# Build and install opencv
ARG OPENCV_VERSION=4.5.2
RUN set -x && \
    tar -zxf opencv.tar.gz && \
    tar -zxf opencv_contrib.tar.gz && \
    cd /context/opencv_contrib && \
    git config --global --add safe.directory /context/opencv_contrib && \
    git checkout tags/${OPENCV_VERSION} && \
    cd /context/opencv && \
    git config --global --add safe.directory /context/opencv && \
    git checkout tags/${OPENCV_VERSION} && \
    mkdir build && \
    cd build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local/opencv4.5 \
        -D WITH_TBB=ON \
        -D ENABLE_FAST_MATH=1 \
        -D CUDA_FAST_MATH=1 \
        -D WITH_CUBLAS=1 \
        -D WITH_CUDA=ON \
        -D BUILD_opencv_cudacodec=OFF \
        -D WITH_V4L=ON \
        -D WITH_QT=OFF \
        -D WITH_OPENGL=ON \
        -D WITH_GSTREAMER=ON \
        -D OPENCV_GENERATE_PKGCONFIG=ON \
        -D OPENCV_PC_FILE_NAME=opencv.pc \
        -D OPENCV_ENABLE_NONFREE=ON \
        -D OPENCV_PYTHON3_INSTALL_PATH=/opt/ros/noetic/lib/python3.8/site-packages \
        -D PYTHON_EXECUTABLE=/usr/bin/python3 \
        -D OPENCV_EXTRA_MODULES_PATH=/context/opencv_contrib/modules \
        -D INSTALL_PYTHON_EXAMPLES=OFF \
        -D INSTALL_C_EXAMPLES=OFF \
        -D BUILD_EXAMPLES=OFF .. && \
    make -j${NUM_THREADS} && \
    make install && \
    cd /context && \
    rm -rf opencv opencv_contrib

ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/opencv4.5/lib

RUN mkdir -p /home/${USER}/catkin_ws/src && \
    chown -R ${UID}:${GID} /home/${USER}

WORKDIR /home/${USER}
USER ${UID}:${GID}
