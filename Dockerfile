# Image based on Ubuntu 18.04 with:
# - CUDA 9.0 + CUDNN7 for tensorflow 1.12.2 (needed for Endernewton detector)
# - CUDA 10.0 + CUDNN7 (needed for YOLOv3)
# - OpenCV 4 + GStreamer 1.0
# - FFMPEG libraries
# - Boost Python (to wrap C++ code in Python)

ARG UBUNTU_VERSION=18.04
ARG ARCH=
ARG CUDA=10.0
FROM nvidia/cuda${ARCH:+-$ARCH}:${CUDA}-base-ubuntu${UBUNTU_VERSION} as base


###############################################################################
#
#							Python
#
###############################################################################

RUN apt-get update && apt-get install -y \
    python3 \
	python3-dev \
    python3-pip \
	python3-numpy

RUN pip3 --no-cache-dir install --upgrade \
    pip \
    setuptools

# Create a symbolic link so that both "python" and "python3" link to same binary
RUN ln -s $(which python3) /usr/local/bin/python


###############################################################################
#
#							OpenCV + Gstreamer
#
###############################################################################

ENV HOME "/home"

# Install tools
RUN \
  apt-get update && apt-get upgrade -y && \
  apt-get install -y \
    wget \
    unzip \
    build-essential \
    cmake \
    pkg-config \
    autoconf \
    automake

# Install gstreamer and opencv dependencies
RUN \
  apt-get update && apt-get upgrade -y && \
  apt-get install -y libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-doc \
    gstreamer1.0-tools \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev && \
  \
  apt-get install -y \
    libgtk-3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libx265-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    openexr \
    libatlas-base-dev \
    libtbb2 \
    libtbb-dev \
    libdc1394-22-dev

# Download OpenCV and build from source
RUN \
  cd ${HOME} && \
  wget -O ${HOME}/opencv.zip https://github.com/opencv/opencv/archive/4.1.0.zip && \
  unzip ${HOME}/opencv.zip && \
  mv ${HOME}/opencv-4.1.0/ ${HOME}/opencv/ && \
  rm -rf ${HOME}/opencv.zip && \
  wget -O ${HOME}/opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.1.0.zip && \
  unzip ${HOME}/opencv_contrib.zip && \
  mv ${HOME}/opencv_contrib-4.1.0/ ${HOME}/opencv_contrib/ && \
  rm -rf ${HOME}/opencv_contrib.zip && \
  \
  cd ${HOME}/opencv && \
  mkdir build && \
  cd build && \
  cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D OPENCV_GENERATE_PKGCONFIG=YES \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_C_EXAMPLES=ON \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D OPENCV_EXTRA_MODULES_PATH=${HOME}/opencv_contrib/modules \
    -D WITH_GSTREAMER=ON \
    -D WITH_GSTREAMER_0_10=OFF \
    -D BUILD_EXAMPLES=ON .. && \
  \
  cd ${HOME}/opencv/build && \
  make -j $(nproc) && \
  make install && \
  ldconfig


###############################################################################
#
#							CUDA 9.0 (for Tensorflow 1.12.2)
#
###############################################################################

# Taken from: https://github.com/tobycheese/9.0-cudnn7-devel-ubuntu18.04

# CHANGED: below, add the two repos from 17.04 and 16.04 so all packages are found
RUN apt-get update && apt-get install -y --no-install-recommends gnupg2 curl ca-certificates && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1704/x86_64 /" >> /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" >> /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get purge --autoremove -y curl && \
    rm -rf /var/lib/apt/lists/*

ENV CUDA_VERSION 9.0.176

ENV CUDA_PKG_VERSION 9-0=$CUDA_VERSION-1
RUN apt-get update && apt-get install -y --no-install-recommends \
 	--allow-change-held-packages \
    cuda-cudart-$CUDA_PKG_VERSION && \
    ln -s cuda-9.0 /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*

# CHANGED: commented out
# nvidia-docker 1.0
#LABEL com.nvidia.volumes.needed="nvidia_driver"
#LABEL com.nvidia.cuda.version="${CUDA_VERSION}"

RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=9.0"

ENV NCCL_VERSION 2.3.7

RUN apt-get update && apt-get install -y --no-install-recommends \
 	--allow-change-held-packages \
    cuda-libraries-$CUDA_PKG_VERSION \
    cuda-cublas-9-0=9.0.176.4-1 \
    libnccl2=$NCCL_VERSION-1+cuda9.0 && \
    apt-mark hold libnccl2 && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
 	--allow-change-held-packages \
	cuda-libraries-dev-$CUDA_PKG_VERSION \
	cuda-nvml-dev-$CUDA_PKG_VERSION \
	cuda-minimal-build-$CUDA_PKG_VERSION \
	cuda-command-line-tools-$CUDA_PKG_VERSION \
	cuda-core-9-0=9.0.176.3-1 \
	cuda-cublas-dev-9-0=9.0.176.4-1 \
	libnccl-dev=$NCCL_VERSION-1+cuda9.0 && \
    rm -rf /var/lib/apt/lists/*

ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs

ENV CUDNN_VERSION 7.4.1.5
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

RUN apt-get update && apt-get install -y --no-install-recommends \
 	--allow-change-held-packages \
    libcudnn7=$CUDNN_VERSION-1+cuda9.0 \
    libcudnn7-dev=$CUDNN_VERSION-1+cuda9.0 && \
    apt-mark hold libcudnn7 && \
    rm -rf /var/lib/apt/lists/*


###############################################################################
#
#							CUDNN & Tensorflow
#
###############################################################################

ARG ARCH
ARG CUDA
ARG CUDNN=7.4.1.5-1

# Install tensorflow dependencies
SHELL ["/bin/bash", "-c"]
RUN apt-get update && apt-get install -y --no-install-recommends \
 	--allow-change-held-packages \
    build-essential \
    cuda-command-line-tools-${CUDA/./-} \
    cuda-cublas-${CUDA/./-} \
    cuda-cufft-${CUDA/./-} \
    cuda-curand-${CUDA/./-} \
    cuda-cusolver-${CUDA/./-} \
    cuda-cusparse-${CUDA/./-} \
    curl \
    libcudnn7=${CUDNN}+cuda${CUDA} \
    libfreetype6-dev \
    libhdf5-serial-dev \
    libzmq3-dev \
    pkg-config \
    software-properties-common \
    unzip

RUN [ ${ARCH} = ppc64le ] || (apt-get update && \
    apt-get install nvinfer-runtime-trt-repo-ubuntu1804-5.0.2-ga-cuda${CUDA} \
    && apt-get update \
    && apt-get install -y --no-install-recommends libnvinfer5=5.0.2-1+cuda${CUDA} \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*)

# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH}

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

ARG TF_PACKAGE=tensorflow-gpu
ARG TF_PACKAGE_VERSION=1.12.2
RUN pip3 install ${TF_PACKAGE}${TF_PACKAGE_VERSION:+==${TF_PACKAGE_VERSION}}


###############################################################################
#
#							Python Packages
#
###############################################################################

# Install dependencies for SocketIO and matplotlib
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
	netbase \
	libsm6 \
	libxext6 \
	libfontconfig1 \
	libxrender1 \
	python3-tk \
	libcanberra-gtk-module \
	libcanberra-gtk3-module

# Install Python packages
COPY requirements.txt /
RUN pip3 install --upgrade pip
RUN pip3 install -r /requirements.txt


###############################################################################
#
#							Container Startup & Command
#
###############################################################################

WORKDIR /ShopfloorMonitor

COPY docker-entrypoint.sh /ShopfloorMonitor
RUN chmod +x /ShopfloorMonitor/docker-entrypoint.sh
ENTRYPOINT ["./docker-entrypoint.sh"]

CMD ["sh", "-c", "python3 -u main.py"]
