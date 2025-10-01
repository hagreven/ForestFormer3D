FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Set as non-interactve
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies for Ubuntu 22.04
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libxrender-dev cmake \
        build-essential software-properties-common wget curl \
        python3-dev python3-pip python3-setuptools python-is-python3 \
        libopenblas-dev libjpeg-dev libpng-dev libtiff-dev \
        libavcodec-dev libavformat-dev libswscale-dev && \
    rm -rf /var/lib/apt/lists/*

# CUDA toolkit environment setup
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install debug tools
RUN pip install debugpy

# OpenMMLab Core Libraries
RUN pip install --no-deps \
    mmengine==0.8.0 \
    mmdet==3.2.0 \
    mmsegmentation==1.0.0 \
    mmdet3d==1.4.0
    #git+https://github.com/open-mmlab/mmdetection3d.git@22aaa47fdb53ce1870ff92cb7e3f96ae38d17f61

# Install MMCV for CUDA 12.1 (compatible with torch 2.1)
RUN pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html --no-deps

# Install MinkowskiEngine (with CUDA 12.x support)
ENV TORCH_CUDA_ARCH_LIST="9.0"  
# H100
RUN git clone https://github.com/NVIDIA/MinkowskiEngine.git \
    && cd MinkowskiEngine \
    && git fetch origin pull/567/head:fix-for-cuda-12.2 \
    && git checkout fix-for-cuda-12.2 \
    && export CXX=c++; export CUDA_HOME=/usr/local/cuda; python setup.py install --blas=openblas --force_cuda

# Compile torch-scatter with CUDA support
# RUN git clone https://github.com/rusty1s/pytorch_scatter.git && \
#     cd pytorch_scatter && \
#     git checkout tags/2.0.9 -b v2.0.9 && \
#     TORCH_CUDA_ARCH_LIST="9.0" FORCE_CUDA=1 pip install .

# Install torch-scatter (2.1.2) and torch-cluster (1.6.3) with CUDA support
RUN export FORCE_CUDA=1; pip install --no-cache-dir torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu122.html

# Install ScanNet superpoint segmentator
RUN git clone https://github.com/Karbo123/segmentator.git /workspace/segmentator && \
    cd /workspace/segmentator/csrc && \
    git reset --hard 76efe46d03dd27afa78df972b17d07f2c6cfb696 && \
    # Overwrite C++ standard version from C++14 to C++17 in the CMakeLists.txt
    # Necessary because PyTorch 2.1.0 requires at least C++17 and C++14 triggered error
    sed -i 's/set(CMAKE_CXX_STANDARD 14)/set(CMAKE_CXX_STANDARD 17)/' CMakeLists.txt && \
    mkdir build && cd build && \
    cmake .. \
        -DCMAKE_PREFIX_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path())') \
        -DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
        -DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR') + '/libpython3.so')") \
        -DCMAKE_INSTALL_PREFIX=$(python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())') && \
    make && make install

# Install Python packages
RUN pip install --no-deps \
    spconv-cu120==2.3.6 \
    addict==2.4.0 \
    yapf==0.33.0 \
    termcolor==2.3.0 \
    packaging==23.1 \
    numpy==1.24.1 \
    rich==13.3.5 \
    opencv-python==4.7.0.72 \
    pycocotools==2.0.6 \
    shapely==1.8.5 \
    scipy==1.10.1 \
    terminaltables==3.1.10 \
    numba==0.57.0 \
    llvmlite==0.40.0 \
    pccm==0.4.7 \
    ccimport==0.4.4 \
    pybind11==2.10.4 \
    ninja==1.11.1 \
    lark==1.1.5 \
    cumm-cu120==0.6.3 \
    pyquaternion==0.9.9 \
    lyft-dataset-sdk==0.0.8 \
    pandas==2.0.1 \
    python-dateutil==2.8.2 \
    matplotlib==3.5.2 \
    pyparsing==3.0.9 \
    cycler==0.11.0 \
    kiwisolver==1.4.4 \
    scikit-learn==1.2.2 \
    joblib==1.2.0 \
    threadpoolctl==3.1.0 \
    cachetools==5.3.0 \
    nuscenes-devkit==1.1.10 \
    trimesh==3.21.6 \
    open3d==0.17.0 \
    plotly==5.18.0 \
    dash==2.14.2 \
    plyfile==1.0.2 \
    flask==3.0.0 \
    werkzeug==3.0.1 \
    click==8.1.7 \
    blinker==1.7.0 \
    itsdangerous==2.1.2 \
    importlib_metadata==2.1.2 \
    zipp==3.17.0 \
    tensorboard==2.15.1 \
    tensorboard-data-server==0.7.2 \
    protobuf \
    absl-py \
    future \
    MarkupSafe==2.0.1 \
    markdown \
    grpcio \
    google-auth-oauthlib \
    google-auth \
    requests-oauthlib \
    oauthlib \
    "laspy[lazrs]" \
    portalocker==3.2.0

# Torch points kernels
RUN export FORCE_CUDA=1; export TORCH_CUDA_ARCH_LIST="9.0"; pip install --no-deps --no-cache-dir torch-points-kernels==0.7.0

# Torch-cluster reinstallation (clean)
# RUN pip uninstall -y torch-cluster && \
#     pip install --no-deps --no-cache-dir torch-cluster

# Replace the following files with updated versions
COPY replace_mmdetection_files/loops.py /opt/conda/lib/python3.10/site-packages/mmengine/runner/
COPY replace_mmdetection_files/base_model.py /opt/conda/lib/python3.10/site-packages/mmengine/model/base_model/
COPY replace_mmdetection_files/transforms_3d.py /opt/conda/lib/python3.10/site-packages/mmdet3d/datasets/transforms/

# Update for compatibility with Python 3.10+ importlib.metadata entry point handling
RUN pip install --upgrade setuptools importlib_metadata fsspec tensorboard

# Set PYTHONPATH
ENV PYTHONPATH=/workspace

# Keep container running
CMD ["bash", "-c", "while true; do sleep 1000; done"]
