# Use the NVIDIA CUDA base image with cuDNN 7
FROM nvcr.io/nvidia/cuda:10.1-cudnn7-devel-centos7

# Set the working directory inside the container
WORKDIR /container

# Set the LD_LIBRARY_PATH to include CUDA 10.2 and CUDA 10.1 libraries
ENV LD_LIBRARY_PATH=/usr/local/cuda-10.2/targets/x86_64-linux/lib:/usr/local/cuda-10.1/lib64:$LD_LIBRARY_PATH

# Replace CentOS mirrors with vault.centos.org (legacy archive)
RUN sed -i 's|^mirrorlist=|#mirrorlist=|g' /etc/yum.repos.d/CentOS-*.repo && \
    sed -i 's|^#baseurl=http://mirror.centos.org|baseurl=http://vault.centos.org|g' /etc/yum.repos.d/CentOS-*.repo

# Install prerequisites for Python 3.7
RUN yum update -y && \
    yum install -y \
    epel-release && \
    yum install -y \
    gcc gcc-c++ make wget zlib-devel bzip2 bzip2-devel xz-devel \
    libffi-devel sqlite sqlite-devel openssl-devel && \
    yum clean all

# Install Python 3.7 from source
RUN wget https://www.python.org/ftp/python/3.7.12/Python-3.7.12.tgz && \
    tar xzf Python-3.7.12.tgz && \
    cd Python-3.7.12 && \
    ./configure --enable-optimizations && \
    make altinstall && \
    cd .. && \
    rm -rf Python-3.7.12 Python-3.7.12.tgz

# Set Python 3.7 as the default Python version
RUN alternatives --install /usr/bin/python python /usr/local/bin/python3.7 1 && \
    alternatives --set python /usr/local/bin/python3.7 && \
    python --version

# Install Anaconda
RUN wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh && \
    bash Anaconda3-2024.10-1-Linux-x86_64.sh -b -p /opt/anaconda && \
    rm Anaconda3-2024.10-1-Linux-x86_64.sh

# Set the PATH for Conda
ENV PATH=/opt/anaconda/bin:$PATH

# Ensure pip is up to date
RUN python -m ensurepip && \
    python -m pip install --upgrade pip setuptools

# Copy your project files into the container
COPY . /container

# Remove any existing virtual environment
RUN rm -rf /container/.venv_conda

# Create a virtual environment with Python 3.7
RUN conda env create --prefix /container/.venv_conda -f environment_linux.yml

RUN conda run --prefix .venv_conda pip install gias2 && \
    conda run --prefix .venv_conda conda install -y numpy==1.20.1 && \
    conda init bash


# The default command to run when the container starts
CMD ["bash"]
