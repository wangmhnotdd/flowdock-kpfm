FROM pytorch/pytorch:2.2.1-cuda11.8-cudnn8-runtime

LABEL authors="Colby T. Ford <colby@tuple.xyz>"

## Install system requirements
RUN apt update && \
    apt-get install -y --reinstall \
        ca-certificates && \
    apt install -y \
        git \
        wget \
        libxml2 \
        libgl-dev \
        libgl1 \
        gcc \
        g++

## Set working directory
RUN mkdir -p /software/flowdock
WORKDIR /software/flowdock

## Clone project
RUN git clone https://github.com/BioinfoMachineLearning/FlowDock /software/flowdock

## Create conda environment
# RUN conda env create -f environments/flowdock_environment.yaml
COPY environments/flowdock_environment_docker.yaml /software/flowdock/environments/flowdock_environment_docker.yaml
RUN conda env create -f environments/flowdock_environment_docker.yaml

# Install ProDy without NumPy dependency
RUN python -m pip install --no-cache-dir --no-dependencies prody==2.4.1

## Automatically activate conda environment
RUN echo "source activate flowdock" >> /etc/profile.d/conda.sh && \
    echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate flowdock" >> ~/.bashrc

## Default shell and command
SHELL ["/bin/bash", "-l", "-c"]
CMD ["/bin/bash"]
