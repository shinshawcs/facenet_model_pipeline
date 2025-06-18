FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3.10-distutils \
    python3-pip \
    curl git unzip wget \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.10 /usr/local/bin/python && \
    ln -s /usr/bin/pip3 /usr/local/bin/pip

RUN pip install --upgrade pip && \
    pip install torch==2.2.2+cu118 torchvision==0.17.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir --prefer-binary && \
    pip install onnx --no-cache-dir

COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

COPY TensorRT-8.6.1.6 /opt/TensorRT
ENV LD_LIBRARY_PATH=/opt/TensorRT/lib:$LD_LIBRARY_PATH
ENV PATH=/opt/TensorRT/bin:$PATH
ENV PYTHONPATH=/opt/TensorRT/python:$PYTHONPATH
RUN pip install --no-index --find-links=/opt/TensorRT/python tensorrt && \
    ln -s /opt/TensorRT/bin/trtexec /usr/local/bin/trtexec

COPY . .   
CMD ["python", "facenet_compile.py"]