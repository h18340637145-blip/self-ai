# Python 3.12 基础镜像
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 安装常用系统工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 更新 Python 基础工具
RUN python -m pip install --no-cache-dir --upgrade \
    pip \
    setuptools \
    wheel

# 安装 PyTorch
RUN python -m pip install --no-cache-dir \
    torch \
    torchvision \
    torchaudio

# 安装 AI / 数据科学工具
RUN python -m pip install --no-cache-dir \
    numpy \
    pandas \
    scikit-learn \
    matplotlib \
    jupyter \
    transformers \
    datasets \
    accelerate \
    huggingface_hub \
    safetensors
# 安装 Qdrant 客户端
RUN python -m pip install --no-cache-dir qdrant-client

# 安装 flask
RUN python -m pip install --no-cache-dir flask
WORKDIR /workspace

VOLUME ["/workspace", "/models"]

EXPOSE 8888

CMD ["python"]