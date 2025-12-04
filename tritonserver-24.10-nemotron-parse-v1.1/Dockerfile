FROM nvcr.io/nvidia/tritonserver:24.10-pyt-python-py3
RUN pip install --upgrade \
        albumentations==1.4.1 \
        einops==0.8.1 \
        numpy==1.26.4 \
        opencv_python==4.8.0.74 \
        opencv_python_headless==4.9.0.80 \ 
        Pillow==10.2.0 \
        torch==2.1.2 \
        torchvision==0.16.2 \
        transformers==4.51.3 \
        timm==1.0.22 \
        torchmetrics==1.3.1
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade \
        bs4 \
        open-clip-torch
