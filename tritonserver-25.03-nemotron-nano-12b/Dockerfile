FROM nvcr.io/nvidia/tritonserver:25.03-pyt-python-py3
RUN pip install --upgrade \
        causal_conv1d \
        "transformers>4.53,<4.54" \
        torch \
        timm \
        "mamba-ssm==2.2.5" \
        accelerate \
        open_clip_torch \
        numpy \
        pillow
RUN apt-get update && apt-get install -y \
        ffmpeg \
        libsm6 \
        libxext6
RUN pip install --upgrade \
        opencv-python-headless