FROM nvcr.io/nvidia/tritonserver:25.11-pyt-python-py3
RUN apt-get update && apt-get install -y ffmpeg
RUN pip install --upgrade \
        transformers \
        datasets[audio] \
        accelerate \
        sentence-transformers
RUN pip install --upgrade \
        librosa \
        soundfile \
        jiwer