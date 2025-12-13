FROM nvcr.io/nvidia/tritonserver:25.03-pyt-python-py3
RUN pip install --upgrade packaging torch
RUN pip install --upgrade \        
        transformers==4.51.0 \
        flash-attn==2.6.3
RUN pip install --upgrade accelerate