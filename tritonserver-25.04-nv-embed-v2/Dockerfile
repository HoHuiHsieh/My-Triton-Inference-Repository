FROM nvcr.io/nvidia/tritonserver:25.04-pyt-python-py3
RUN pip3 install torch==2.2.0 \
                 transformers==4.42.4 \
                 packaging \
                 datasets
RUN pip3 install --upgrade flash-attn==2.2.0
