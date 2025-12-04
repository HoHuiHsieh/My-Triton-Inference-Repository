# !/bin/bash

export WORKSPACE=/workspace

# Run the Docker container with the specified configurations
docker run -itd --gpus '"device=0"' \
    --name tritonserver-nemotron-nano-12b \
    -v ${PWD}/../../Huggingface/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16:${WORKSPACE}/model/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16 \
    -v ${PWD}/repository:${WORKSPACE}/repository \
    -v ${PWD}:${WORKSPACE}/src \
    -p 8000:8000 \
    -p 8001:8001 \
    -p 8002:8002 \
    -w $WORKSPACE \
    tritonserver:25.03-nemotron-nano-12b \
    tritonserver --model-repository=${WORKSPACE}/repository \
                 --model-control-mode=poll \
                 --repository-poll-secs=1