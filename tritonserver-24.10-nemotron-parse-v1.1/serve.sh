# !/bin/bash

export WORKSPACE=/workspace

# Run the Docker container with the specified configurations
docker run -itd --gpus '"device=0"' \
    --name tritonserver-nemotron-parse-v1.1 \
    -v ${PWD}/../../Huggingface/NVIDIA-Nemotron-Parse-v1.1:${WORKSPACE}/model/NVIDIA-Nemotron-Parse-v1.1 \
    -v ${PWD}/repository:${WORKSPACE}/repository \
    -v ${PWD}:${WORKSPACE}/src \
    -p 8000:8000 \
    -p 8001:8001 \
    -p 8002:8002 \
    -w $WORKSPACE \
    tritonserver:24.10-nemotron-parse-v1.1 \
    tritonserver --model-repository=${WORKSPACE}/repository \
                 --model-control-mode=poll \
                 --repository-poll-secs=1