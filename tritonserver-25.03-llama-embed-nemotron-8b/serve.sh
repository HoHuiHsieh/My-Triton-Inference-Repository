# !/bin/bash
export WORKSPACE=/workspace

# Run the Docker container with the specified configurations
docker run -itd --gpus '"device=0"' \
    --name tritonserver-llama-embed-nemotron-8b \
    -v ${PWD}/../../Huggingface/llama-embed-nemotron-8b:${WORKSPACE}/model/llama-embed-nemotron-8b \
    -v ${PWD}/repository:${WORKSPACE}/repository \
    -v ${PWD}:${WORKSPACE}/src \
    -p 8000:8000 \
    -p 8001:8001 \
    -p 8002:8002 \
    -w $WORKSPACE \
    tritonserver:25.03-llama-embed-nemotron-8b \
    tritonserver --model-repository=${WORKSPACE}/repository \
                 --model-control-mode=poll \
                 --repository-poll-secs=1