# !/bin/bash

# Load environment variables from .env file
# if [ -f .env ]; then
#     export $(grep -v '^#' .env | xargs)
# fi
export WORKSPACE=/workspace

# Run the Docker container with the specified configurations
docker run -itd --rm --gpus '"device=0"' \
    --name tritonserver-embeddinggemma-300m \
    -v ${PWD}/../../Huggingface/embeddinggemma-300m:${WORKSPACE}/model/embeddinggemma-300m \
    -v ${PWD}/../../Huggingface/whisper-large-v3-turbo:${WORKSPACE}/model/whisper-large-v3-turbo \
    -v ${PWD}/repository:${WORKSPACE}/repository \
    -v ${PWD}:${WORKSPACE}/src \
    -p 8000:8000 \
    -p 8001:8001 \
    -p 8002:8002 \
    -w $WORKSPACE \
    tritonserver:25.11-transformers \
    tritonserver --model-repository=${WORKSPACE}/repository \
                 --model-control-mode=poll \
                 --repository-poll-secs=1