# !/bin/bash
export WORKSPACE=/workspace

# Run the Docker container with the specified configurations
docker run -itd --rm --gpus '"device=0"' \
    --name tensorrtllm-gpt-oss-20b \
    -v ${PWD}/../../Huggingface/gpt-oss-20b:${WORKSPACE}/model/gpt-oss-20b \
    -v ${PWD}:${WORKSPACE}/src \
    -p 8000:8000 \
    -w $WORKSPACE \
    -e NCCL_DEBUG=WARN \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    tensorrt-llm-hf:1.2.0rc5 \
    trtllm-serve ${WORKSPACE}/model/gpt-oss-20b \
        --extra_llm_api_options ${WORKSPACE}/src/gpt-oss-20b-throughput.yaml \
        --host 0.0.0.0 \
        --port 8000