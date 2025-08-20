# !/bin/bash
# This script is used to start a Docker container with the Triton Inference Server.
# It mounts the model and engine directories to the container and sets the working directory inside the container.
# Ensure you have the required environment variables set before running this script.
# Usage: ./start-tritonserver.sh

# Parse command line arguments
DEPLOY=true
trtllm-serve ./model \
    --backend pytorch \
    --extra_llm_api_options /root/triton/src/extra_llm_api_options.yaml