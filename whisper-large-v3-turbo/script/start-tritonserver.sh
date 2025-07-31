# !/bin/bash
# This script is used to start a Docker container with the Triton Inference Server.
# It mounts the model and engine directories to the container and sets the working directory inside the container.
# Ensure you have the required environment variables set before running this script.
# Usage: ./start-tritonserver.sh

# Parse command line arguments
DEPLOY=false
CONTAINER_NAME="triton-whisper-large-v3-turbo"
WORKSPACE=/workspace

# Check for deploy flag
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --deploy)
            DEPLOY=true
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# Remove existing container with the same name if it exists
if [ "$(docker ps -aq -f name=^${CONTAINER_NAME}$)" ]; then
    docker rm -f ${CONTAINER_NAME}
fi


# Determine the command to run based on deploy flag
if [ "$DEPLOY" = true ]; then
    CMD="tritonserver --model-repository=${WORKSPACE}/repository"
else
    CMD="bash"
fi

#  Run the Triton Inference Server container
docker run -itd --gpus "device=0" \
    --name ${CONTAINER_NAME} \
    -v ${PWD}:${WORKSPACE} \
    -p 8000:8000 \
    -p 8001:8001 \
    -p 8002:8002 \
    -w ${WORKSPACE} \
    nvcr.io/nvidia/tritonserver:25.04-python-py3 ${CMD}