# !/bin/bash
# This script is used to start the Triton Inference Server with the Llama 3.1 8B Instruct model.
# It sets up the environment, loads the model and engine directories, and runs the Triton server.
# Ensure you have the required environment variables set before running this script.
# Usage: ./run-tritonserver.sh

# Set the environment variables
export ROOT_DIR=/root/triton
export MODEL_DIR=${ROOT_DIR}/model/whisper-large-v3-turbo
export REPO_DIR=./repo

# Update the repo directory to the latest version
rm -rf ${REPO_DIR}
cp -a ./raw-repository ${REPO_DIR}

# Fill in the template files with the appropriate values
echo "Filling in 'nv-embed-v2/config.pbtxt' template with values..."
python3 ./src/fill_template.py -i ${REPO_DIR}/whisper-large-v3-turbo/config.pbtxt model_path:${MODEL_DIR}
