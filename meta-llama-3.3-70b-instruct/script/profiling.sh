#!/bin/bash
# This script is used to profile the Llama 3.3 70B Instruct model using Triton Inference Server with TensorRT-LLM backend.
# It sets up the environment, installs the necessary dependencies, and runs the profiling command.

# Ensure you have the required environment variables set before running this script.
# pip install git+https://github.com/triton-inference-server/triton_cli.git

# Run the profiling command with the specified parameters.
# Adjust the parameters as needed for your profiling requirements.
triton profile -m llama-3.3-70b-instruct \
    --backend tensorrtllm \
    --service-kind triton \
    --streaming \
    --measurement-interval 10000 \
    --output-tokens-mean 200 \
    --output-tokens-stddev 0 \
    --synthetic-input-tokens-mean 100 \
    --synthetic-input-tokens-stddev 0
