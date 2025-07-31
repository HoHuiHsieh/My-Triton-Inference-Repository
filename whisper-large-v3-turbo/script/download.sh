#!/bin/bash
# This script is used to download the model from Hugging Face.

# Ensure you have git and git-lfs installed before running this script.
apt update && apt install -y git git-lfs

# Initialize git-lfs if not already done
git lfs install

# Clone the model repository from Hugging Face
git clone https://huggingface.co/openai/whisper-large-v3-turbo

# Move the downloaded model to the desired directory
mv -r ./whisper-large-v3-turbo ./model