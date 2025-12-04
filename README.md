# My Triton Inference Repository

A collection of AI models deployed on NVIDIA Triton Inference Server for various tasks including document parsing, vision-language understanding, text embedding, and speech recognition.

## Features

- **Document Parsing**: Extract text and detect bounding boxes from document images
- **Vision-Language Understanding**: Generate descriptions for images and videos
- **Text Embeddings**: Create semantic embeddings for queries and documents
- **Speech Recognition**: Transcribe audio files to text with high accuracy
- **Production-Ready**: All models deployed with Triton Inference Server for scalable inference
- **Easy Integration**: Simple gRPC client examples for each model

## Model List

| Model | Description | Triton Version | Directory |
|-------|-------------|----------------|-----------|
| **NVIDIA Nemotron Parse v1.1** | Document parsing with bounding box detection and text extraction | 24.10 | [tritonserver-24.10-nemotron-parse-v1.1](./tritonserver-24.10-nemotron-parse-v1.1) |
| **NVIDIA Nemotron Nano 12B v2 VL** | Vision-language model for image and video description | 25.03 | [tritonserver-25.03-nemotron-nano-12b](./tritonserver-25.03-nemotron-nano-12b) |
| **EmbeddingGemma-300M** | Text embedding model for semantic search and similarity | 25.11 | [tritonserver-25.11-embeddinggemma-300m](./tritonserver-25.11-embeddinggemma-300m) |
| **Whisper-Large-V3-Turbo** | Automatic speech recognition (ASR) for audio transcription | 25.11 | [tritonserver-25.11-whisper-large-v3-turbo](./tritonserver-25.11-whisper-large-v3-turbo) |

## Getting Started

Each model directory contains:
- `Dockerfile`: Container image definition
- `build.sh`: Script to build the container
- `serve.sh`: Script to start the Triton server
- `repository/`: Model repository with configuration
- `test/`: Test scripts with usage examples
- `README.md`: Detailed usage instructions

Navigate to each model's directory for specific setup and usage instructions.

## Contact Information

For questions, feedback, or collaboration:

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/hohuihsieh)

---

*Built with ❤️ using NVIDIA Triton Inference Server*
