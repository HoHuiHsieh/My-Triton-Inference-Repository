# My Triton Inference Repository

A collection of AI models deployed on NVIDIA Triton Inference Server for various tasks including document parsing, vision-language understanding, text embedding, and speech recognition.

## Features

- **Large Language Models**: OpenAI-compatible chat API with function calling support
- **Document Parsing**: Extract text and detect bounding boxes from document images
- **Vision-Language Understanding**: Generate descriptions for images and videos
- **Text Embeddings**: Create semantic embeddings for queries and documents
- **Speech Recognition**: Transcribe audio files to text with high accuracy
- **Production-Ready**: All models deployed with optimized inference servers for scalable deployment
- **Easy Integration**: Simple API client examples for each model
- **Performance Benchmarking**: Built-in tools for throughput and latency testing

## Model List

| Model | Description | Framework/Version | Directory |
|-------|-------------|-------------------|-----------|
| **OpenAI GPT OSS 20B** | Large language model with OpenAI-compatible API and function calling | TensorRT-LLM 1.2.0rc5 | [tensorrtllm-1.2.0rc5-openai-gpt-oss-20b](./tensorrtllm-1.2.0rc5-openai-gpt-oss-20b) |
| **NVIDIA Nemotron Parse v1.1** | Document parsing with bounding box detection and text extraction | Triton 24.10 | [tritonserver-24.10-nemotron-parse-v1.1](./tritonserver-24.10-nemotron-parse-v1.1) |
| **NVIDIA Llama Embed Nemotron 8B** | High-quality 8B embedding model for semantic search and Q&A retrieval | Triton 25.03 | [tritonserver-25.03-llama-embed-nemotron-8b](./tritonserver-25.03-llama-embed-nemotron-8b) |
| **NVIDIA Nemotron Nano 12B v2 VL** | Vision-language model for image and video description | Triton 25.03 | [tritonserver-25.03-nemotron-nano-12b](./tritonserver-25.03-nemotron-nano-12b) |
| **NVIDIA NV-Embed-v2** | State-of-the-art embedding model with L2-normalized outputs for RAG and semantic search | Triton 25.04 | [tritonserver-25.04-nv-embed-v2](./tritonserver-25.04-nv-embed-v2) |
| **EmbeddingGemma-300M** | Text embedding model for semantic search and similarity | Triton 25.11 | [tritonserver-25.11-embeddinggemma-300m](./tritonserver-25.11-embeddinggemma-300m) |
| **Whisper-Large-V3-Turbo** | Automatic speech recognition (ASR) for audio transcription | Triton 25.11 | [tritonserver-25.11-whisper-large-v3-turbo](./tritonserver-25.11-whisper-large-v3-turbo) |

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

*Built with ❤️ using NVIDIA Triton Inference Server and TensorRT-LLM*
