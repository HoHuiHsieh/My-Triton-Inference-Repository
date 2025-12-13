# NVIDIA Llama Embed Nemotron 8B

## Model Name
`llama-embed-nemotron-8b`

## Model Source
[Hugging Face Model Page](https://huggingface.co/nvidia/llama-embed-nemotron-8b)

> **Note**: You need to download the model files from Hugging Face yourself. After downloading, update the model path in `serve.sh` and `config.pbtxt` to point to your local model directory.

## How to Use

### Test the Model
Run the test script:

```bash
python test/test-llama-embed-nemotron-8b.py
```

The test script will run three tests:
1. Query embedding generation
2. Document embeddings generation
3. Semantic similarity search

### Python Example - Query Embedding

```python
import numpy as np
import tritonclient.grpc as grpcclient

# Create Triton client
triton_client = grpcclient.InferenceServerClient(url="localhost:8001")

# Query text
query = "How do neural networks learn patterns from examples?"

# Prepare input
input_query = grpcclient.InferInput("query", [1], "BYTES")
query_data = np.array([query], dtype=object)
input_query.set_data_from_numpy(query_data)

# Prepare output
output_embeddings = grpcclient.InferRequestedOutput("embeddings")

# Run inference
results = triton_client.infer(
    model_name="llama-embed-nemotron-8b",
    inputs=[input_query],
    outputs=[output_embeddings]
)

# Get embeddings
embeddings = results.as_numpy("embeddings")
print(f"Query embedding shape: {embeddings.shape}")
print(f"Embedding norm: {np.linalg.norm(embeddings):.6f}")
```

### Python Example - Document Embeddings

```python
import numpy as np
import tritonclient.grpc as grpcclient

# Create Triton client
triton_client = grpcclient.InferenceServerClient(url="localhost:8001")

# Document texts
documents = [
    "Deep learning models adjust their weights through backpropagation.",
    "Market prices are determined by supply and demand.",
    "Neural networks learn through iterative training.",
]

# Prepare input - shape [1, num_documents]
input_documents = grpcclient.InferInput("documents", [1, len(documents)], "BYTES")
documents_data = np.array([documents], dtype=object)
input_documents.set_data_from_numpy(documents_data)

# Prepare output
output_embeddings = grpcclient.InferRequestedOutput("embeddings")

# Run inference
results = triton_client.infer(
    model_name="llama-embed-nemotron-8b",
    inputs=[input_documents],
    outputs=[output_embeddings]
)

# Get embeddings
embeddings = results.as_numpy("embeddings")
print(f"Document embeddings shape: {embeddings.shape}")
print(f"Number of embeddings: {embeddings.shape[0]}")
```

### Python Example - Semantic Similarity Search

```python
import numpy as np
import tritonclient.grpc as grpcclient

# Create Triton client
triton_client = grpcclient.InferenceServerClient(url="localhost:8001")

query = "How do neural networks learn patterns from examples?"
documents = [
    "Deep learning models adjust their weights through backpropagation.",
    "Market prices are determined by supply and demand.",
    "Neural networks learn through iterative training.",
]

# Get query embedding
input_query = grpcclient.InferInput("query", [1], "BYTES")
query_data = np.array([query], dtype=object)
input_query.set_data_from_numpy(query_data)

output = grpcclient.InferRequestedOutput("embeddings")
query_result = triton_client.infer(
    model_name="llama-embed-nemotron-8b",
    inputs=[input_query],
    outputs=[output]
)
query_embedding = query_result.as_numpy("embeddings")

# Get document embeddings
input_docs = grpcclient.InferInput("documents", [1, len(documents)], "BYTES")
docs_data = np.array([documents], dtype=object)
input_docs.set_data_from_numpy(docs_data)

doc_result = triton_client.infer(
    model_name="llama-embed-nemotron-8b",
    inputs=[input_docs],
    outputs=[output]
)
doc_embeddings = doc_result.as_numpy("embeddings")

# Calculate similarity scores (cosine similarity via dot product)
# Embeddings are already normalized
scores = []
for i in range(len(documents)):
    score = np.dot(query_embedding[0], doc_embeddings[i])
    scores.append(score)

# Rank documents by similarity
ranked_indices = np.argsort(scores)[::-1]

print("Ranked documents (most relevant first):")
for rank, idx in enumerate(ranked_indices, 1):
    print(f"{rank}. Score: {scores[idx]:.4f} - {documents[idx][:80]}...")
```

### Server Configuration

Update the GPU device in `config.pbtxt`:
```protobuf
parameters {
    key: "device"
    value: {
        string_value: "cuda:0"  # Change device as needed
    }
}
```

### Inputs
- **query** (optional): Single query text for query embedding (shape: [1])
- **documents** (optional): Multiple document texts for document embeddings (shape: [1, num_documents])

> **Note**: Either `query` or `documents` must be provided, but not both in the same request.

### Outputs
- **embeddings**: Normalized embedding vectors (FP32)
  - For query: shape [1, embedding_dim]
  - For documents: shape [num_documents, embedding_dim]

### Requirements
Install the required Python packages:
```bash
pip install -r test/requirement.txt
```

Main dependencies:
```bash
pip install tritonclient[grpc] numpy
```

### Features
- **High-Quality Embeddings**: State-of-the-art 8B parameter model for semantic understanding
- **Normalized Outputs**: Embeddings are L2-normalized for direct cosine similarity
- **Batch Document Processing**: Process multiple documents efficiently in a single request
- **Semantic Search**: Perfect for question-answering and information retrieval tasks
- **Task-Specific Instructions**: Supports instruction-based embedding generation
- **GPU Acceleration**: CUDA support for fast inference

### Use Cases
- **Semantic Search**: Find relevant documents for user queries
- **Question Answering**: Retrieve passages that answer questions
- **Document Clustering**: Group similar documents together
- **Similarity Comparison**: Compare text similarity for various applications
- **Information Retrieval**: Build search engines and recommendation systems
