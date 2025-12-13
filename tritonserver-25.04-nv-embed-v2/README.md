# NVIDIA NV-Embed-v2

## Model Name
`nv-embed-v2`

## Model Source
[Hugging Face Model Page](https://huggingface.co/nvidia/NV-Embed-v2)

> **Note**: You need to download the model files from Hugging Face yourself. After downloading, update the model path in `serve.sh` and `config.pbtxt` to point to your local model directory.

## How to Use

### Test the Model
Run the test script:

```bash
python test/test-nv-embed-v2.py
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
    model_name="nv-embed-v2",
    inputs=[input_query],
    outputs=[output_embeddings]
)

# Get embeddings
embeddings = results.as_numpy("embeddings")
print(f"Query embedding shape: {embeddings.shape}")
print(f"Embedding norm: {np.linalg.norm(embeddings):.6f}")  # Should be ~1.0
```

### Python Example - Document Embeddings

```python
import numpy as np
import tritonclient.grpc as grpcclient

# Create Triton client
triton_client = grpcclient.InferenceServerClient(url="localhost:8001")

# Document texts
documents = [
    "Neural networks learn through iterative training.",
    "Deep learning models adjust their weights through backpropagation.",
    "Market prices are determined by supply and demand.",
]

# Prepare input - shape [1, num_documents]
input_documents = grpcclient.InferInput("documents", [1, len(documents)], "BYTES")
documents_data = np.array([documents], dtype=object)
input_documents.set_data_from_numpy(documents_data)

# Prepare output
output_embeddings = grpcclient.InferRequestedOutput("embeddings")

# Run inference
results = triton_client.infer(
    model_name="nv-embed-v2",
    inputs=[input_documents],
    outputs=[output_embeddings]
)

# Get embeddings
embeddings = results.as_numpy("embeddings")
print(f"Document embeddings shape: {embeddings.shape}")
print(f"Number of embeddings: {embeddings.shape[0]}")

# Verify normalized embeddings
for i in range(embeddings.shape[0]):
    norm = np.linalg.norm(embeddings[i])
    print(f"Document {i+1} norm: {norm:.6f}")  # Should be ~1.0
```

### Python Example - Semantic Similarity Search

```python
import numpy as np
import tritonclient.grpc as grpcclient

# Create Triton client
triton_client = grpcclient.InferenceServerClient(url="localhost:8001")

query = "How do neural networks learn patterns from examples?"
documents = [
    "Neural networks learn through iterative training.",
    "Deep learning models adjust their weights through backpropagation.",
    "Market prices are determined by supply and demand.",
    "The recipe for chocolate chip cookies includes flour and sugar.",
]

# Get query embedding
input_query = grpcclient.InferInput("query", [1], "BYTES")
query_data = np.array([query], dtype=object)
input_query.set_data_from_numpy(query_data)

output = grpcclient.InferRequestedOutput("embeddings")
query_result = triton_client.infer(
    model_name="nv-embed-v2",
    inputs=[input_query],
    outputs=[output]
)
query_embedding = query_result.as_numpy("embeddings")

# Get document embeddings
input_docs = grpcclient.InferInput("documents", [1, len(documents)], "BYTES")
docs_data = np.array([documents], dtype=object)
input_docs.set_data_from_numpy(docs_data)

doc_result = triton_client.infer(
    model_name="nv-embed-v2",
    inputs=[input_docs],
    outputs=[output]
)
doc_embeddings = doc_result.as_numpy("embeddings")

# Calculate similarity scores (cosine similarity via dot product)
# Embeddings are already L2-normalized
if query_embedding.ndim == 2:
    query_embedding = query_embedding[0]

scores = []
for i in range(len(documents)):
    # Dot product of normalized vectors = cosine similarity
    score = np.dot(query_embedding, doc_embeddings[i])
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
- **embeddings**: L2-normalized embedding vectors (FP32)
  - For query: shape [1, embedding_dim]
  - For documents: shape [num_documents, embedding_dim]
  - All embeddings have L2 norm ≈ 1.0

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
- **State-of-the-Art Performance**: NVIDIA's latest embedding model with enhanced accuracy
- **L2-Normalized Embeddings**: All embeddings are normalized for direct cosine similarity
- **Batch Document Processing**: Efficiently process multiple documents in a single request
- **Semantic Search**: Optimized for question-answering and information retrieval
- **GPU Acceleration**: CUDA support for fast inference
- **Validated Outputs**: Automatic normalization ensures valid similarity scores in [-1, 1]

### Use Cases
- **Semantic Search**: Find relevant documents for user queries
- **Question Answering**: Retrieve passages that answer questions
- **Document Clustering**: Group similar documents together
- **Similarity Comparison**: Compare text similarity with high precision
- **Information Retrieval**: Build advanced search engines and recommendation systems
- **Retrieval-Augmented Generation (RAG)**: Enhance LLM outputs with relevant context

### Model Characteristics
- **Normalized Embeddings**: L2 norm = 1.0 for all outputs
- **Similarity Metric**: Cosine similarity via dot product
- **Score Range**: [-1, 1] where 1 = identical, 0 = orthogonal, -1 = opposite
