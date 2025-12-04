# EmbeddingGemma-300M

## Model Name
`embeddinggemma-300m`

## Model Source
[Hugging Face Model Page](https://huggingface.co/google/embeddinggemma-300m)

> **Note**: You need to download the model files from Hugging Face yourself. After downloading, update the model path in `serve.sh` and `config.pbtxt` to point to your local model directory.

## How to Use

### Test the Model
Run the test script:

```bash
python test/test-embeddinggemma-300m.py
```

### Python Example - Query Embedding

```python
import numpy as np
import tritonclient.grpc as grpcclient

# Create Triton client
triton_client = grpcclient.InferenceServerClient(url="0.0.0.0:8001")

# Query text
query = "Which planet is known as the Red Planet?"

# Prepare input
query_input = grpcclient.InferInput("query", [1], "BYTES")
query_data = np.array([query.encode('utf-8')], dtype=object)
query_input.set_data_from_numpy(query_data)

# Prepare output
output = grpcclient.InferRequestedOutput("embeddings")

# Run inference
results = triton_client.infer(
    model_name="embeddinggemma-300m",
    inputs=[query_input],
    outputs=[output]
)

# Get embeddings
embeddings = results.as_numpy("embeddings")
print(f"Query embedding shape: {embeddings.shape}")  # (1, 768)
```

### Python Example - Document Embeddings

```python
import numpy as np
import tritonclient.grpc as grpcclient

# Create Triton client
triton_client = grpcclient.InferenceServerClient(url="0.0.0.0:8001")

# Document text
document = "Mars, known for its reddish appearance, is often referred to as the Red Planet."

# Prepare input
doc_input = grpcclient.InferInput("documents", [1, 1], "BYTES")
doc_data = np.array([[document.encode('utf-8')]], dtype=object)
doc_input.set_data_from_numpy(doc_data)

# Prepare output
output = grpcclient.InferRequestedOutput("embeddings")

# Run inference
results = triton_client.infer(
    model_name="embeddinggemma-300m",
    inputs=[doc_input],
    outputs=[output]
)

# Get embeddings
embeddings = results.as_numpy("embeddings")
print(f"Document embedding shape: {embeddings.shape}")  # (1, 768)
```

### Calculate Similarity

```python
# Calculate cosine similarity between query and documents
similarity = np.dot(query_embeddings, doc_embeddings.T)
```

### Inputs
- `query`: Query text in bytes format (shape: [1]) - for query embeddings
- `documents`: Document text(s) in bytes format (shape: [1, -1]) - for document embeddings

### Outputs
- `embeddings`: 768-dimensional embedding vectors (FP32)
