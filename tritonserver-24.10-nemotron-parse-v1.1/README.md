# NVIDIA Nemotron Parse v1.1

## Model Name
`nvidia-nemotron-parse-v1.1`

## Model Source
[Hugging Face Model Page](https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.1)

> **Note**: You need to download the model files from Hugging Face yourself. After downloading, update the model path in `serve.sh` and `config.pbtxt` to point to your local model directory.

## How to Use

### Test the Model
Run the test script with a document image:

```bash
python test/test-nemotron-parse.py /path/to/document.png
```

Or run without arguments to use a sample image:

```bash
python test/test-nemotron-parse.py
```

### Python Example
```python
import numpy as np
import tritonclient.grpc as grpcclient

# Create Triton client
triton_client = grpcclient.InferenceServerClient(url="0.0.0.0:8001")

# Read image bytes
with open("document.png", "rb") as f:
    image_bytes = f.read()

# Prepare input
input_image = grpcclient.InferInput("input_image", [1], "BYTES")
input_image_data = np.array([image_bytes], dtype=object)
input_image.set_data_from_numpy(input_image_data)

# Prepare outputs
output_image = grpcclient.InferRequestedOutput("output_image")
output_text = grpcclient.InferRequestedOutput("output_text")

# Run inference
results = triton_client.infer(
    model_name="nvidia-nemotron-parse-v1.1",
    inputs=[input_image],
    outputs=[output_image, output_text]
)

# Get results
output_image_bytes = results.as_numpy("output_image")[0]
output_texts = results.as_numpy("output_text")

# Save output image with bounding boxes
with open("output_parsed.png", "wb") as f:
    f.write(output_image_bytes)

# Print extracted text
for idx, text in enumerate(output_texts, 1):
    text_str = text.decode('utf-8') if isinstance(text, bytes) else str(text)
    print(f"{idx}. {text_str}")
```

### Inputs
- `input_image`: Document image in bytes format (PNG, JPEG, etc.)

### Outputs
- `output_image`: Image with bounding boxes around detected elements
- `output_text`: Array of extracted text from each detected element
