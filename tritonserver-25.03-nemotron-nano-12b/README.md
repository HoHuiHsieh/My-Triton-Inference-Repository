# NVIDIA Nemotron Nano 12B v2 VL

## Model Name
`nvidia-nemotron-nano-12b-v2-vl`

## Model Source
[Hugging Face Model Page](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16)

> **Note**: You need to download the model files from Hugging Face yourself. After downloading, update the model path in `serve.sh` and `config.pbtxt` to point to your local model directory.

## How to Use

### Test the Model

Test with an image:
```bash
python test/test-nemotron-nano.py image /path/to/image.jpg
```

Test with a video:
```bash
python test/test-nemotron-nano.py video /path/to/video.mp4
```

### Python Example - Image Description

```python
import numpy as np
import tritonclient.grpc as grpcclient

# Create Triton client
triton_client = grpcclient.InferenceServerClient(url="0.0.0.0:8001")

# Read image bytes
with open("image.jpg", "rb") as f:
    image_bytes = f.read()

# Prepare input
input_image = grpcclient.InferInput("input_image", [1], "BYTES")
input_image_data = np.array([image_bytes], dtype=object)
input_image.set_data_from_numpy(input_image_data)

# Prepare output
output_text = grpcclient.InferRequestedOutput("output_text")

# Run inference
results = triton_client.infer(
    model_name="nvidia-nemotron-nano-12b-v2-vl",
    inputs=[input_image],
    outputs=[output_text]
)

# Get description
output = results.as_numpy("output_text")[0]
description = output.decode('utf-8') if isinstance(output, bytes) else str(output)
print(description)
```

### Python Example - Video Description

```python
import numpy as np
import tritonclient.grpc as grpcclient

# Create Triton client
triton_client = grpcclient.InferenceServerClient(url="0.0.0.0:8001")

# Read video bytes
with open("video.mp4", "rb") as f:
    video_bytes = f.read()

# Prepare input
input_video = grpcclient.InferInput("input_video", [1], "BYTES")
input_video_data = np.array([video_bytes], dtype=object)
input_video.set_data_from_numpy(input_video_data)

# Prepare output
output_text = grpcclient.InferRequestedOutput("output_text")

# Run inference
results = triton_client.infer(
    model_name="nvidia-nemotron-nano-12b-v2-vl",
    inputs=[input_video],
    outputs=[output_text]
)

# Get description
output = results.as_numpy("output_text")[0]
description = output.decode('utf-8') if isinstance(output, bytes) else str(output)
print(description)
```

### Inputs
- `input_image`: Image file in bytes format (PNG, JPEG, etc.) - for image description
- `input_video`: Video file in bytes format (MP4, AVI, etc.) - for video description

### Outputs
- `output_text`: Generated text description of the image or video
