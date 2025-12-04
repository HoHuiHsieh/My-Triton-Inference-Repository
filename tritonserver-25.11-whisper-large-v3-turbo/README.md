# Whisper-Large-V3-Turbo

## Model Name
`whisper-large-v3-turbo`

## Model Source
[Hugging Face Model Page](https://huggingface.co/openai/whisper-large-v3-turbo)

> **Note**: You need to download the model files from Hugging Face yourself. After downloading, update the model path in `serve.sh` and `config.pbtxt` to point to your local model directory.

## How to Use

### Test the Model

Test with a local audio file:
```bash
python test/test-whisper-large-v3-turbo.py /path/to/audio.wav
```

Test with LibriSpeech dataset sample:
```bash
python test/test-whisper-large-v3-turbo.py
```

### Python Example

```python
import numpy as np
import tritonclient.grpc as grpcclient
import soundfile as sf
import io

# Create Triton client
triton_client = grpcclient.InferenceServerClient(url="0.0.0.0:8001")

# Load audio file
audio_array, sampling_rate = sf.read("audio.wav")

# Convert audio array to WAV bytes
audio_bytes_io = io.BytesIO()
sf.write(audio_bytes_io, audio_array, sampling_rate, format='WAV')
audio_bytes = audio_bytes_io.getvalue()

# Prepare input
input_audio = grpcclient.InferInput("input_audio", [1], "BYTES")
input_audio_data = np.array([audio_bytes], dtype=object)
input_audio.set_data_from_numpy(input_audio_data)

# Prepare output
output = grpcclient.InferRequestedOutput("output_text")

# Run inference
results = triton_client.infer(
    model_name="whisper-large-v3-turbo",
    inputs=[input_audio],
    outputs=[output]
)

# Get transcription
output_text = results.as_numpy("output_text")
transcription = output_text[0].decode('utf-8') if isinstance(output_text[0], bytes) else str(output_text[0])
print(transcription)
```

### Requirements
Install required dependencies:
```bash
pip install soundfile datasets
```

For dataset support, install FFmpeg:
```bash
apt-get install ffmpeg  # Linux
# or
brew install ffmpeg     # macOS
```

### Inputs
- `input_audio`: Audio file in bytes format (WAV, MP3, FLAC, OGG, M4A)

### Outputs
- `output_text`: Transcribed text from the audio

### Tips
- For best results, use 16kHz mono audio files
- Supported formats: WAV, MP3, FLAC, OGG, M4A
