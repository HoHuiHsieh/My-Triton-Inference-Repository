import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

import video_io  # Helper module for loading frames

# Load model and processor
model_path = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"  # Or use a local path
device = "cuda:0"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map=device,
    torch_dtype=torch.bfloat16
).eval()
tokenizer = AutoTokenizer.from_pretrained(model_path)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

# Load frames from directory
frames_dir = "path/to/frames_directory"
video_fps = 1  # FPS used when extracting frames (for temporal understanding)

frames = video_io.load_frames_from_directory(frames_dir)
image_urls, metadata = video_io.frames_to_data_urls_with_metadata(frames, video_fps)

print(f"Loaded {len(frames)} frames, metadata: {metadata}")

# Prepare messages
messages = [
    {"role": "system", "content": "/no_think"},
    {
        "role": "user",
        "content": [
            {"type": "video", "video": ""},
            {"type": "text", "text": "\nDescribe what you see."},
        ],
    }
]

# Generate prompt and process inputs
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Process with FPS metadata for better temporal understanding
if metadata:
    inputs = processor(
        text=[prompt],
        videos=frames,
        videos_kwargs={'video_metadata': metadata},
        return_tensors="pt",
    )
else:
    inputs = processor(
        text=[prompt],
        videos=frames,
        return_tensors="pt",
    )
inputs = inputs.to(device)

# Set video pruning rate for efficient inference
model.video_pruning_rate = 0.75

# Generate output
generated_ids = model.generate(
    pixel_values_videos=inputs.pixel_values_videos,
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=128,
)

# Decode output
output_text = processor.batch_decode(
    generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
)[0]
print(output_text)
