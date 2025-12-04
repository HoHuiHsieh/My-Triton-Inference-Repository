import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

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

# Load multiple images
images = [
    Image.open("path/to/image1.jpg"),
    Image.open("path/to/image2.jpg"),
]

# Prepare messages with multiple images
messages = [
    {"role": "system", "content": "/no_think"},
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "/path/to/image1"},
            {"type": "image", "image": "/path/to/image2"},
            {"type": "text", "text": "\nDescribe the two images in detail."},
        ],
    }
]

# Generate prompt and process inputs
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[prompt], images=images, return_tensors="pt").to(device)

# Generate output
generated_ids = model.generate(
    pixel_values=inputs.pixel_values,
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=1024,
    do_sample=False,
    eos_token_id=tokenizer.eos_token_id,
)

# Decode output
output_text = processor.batch_decode(
    generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
)[0]
print(output_text)
