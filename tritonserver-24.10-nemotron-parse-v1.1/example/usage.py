import torch
from PIL import Image, ImageDraw
from transformers import AutoModel, AutoProcessor, AutoTokenizer, AutoConfig, AutoImageProcessor, GenerationConfig
from postprocessing import extract_classes_bboxes, transform_bbox_to_original, postprocess_text

# Load model and processor
model_path = "nvidia/NVIDIA-Nemotron-Parse-v1.1"  # Or use a local path
device = "cuda:0"

model = AutoModel.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(model_path)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

# Load image
image = Image.open("path/to/your/image.jpg")
task_prompt = "</s><s><predict_bbox><predict_classes><output_markdown>"

# Process image
inputs = processor(images=[image], text=task_prompt, return_tensors="pt").to(device)
prompt_ids = processor.tokenizer.encode(task_prompt, return_tensors="pt", add_special_tokens=False).cuda()


generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)
# Generate text
outputs = model.generate(**inputs,  generation_config=generation_config)

# Decode the generated text
generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
