import io
from PIL import Image

def prepare_with_images(image_bytes, tokenizer):
    """Generate text description from video frames."""
    # Load multiple images
    images = []
    for img_bytes in image_bytes:
        img_stream = io.BytesIO(img_bytes)
        img_stream.seek(0)
        try:
            img = Image.open(img_stream)
            img.load()  # Force loading to validate
            # Check and convert image mode if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # Append to images list
            images.append(img)
        except Exception as e:
            error_msg = f"Failed to decode image: {str(e)}. Received {len(image_bytes)} bytes with header {image_bytes[:20]}"
            print(f"ERROR: {error_msg}")
            raise ValueError(error_msg)
        
    # Prepare messages with multiple images
    messages = [
        {"role": "system", "content": "/no_think"},
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": f"image{i}"} for i in range(1, len(images) + 1)],
                {"type": "text", "text": "\nDescribe the images in detail."},
            ],
        }
    ]

    # Apply chat template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Return prepared inputs
    return {
        "text": [prompt],
        "images": images,
        "return_tensors": "pt"
    }