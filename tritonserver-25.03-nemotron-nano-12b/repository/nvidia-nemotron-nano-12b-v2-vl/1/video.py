import io
from PIL import Image
import video_io  # Helper module for loading frames


def prepare_with_video(video_bytes, tokenizer):
    """Generate text description from video frames."""
    
    video_stream = io.BytesIO(video_bytes)
    video_stream.seek(0)
    frames = video_io.load_frames_from_bytesio(video_stream)

    # Extract frames from video bytes
    video_fps = 1  # FPS used when extracting frames (for temporal understanding)
    image_urls, metadata = video_io.frames_to_data_urls_with_metadata(frames, video_fps)

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

    # Apply chat template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Return prepared inputs
    if metadata:
        return {
            "text": [prompt],
            "videos": frames,
            "videos_kwargs": {'video_metadata': metadata} if metadata else {},
            "return_tensors": "pt"
        }
    else:
        return {
            "text": [prompt],
            "videos": frames,
            "return_tensors": "pt"
        }
    