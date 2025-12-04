#!/usr/bin/env python
"""
Test script for NVIDIA Nemotron Nano model on Triton Inference Server.
"""
import sys
import os
import numpy as np
import tritonclient.grpc as grpcclient
from PIL import Image
import io


def test_with_image(triton_client, model_name, image_path):
    """Test the model with an image input."""
    print("\n" + "="*80)
    print("TEST: Image Description")
    print("="*80)
    
    # Load image
    print(f"Loading image: {image_path}")
    
    # Read raw image bytes
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    
    # Validate it's a proper image
    try:
        image = Image.open(image_path)
        print(f"Image size: {image.width}x{image.height}")
        print(f"Image format: {image.format}")
        print(f"Image mode: {image.mode}")
    except Exception as e:
        print(f"Error validating image: {e}")
        raise
    
    print(f"Image bytes size: {len(image_bytes)} bytes")
    
    # Prepare input - input_image expects multiple images with dims [-1]
    input_image = grpcclient.InferInput("input_image", [1], "BYTES")
    
    # Create numpy array with proper dtype for binary data
    input_image_data = np.array([image_bytes], dtype=object)
    
    print(f"Input array shape: {input_image_data.shape}")
    print(f"Input array dtype: {input_image_data.dtype}")
    
    input_image.set_data_from_numpy(input_image_data)
    
    # Prepare output
    output_text = grpcclient.InferRequestedOutput("output_text")
    
    # Perform inference
    print("Sending image to Triton server...")
    results = triton_client.infer(
        model_name=model_name,
        inputs=[input_image],
        outputs=[output_text]
    )
    
    # Get output
    output_text_result = results.as_numpy("output_text")[0]
    text_str = output_text_result.decode('utf-8') if isinstance(output_text_result, bytes) else str(output_text_result)
    
    print(f"\n{'='*80}")
    print("IMAGE DESCRIPTION RESULTS:")
    print(f"{'='*80}")
    print(text_str)
    print(f"{'='*80}")
    
    print("\n✓ Image description test PASSED")
    
    return text_str


def test_with_video(triton_client, model_name, video_path):
    """Test the model with a video input."""
    print("\n" + "="*80)
    print("TEST: Video Description")
    print("="*80)
    
    # Load video
    print(f"Loading video: {video_path}")
    
    # Read raw video bytes
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    
    print(f"Video bytes size: {len(video_bytes)} bytes")
    
    # Prepare input - input_video expects dims [1]
    input_video = grpcclient.InferInput("input_video", [1], "BYTES")
    
    # Create numpy array with proper dtype for binary data
    input_video_data = np.array([video_bytes], dtype=object)
    
    print(f"Input array shape: {input_video_data.shape}")
    print(f"Input array dtype: {input_video_data.dtype}")
    
    input_video.set_data_from_numpy(input_video_data)
    
    # Prepare output
    output_text = grpcclient.InferRequestedOutput("output_text")
    
    # Perform inference
    print("Sending video to Triton server...")
    results = triton_client.infer(
        model_name=model_name,
        inputs=[input_video],
        outputs=[output_text]
    )
    
    # Get output
    output_text_result = results.as_numpy("output_text")[0]
    text_str = output_text_result.decode('utf-8') if isinstance(output_text_result, bytes) else str(output_text_result)
    
    print(f"\n{'='*80}")
    print("VIDEO DESCRIPTION RESULTS:")
    print(f"{'='*80}")
    print(text_str)
    print(f"{'='*80}")
    
    print("\n✓ Video description test PASSED")
    
    return text_str


def main():
    """Main function to run tests."""
    # Configuration
    triton_url = os.getenv("TRITON_URL", "localhost:8001")
    model_name = "nvidia-nemotron-nano-12b-v2-vl"
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Test file paths
    image_path = os.path.join(script_dir, "test.png")
    video_path = os.path.join(script_dir, "test.mp4")
    
    print("="*80)
    print("NVIDIA Nemotron Nano 12B v2 VL - Triton Inference Server Tests")
    print("="*80)
    print(f"Triton Server URL: {triton_url}")
    print(f"Model Name: {model_name}")
    
    try:
        # Create Triton client
        triton_client = grpcclient.InferenceServerClient(url=triton_url)
        
        # Check server health
        if not triton_client.is_server_live():
            print("ERROR: Triton server is not live!")
            sys.exit(1)
        
        if not triton_client.is_server_ready():
            print("ERROR: Triton server is not ready!")
            sys.exit(1)
        
        # Check if model is ready
        if not triton_client.is_model_ready(model_name):
            print(f"ERROR: Model '{model_name}' is not ready!")
            sys.exit(1)
        
        print("✓ Server is live and ready")
        print(f"✓ Model '{model_name}' is ready")
        
        # Run tests
        tests_passed = 0
        tests_failed = 0
        
        # Test with image
        if os.path.exists(image_path):
            try:
                test_with_image(triton_client, model_name, image_path)
                tests_passed += 1
            except Exception as e:
                print(f"\n✗ Image test FAILED: {e}")
                import traceback
                traceback.print_exc()
                tests_failed += 1
        else:
            print(f"\nSkipping image test: {image_path} not found")
        
        # Test with video
        if os.path.exists(video_path):
            try:
                test_with_video(triton_client, model_name, video_path)
                tests_passed += 1
            except Exception as e:
                print(f"\n✗ Video test FAILED: {e}")
                import traceback
                traceback.print_exc()
                tests_failed += 1
        else:
            print(f"\nSkipping video test: {video_path} not found")
        
        # Print summary
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        print(f"Tests Passed: {tests_passed}")
        print(f"Tests Failed: {tests_failed}")
        print("="*80)
        
        if tests_failed > 0:
            sys.exit(1)
        else:
            print("\n✓ All tests PASSED!")
            sys.exit(0)
            
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()