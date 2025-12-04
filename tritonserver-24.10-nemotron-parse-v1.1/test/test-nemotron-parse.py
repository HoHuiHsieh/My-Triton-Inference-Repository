#!/usr/bin/env python
"""
Test script for NVIDIA Nemotron Parse v1.1 model on Triton Inference Server.
Tests document parsing with bounding box detection and text extraction.
"""
import sys
import os
import numpy as np
import tritonclient.grpc as grpcclient
from PIL import Image, ImageDraw
import io


def test_document_parsing(triton_client, model_name, image_path):
    """Test document parsing with an image."""
    print("\n" + "="*80)
    print("TEST: Document Parsing")
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
    print(f"First 20 bytes: {image_bytes[:20]}")
    
    # Prepare input - input_image has dims [1]
    # Use STRING type as configured in config.pbtxt
    input_image = grpcclient.InferInput("input_image", [1], "BYTES")
    
    # Create numpy array with proper dtype for binary data
    input_image_data = np.array([image_bytes], dtype=object)
    
    print(f"Input array shape: {input_image_data.shape}")
    print(f"Input array dtype: {input_image_data.dtype}")
    print(f"Input data type: {type(input_image_data[0])}")
    print(f"Input data length: {len(input_image_data[0])}")
    
    input_image.set_data_from_numpy(input_image_data)
    
    # Prepare outputs
    output_image = grpcclient.InferRequestedOutput("output_image")
    output_text = grpcclient.InferRequestedOutput("output_text")
    
    # Perform inference
    print("Sending image to Triton server...")
    results = triton_client.infer(
        model_name=model_name,
        inputs=[input_image],
        outputs=[output_image, output_text]
    )
    
    # Get outputs
    output_image_bytes = results.as_numpy("output_image")[0]
    output_texts = results.as_numpy("output_text")
    
    print(f"\n{'='*80}")
    print("PARSING RESULTS:")
    print(f"{'='*80}")
    print(f"Number of detected elements: {len(output_texts)}")
    print(f"\nExtracted Text by Element:")
    print(f"{'-'*80}")
    
    for idx, text in enumerate(output_texts, 1):
        text_str = text.decode('utf-8') if isinstance(text, bytes) else str(text)
        # Truncate long text for display
        if len(text_str) > 200:
            display_text = text_str[:200] + "..."
        else:
            display_text = text_str
        print(f"{idx}. {display_text}")
        print(f"{'-'*80}")
    
    # Save output image with bounding boxes
    output_image_path = image_path.rsplit('.', 1)[0] + '_parsed.png'
    with open(output_image_path, 'wb') as f:
        f.write(output_image_bytes)
    
    print(f"\n✓ Output image with bounding boxes saved to: {output_image_path}")
    print("✓ Document parsing test PASSED")
    
    return output_texts, output_image_bytes


def test_with_sample_images(triton_client, model_name):
    """Test with sample images if available."""
    print("\n" + "="*80)
    print("Searching for sample images...")
    print("="*80)
    
    # Look for common image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.pdf']
    sample_dirs = [
        '../example',
        '.',
        '../',
        '../../'
    ]
    
    found_images = []
    for directory in sample_dirs:
        if os.path.exists(directory):
            for file in os.listdir(directory):
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    full_path = os.path.join(directory, file)
                    found_images.append(full_path)
                    if len(found_images) >= 3:  # Limit to 3 samples
                        break
        if found_images:
            break
    
    if not found_images:
        print("⚠ No sample images found in common directories")
        print("  Place test images (JPG, PNG, PDF) in the example/ or test/ directory")
        return None
    
    print(f"Found {len(found_images)} sample image(s):")
    for img in found_images:
        print(f"  - {img}")
    
    results = []
    for image_path in found_images:
        try:
            result = test_document_parsing(triton_client, model_name, image_path)
            results.append(result)
        except Exception as e:
            print(f"✗ Failed to process {image_path}: {e}")
    
    return results


def create_sample_document():
    """Create a simple sample document image for testing."""
    print("\n" + "="*80)
    print("Creating sample document image...")
    print("="*80)
    
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        # Create a simple document-like image
        width, height = 800, 600
        image = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(image)
        
        # Try to use a default font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
        except:
            font = ImageFont.load_default()
            title_font = font
        
        # Draw title
        draw.text((50, 50), "Sample Document", fill='black', font=title_font)
        
        # Draw some paragraphs
        paragraphs = [
            "This is a sample document created for testing the Nemotron Parse model.",
            "The model can detect and extract text from various document elements.",
            "It can identify paragraphs, tables, figures, and other components.",
        ]
        
        y_position = 120
        for para in paragraphs:
            draw.text((50, y_position), para, fill='black', font=font)
            y_position += 50
        
        # Draw a simple table
        draw.rectangle([50, 300, 750, 400], outline='black', width=2)
        draw.line([50, 350, 750, 350], fill='black', width=1)
        draw.line([400, 300, 400, 400], fill='black', width=1)
        
        draw.text((60, 310), "Header 1", fill='black', font=font)
        draw.text((410, 310), "Header 2", fill='black', font=font)
        draw.text((60, 360), "Data 1", fill='black', font=font)
        draw.text((410, 360), "Data 2", fill='black', font=font)
        
        # Save image
        sample_path = "sample_document.png"
        image.save(sample_path)
        
        print(f"✓ Created sample document: {sample_path}")
        return sample_path
        
    except Exception as e:
        print(f"✗ Failed to create sample document: {e}")
        return None


def main():
    """Main test function."""
    print("="*80)
    print("NVIDIA Nemotron Parse v1.1 Triton Inference Server Test")
    print("="*80)
    
    # Create Triton client
    try:
        triton_client = grpcclient.InferenceServerClient(
            url="0.0.0.0:8001",
            verbose=False,
            ssl=False
        )
        print("✓ Successfully connected to Triton Server at 0.0.0.0:8001")
    except Exception as e:
        print(f"✗ Failed to create Triton client: {str(e)}")
        sys.exit(1)
    
    model_name = "nvidia-nemotron-parse-v1.1"
    
    # Check if model is ready
    try:
        if triton_client.is_model_ready(model_name):
            print(f"✓ Model '{model_name}' is ready")
        else:
            print(f"✗ Model '{model_name}' is not ready")
            sys.exit(1)
    except Exception as e:
        print(f"✗ Error checking model status: {str(e)}")
        sys.exit(1)
    
    try:
        # Test with command line argument if provided
        if len(sys.argv) > 1:
            image_path = sys.argv[1]
            if not os.path.exists(image_path):
                print(f"✗ Image file not found: {image_path}")
                sys.exit(1)
            test_document_parsing(triton_client, model_name, image_path)
        else:
            # Test with sample images
            results = test_with_sample_images(triton_client, model_name)
            
            # If no sample images found, create one
            if not results:
                sample_path = create_sample_document()
                if sample_path and os.path.exists(sample_path):
                    test_document_parsing(triton_client, model_name, sample_path)
        
        # Print statistics
        print("\n" + "="*80)
        print("Inference Statistics")
        print("="*80)
        statistics = triton_client.get_inference_statistics(model_name=model_name)
        print(statistics)
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED ✓")
        print("="*80)
        print("\n📝 Usage Examples:")
        print("  python test-nemotron-parse.py")
        print("    → Tests with sample images or creates a test document")
        print("\n  python test-nemotron-parse.py /path/to/document.jpg")
        print("    → Tests with your own document image")
        print("\n💡 Tips:")
        print("  - Supports: JPG, PNG, PDF, TIFF formats")
        print("  - Best results with clear, high-resolution document images")
        print("  - Output images with bounding boxes are saved as *_parsed.png")
        print("  - Model detects: paragraphs, tables, figures, captions, etc.")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()