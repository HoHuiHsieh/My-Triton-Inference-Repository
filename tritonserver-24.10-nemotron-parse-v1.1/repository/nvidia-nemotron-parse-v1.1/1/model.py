import io
import sys
import os
import json
import numpy as np
import glob
import triton_python_backend_utils as pb_utils
import torch
from PIL import Image, ImageDraw
from transformers import AutoModel, AutoProcessor, AutoTokenizer, AutoConfig, AutoImageProcessor, GenerationConfig
from postprocessing import extract_classes_bboxes, transform_bbox_to_original, postprocess_text


class TritonPythonModel:

    def initialize(self, args):
        """
        This function allows the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        # Parse model configs
        model_config = json.loads(args["model_config"])
        model_path = model_config["parameters"]["model_path"]["string_value"]
        model_device = model_config["parameters"]["device"]["string_value"]
       # Load model with tokenizer on specified device
        print(
            f"Initializing NVIDIA-Nemotron-Parse-v1.1 model on device: {model_device}")
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        ).to(model_device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True)
        self.generation_config = GenerationConfig.from_pretrained(
            model_path, trust_remote_code=True)
        self.model_device = model_device

    def execute(self, requests):
        """
        This function is called when an inference is requested for this model. 

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for idx, request in enumerate(requests):
            try:
                # Get image input
                image_input = pb_utils.get_input_tensor_by_name(request, "input_image")
                image_bytes = image_input.as_numpy()[0]

                # Create BytesIO and reset position
                image_stream = io.BytesIO(image_bytes)
                image_stream.seek(0)

                try:
                    image = Image.open(image_stream)
                    image.load()  # Force loading to validate

                    # Check and convert image mode if necessary
                    if image.mode != 'RGB':
                        image = image.convert('RGB')

                    # # Resize and crop image if needed
                    # # Max Input Resolution (Width, Height): 1648, 2048
                    # # Min Input Resolution (Width, Height): 1024, 1280
                    # # Maintaining aspect ratio
                    # max_width, max_height = 1648, 2048
                    # min_width, min_height = 1024, 1280
                    # if image.width > max_width or image.height > max_height:
                    #     image.thumbnail((max_width, max_height), Image.LANCZOS)
                    # elif image.width < min_width or image.height < min_height:
                    #     image = image.resize((min_width, min_height), Image.LANCZOS)
                    # # Cut width and height to be within limits while maintaining aspect ratio
                    # image = image.crop((0, 0, min(image.width, max_width), min(image.height, max_height)))
                    
                except Exception as e:
                    error_msg = f"Failed to decode image: {str(e)}. Received {len(image_bytes)} bytes with header {image_bytes[:20]}"
                    print(f"ERROR: {error_msg}")
                    raise ValueError(error_msg)

                # Define task prompt
                task_prompt = "</s><s><predict_bbox><predict_classes><output_markdown>"

                # Prepare model inputs
                inputs = self.processor(
                    images=image,
                    text=task_prompt,
                    return_tensors="pt"
                ).to(self.model_device)

                # Generate text
                outputs = self.model.generate(
                    **inputs,
                    generation_config=self.generation_config
                )

                # Decode the generated text
                generated_text = self.processor.batch_decode(
                    outputs,
                    skip_special_tokens=True
                )[0]

                # Post-process to extract classes and bounding boxes
                classes, bboxes, texts = extract_classes_bboxes(generated_text)
                bboxes = [
                    transform_bbox_to_original(bbox, image.width, image.height)
                    for bbox in bboxes
                ]

                # Specify output formats for postprocessing
                table_format = 'markdown'  # latex | HTML | markdown
                text_format = 'markdown'  # markdown | plain
                blank_text_in_figures = False  # remove text inside 'Picture' class
                texts = [
                    postprocess_text(
                        text,
                        cls=cls,
                        table_format=table_format,
                        text_format=text_format,
                        blank_text_in_figures=blank_text_in_figures)
                    for text, cls in zip(texts, classes)
                ]
                draw = ImageDraw.Draw(image)
                for bbox in bboxes:
                    draw.rectangle((bbox[0], bbox[1], bbox[2], bbox[3]), outline="red")
                
                # Convert image back to bytes
                output_image_io = io.BytesIO()
                image.save(output_image_io, format='PNG')
                output_image = output_image_io.getvalue()

                # Prepare results
                output_image_tensor = pb_utils.Tensor(
                    "output_image", 
                    np.array([output_image]).astype(np.object_)
                )
                output_text_tensor = pb_utils.Tensor(
                    "output_text", 
                    np.array(texts).astype(np.object_)
                )
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[output_image_tensor, output_text_tensor]
                )
                responses.append(inference_response)

            except Exception as error:
                import traceback
                error_msg = f"Error processing request: {type(error).__name__}: {str(error)}"
                print(f"ERROR: {error_msg}")
                traceback.print_exc()
                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(error_msg)
                ))

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """
        This function allows the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")
