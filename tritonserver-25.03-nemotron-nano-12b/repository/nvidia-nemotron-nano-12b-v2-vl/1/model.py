import io
import sys
import os
import json
import numpy as np
import glob
import triton_python_backend_utils as pb_utils
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
# Helper function for preparing inputs with images
from frames import prepare_with_images
# Helper function for preparing inputs with video
from video import prepare_with_video


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
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map=model_device,
            torch_dtype=torch.bfloat16
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True)
        self.device = model_device

        # Set video pruning rate for efficient inference
        self.model.video_pruning_rate = 0.75

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
                image_input = pb_utils.get_input_tensor_by_name(request, 
                                                                "input_image")
                video_input = pb_utils.get_input_tensor_by_name(request, 
                                                                "input_video")
                
                if image_input is not None:
                    # Get list of image input
                    image_bytes = image_input.as_numpy()

                    # Prepare with images
                    configs = prepare_with_images(image_bytes, self.tokenizer)
                    inputs = self.processor(**configs).to(self.device)

                    # Generate output
                    generated_ids = self.model.generate(
                        pixel_values=inputs.pixel_values,
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=1024,
                        do_sample=False,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                
                elif video_input is not None:
                    # Get video input if provided
                    video_bytes = video_input.as_numpy()[0]

                    # Prepare with video
                    configs = prepare_with_video(video_bytes, self.tokenizer)
                    inputs = self.processor(**configs).to(self.device)

                    # Generate output
                    generated_ids = self.model.generate(
                        pixel_values_videos=inputs.pixel_values_videos,
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=128,
                    )
                    
                else:
                    raise ValueError("No valid input provided!!")

                # Decode output
                output_text = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False
                )[0]

                # Prepare results
                output_text_tensor = pb_utils.Tensor(
                    "output_text",
                    np.array([output_text]).astype(np.object_)
                )
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[output_text_tensor]
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
