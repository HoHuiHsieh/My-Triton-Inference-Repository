import sys
import os
import json
import numpy as np
import glob
import triton_python_backend_utils as pb_utils
import torch
import torch.nn.functional as F
from transformers import pipeline


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

        # load model with tokenizer
        self.pipe = pipeline(
            "text-generation",
            model=model_path,
            torch_dtype="auto",
            device_map="auto",
        )

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
                # Get inputs
                # - messages: List of messages in JSON format
                messages_tensor = pb_utils.get_input_tensor_by_name(
                    request, "messages")
                messages_json_string = messages_tensor.as_numpy()
                messages_json_string = messages_json_string[0].decode("utf8")

                # - max_new_tokens: Maximum number of new tokens to generate
                #   (optional, default is 256)
                max_new_tokens_tensor = pb_utils.get_input_tensor_by_name(request,
                                                                          "max_new_tokens")
                max_new_tokens: int = max_new_tokens_tensor.as_numpy()[0] \
                    if max_new_tokens_tensor else 256

                # Convert JSON string to Python object
                try:
                    # Decode messages and parse as JSON
                    messages = json.loads(messages_json_string)

                    # check message format, it should be a list of dicts
                    if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
                        raise ValueError(
                            f"Invalid messages format: {messages_json_string}. Expected a list of dictionaries.")
                    
                    # check for required keys in each message
                    for m in messages:
                        if "role" not in m or "content" not in m:
                            raise ValueError(
                                f"Invalid message format: {m}. Expected keys are 'role' and 'content'.")

                except json.JSONDecodeError as e:
                    raise RuntimeError(
                        f"Failed to decode message JSON: {messages_json_string}. Error: {e}")

                # Generate text using the pipeline
                outputs = self.pipe(
                    messages,
                    max_new_tokens=max_new_tokens,
                )
                generated_text  = outputs[0]["generated_text"][-1]

                # Prepare results
                context_output = pb_utils.Tensor("generated_text", 
                                                 np.array([generated_text], dtype=np.object_))

                # Create inference response
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[context_output]
                )
                responses.append(inference_response)

            except Exception as error:
                print(sys.exc_info()[2])
                responses.append(pb_utils.InferenceResponse(output_tensors=[],
                                                            error=pb_utils.TritonError(error)))

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """
        This function allows the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")
