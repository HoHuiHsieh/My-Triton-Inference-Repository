import sys
import os
import json
import numpy as np
import glob
import triton_python_backend_utils as pb_utils
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


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
        self.model = AutoModel.from_pretrained(model_path,
                                               trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_length = 32768
        self.task_name_to_instruct = {
            "default": "Given a question, retrieve passages that answer the question",
        }

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
                input_text = pb_utils.get_input_tensor_by_name(
                    request, "input_text")
                input_text = [p.decode("utf8") for p in input_text.as_numpy()[0]]

                input_type = pb_utils.get_input_tensor_by_name(
                    request, "input_type")
                if input_type is not None:
                    input_type = np.squeeze(input_type.as_numpy(), axis=0)
                    input_type = str(input_type[0].decode("utf8"))
                else:
                    # Default to appropriate input type based on input size
                    if len(input_text) == 1:
                        input_type = "query"
                    else:
                        input_type = "passage"

                # Switch input_type for different prefix
                if input_type == "query":
                    prefix = "Instruct: " \
                        + self.task_name_to_instruct["default"] \
                        + "\nQuery: "
                elif input_type == "passage":
                    prefix = ""
                else:
                    raise ValueError("Invalid input type!!")

                # count the number of tokens
                tokens = self.tokenizer(
                    input_text,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                prompt_tokens = torch.sum(tokens["attention_mask"])

                # get the embeddings
                embeddings = self.model.encode(input_text,
                                               instruction=prefix,
                                               max_length=self.max_length)
                
                # normalize embeddings
                embeddings = F.normalize(embeddings, p=2, dim=1)
                embeddings = embeddings.detach().numpy()
            
                # Prepare results
                context_output = pb_utils.Tensor("embeddings", np.array([embeddings], dtype=np.float32))
                prompt_tokens_output = pb_utils.Tensor("prompt_tokens", np.array([prompt_tokens], dtype=np.int32))
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[context_output, prompt_tokens_output]
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
