# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import json
import sys
import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.
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
        model_config = json.loads(args['model_config'])
        tokenizer_dir = model_config['parameters']['tokenizer_dir'][
            'string_value']

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir,
                                                       legacy=False,
                                                       padding_side='left',
                                                       trust_remote_code=True)

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse.
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
                input_tokens = pb_utils.get_input_tensor_by_name(
                    request, "tokens")
                input_tokens = input_tokens.as_numpy()

                # Decode the tokens
                output = self.tokenizer.decode(input_tokens)

                # Prepare results
                output_tensor = pb_utils.Tensor(
                    "output", np.array([output], dtype=np.object_))
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[output_tensor]
                )
                responses.append(inference_response)

            except Exception as error:
                print(sys.exc_info()[2])
                responses.append(pb_utils.InferenceResponse(output_tensors=[],
                                                            error=pb_utils.TritonError(error)))

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

        # tokens_batch = []
        # sequence_lengths = []
        # for idx, request in enumerate(requests):
        #     for input_tensor in request.inputs():
        #         if input_tensor.name() == "TOKENS_BATCH":
        #             tokens_batch.append(input_tensor.as_numpy())
        #         elif input_tensor.name() == "SEQUENCE_LENGTH":
        #             sequence_lengths.append(input_tensor.as_numpy())
        #         else:
        #             raise ValueError(f"unknown input {input_tensor.name}")

        # # batch decode
        # list_of_tokens = []
        # req_idx_offset = 0
        # req_idx_offsets = [req_idx_offset]
        # for idx, token_batch in enumerate(tokens_batch):
        #     for batch_idx, beam_tokens in enumerate(token_batch):
        #         for beam_idx, tokens in enumerate(beam_tokens):
        #             seq_len = sequence_lengths[idx][batch_idx][beam_idx]
        #             list_of_tokens.append(tokens[:seq_len])
        #             req_idx_offset += 1

        #     req_idx_offsets.append(req_idx_offset)

        # # all_outputs = self.tokenizer.batch_decode(
        # #     list_of_tokens,
        # #     skip_special_tokens=self.skip_special_tokens,
        # # )
        # all_outputs = []
        # for token in list_of_tokens:
        #     output = self.tokenizer.decode(
        #         token,
        #         skip_special_tokens=self.skip_special_tokens
        #     )
        #     # Check if output contains the Unicode replacement character (�)
        #     if '\ufffd' in output or b'\xef\xbf\xbd' in output.encode('utf8'):
        #         # Convert the token to its bytes representation instead of using a template
        #         token_bytes_repr = f"t'{token[0]}'"
        #         all_outputs.append(token_bytes_repr)
        #     else:
        #         all_outputs.append(output)

        # # construct responses
        # responses = []
        # for idx, request in enumerate(requests):
        #     req_outputs = []
        #     for x in all_outputs[req_idx_offsets[idx]:req_idx_offsets[idx + 1]]:
        #         req_outputs = [
        #             x.encode('utf8')
        #             for x in all_outputs[req_idx_offsets[idx]:req_idx_offsets[idx + 1]]
        #         ]

        #     output_tensor = pb_utils.Tensor(
        #         'OUTPUT',
        #         np.array(req_outputs).astype(self.output_dtype))

        #     outputs = [output_tensor]

        #     # Create InferenceResponse. You can set an error here in case
        #     # there was a problem with handling this inference request.
        #     # Below is an example of how you can set errors in inference
        #     # response:
        #     #
        #     # pb_utils.InferenceResponse(
        #     #    output_tensors=..., TritonError("An error occurred"))
        #     inference_response = pb_utils.InferenceResponse(
        #         output_tensors=outputs)
        #     responses.append(inference_response)
        # # You should return a list of pb_utils.InferenceResponse. Length
        # # of this list must match the length of `requests` list.
        # return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
