import sys
import os
import json
import numpy as np
import triton_python_backend_utils as pb_utils
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


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
        device = model_config["parameters"]["device"]["string_value"]
       
        torch_dtype = torch.float16 if "cuda" in device else torch.float32

        # Load model with tokenizer on specified device
        print(f"Initializing Whisper model on device: {device}")
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path, 
            torch_dtype=torch_dtype, 
            low_cpu_mem_usage=True, 
            use_safetensors=True
        )
        model.to(device)
        processor = AutoProcessor.from_pretrained(model_path)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
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
                # Get audio input as bytes (raw audio data)
                input_audio_tensor = pb_utils.get_input_tensor_by_name(request, "input_audio")
                if input_audio_tensor is None:
                    raise ValueError("input_audio is required")
                
                input_audio_array = input_audio_tensor.as_numpy()
                # input_audio_array is a bytes array containing raw audio data
                # The pipeline can accept raw bytes directly
                input_audio_bytes = input_audio_array[0] if input_audio_array.ndim > 0 else input_audio_array

                # Run speech recognition pipeline with long-form transcription support
                # The pipeline can accept bytes, file path, or numpy array
                # For audio longer than 30 seconds, we need to enable chunking
                result = self.pipe(
                    input_audio_bytes,
                    chunk_length_s=30,  # Process in 30-second chunks
                    batch_size=8,        # Batch size for processing chunks
                    return_timestamps=False  # Set to True if you want timestamps
                )
                
                # Extract text from result (result is a dict with "text" key)
                output_text = result["text"] if isinstance(result, dict) else str(result)
                
                # Prepare results - output is TYPE_STRING with dims [1]
                output_text_array = np.array([output_text.encode('utf-8')], dtype=object)
                output_text_tensor = pb_utils.Tensor(
                    "output_text", output_text_array
                )
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[output_text_tensor]
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