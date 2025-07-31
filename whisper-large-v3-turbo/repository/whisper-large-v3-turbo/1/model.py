"""
"""
import traceback
import json
import numpy as np
import torch
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import triton_python_backend_utils as pb_utils
from opencc import OpenCC


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
        model_config = json.loads(args["model_config"])
        model_path = model_config["parameters"]["model_path"]["string_value"]

        # Check if model path is specified
        if not model_path:
            raise ValueError(
                "Model path is not specified in the configuration.")

        # Set device and dtype based on availability
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Prepare model
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        model.to(device)

        # Prepare processor
        processor = AutoProcessor.from_pretrained(model_path)

        # Create the ASR pipeline
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )

        # Prepare OpenCC for Simplified to Traditional Chinese conversion
        self.s2t_convert = OpenCC("s2t.json")

    def execute(self, requests):
        """
        """

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for idx, request in enumerate(requests):
            try:
                # Get input tensor by name
                text_tensor = pb_utils.get_input_tensor_by_name(request,
                                                                "input.audio")
                input_audio: str = text_tensor.as_numpy()[0]

                # Convert audio to text
                asr_text = self.pipe(input_audio)["text"]

                # Convert text from Simplified to Traditional Chinese
                output_text = self.s2t_convert.convert(asr_text)

                # Prepare output tensor
                output_text = np.array([output_text], dtype=np.object_)
                output_text = pb_utils.Tensor("output.text", output_text)

                # Create inference response
                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[output_text]
                ))

            except Exception as error:
                print(traceback.format_exc())
                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(error)
                ))

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")
