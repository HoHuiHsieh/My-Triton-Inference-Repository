"""
"""
import io
import warnings
from base64 import b64decode
import traceback
import json
import numpy as np
import torch
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import triton_python_backend_utils as pb_utils
from opencc import OpenCC

# Suppress specific deprecation warnings that are beyond our control
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")


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
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
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
                audio_tensor = pb_utils.get_input_tensor_by_name(request,
                                                                 "input.audio")

                # Decode audio data
                audio_data: str = audio_tensor.as_numpy()[0]
                audio_data = b64decode(audio_data)

                # Convert audio data to a format suitable for the ASR model
                audio_data = io.BytesIO(audio_data)
                
                # Use the recommended loading method to avoid deprecation warnings
                try:
                    # Try the newer method first (if available in torchaudio >= 2.9)
                    if hasattr(torchaudio, 'load_with_torchcodec'):
                        audio_data, sample_rate = torchaudio.load_with_torchcodec(audio_data)
                    else:
                        # Use the current method with warning suppression
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            audio_data, sample_rate = torchaudio.load(audio_data)
                except Exception:
                    # Fallback to the old method if anything fails
                    audio_data, sample_rate = torchaudio.load(audio_data)
                
                # Convert to mono if stereo
                if audio_data.shape[0] > 1:
                    audio_data = torch.mean(audio_data, dim=0, keepdim=True)
                
                # Resample to 16kHz
                audio_data = torchaudio.functional.resample(audio_data,
                                                            orig_freq=sample_rate,
                                                            new_freq=16000)
                
                # Convert to numpy and flatten to 1D
                audio_data = audio_data.squeeze().detach().numpy()

                # Convert audio to text using the pipeline
                # Use input_features parameter to avoid deprecation warning
                asr_result = self.pipe(
                    {"array": audio_data, "sampling_rate": 16000},
                    generate_kwargs={
                        "language": "chinese",
                        "return_timestamps": True
                    }
                )
                asr_text = asr_result["text"]

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
                # print(traceback.format_exc())
                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(str(error))
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
