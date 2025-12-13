import sys
import os
import json
import numpy as np
import glob
import triton_python_backend_utils as pb_utils
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Average pooling with attention mask."""

    last_hidden_states = last_hidden_states.to(torch.float32)
    last_hidden_states_masked = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    embedding = last_hidden_states_masked.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    embedding = F.normalize(embedding, dim=-1)
    
    return embedding

# Define task and queries
def get_instruction(task_instruction: str, query: str) -> str:
    return f"Instruct: {task_instruction}\nQuery: {query}"



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
        print(f"Initializing EmbeddingGemma model on device: {model_device}")
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left",
        )

        # Load model
        attn_implementation = "flash_attention_2" if torch.cuda.is_available() else "eager"
        model = AutoModel.from_pretrained(
            model_path, 
            trust_remote_code=True,
            torch_dtype=torch.float16,
            attn_implementation=attn_implementation,
        ).eval()
        self.model = model.to("cuda:0" if torch.cuda.is_available() else "cpu")
        self.task = "Given a question, retrieve passages that answer the question"

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
                # Get query as a string (dims: [1])
                # With max_batch_size: 0, no batch dimension is added by Triton
                query = pb_utils.get_input_tensor_by_name(request, "query")
                if query is not None:
                    query_array = query.as_numpy()
                    # Extract the string from shape (1,)
                    query = query_array[0].decode("utf8") if isinstance(query_array[0], bytes) else str(query_array[0])
                    # Model is instruction-aware, which requires each query to have a short instruction with the task instruction
                    input_texts = [ get_instruction(self.task, query) ]
 
                 # Get documents as a list of strings (dims: [1, -1])
                documents = pb_utils.get_input_tensor_by_name(request, "documents")
                if documents is not None:
                    documents_array = documents.as_numpy()
                    # For dims [1, -1]: if shape is (1, n), extract from first dimension
                    # For dims [-1]: if shape is (n,), use directly
                    if documents_array.ndim == 2:
                        # Shape: (1, n) - extract strings from first row
                        input_texts = [doc.decode("utf8") if isinstance(doc, bytes) else str(doc) 
                                   for doc in documents_array[0]]
                    else:
                        # Shape: (n,) - use directly
                        input_texts = [doc.decode("utf8") if isinstance(doc, bytes) else str(doc) 
                                   for doc in documents_array]
                
                # Tokenize the input texts
                batch_dict = self.tokenizer(
                    text=input_texts,
                    max_length=4096,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.model.device)
                
                attention_mask = batch_dict["attention_mask"]

                # Forward pass
                model_outputs = self.model(**batch_dict)

                # Average pooling
                embeddings = average_pool(model_outputs.last_hidden_state, attention_mask)

                # Convert to numpy if it's a torch tensor and ensure it's on CPU
                if isinstance(embeddings, torch.Tensor):
                    embeddings = embeddings.detach().cpu().numpy()
                
                # Ensure embeddings is 2D with shape (1, embedding_dim) for output dims: [1, -1]
                if embeddings.ndim == 1:
                    embeddings = embeddings.reshape(1, -1)
                
                # Prepare results - output shape should be (1, embedding_dim)
                context_output = pb_utils.Tensor(
                    "embeddings", embeddings.astype(np.float32)
                )
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