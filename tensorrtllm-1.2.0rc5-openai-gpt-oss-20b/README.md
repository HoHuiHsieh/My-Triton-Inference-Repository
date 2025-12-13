# OpenAI GPT OSS 20B (TensorRT-LLM)

## Model Name
`gpt-oss-20b`

## Model Source
[Hugging Face](https://huggingface.co/openai/gpt-oss-20b)

> **Note**: You need to download the model files from Hugging Face yourself. After downloading, update the model path in `serve.sh` and create the `gpt-oss-20b-throughput.yaml` configuration file as needed.

## How to Use

### Test the Model

Run the test script with function calling:

```bash
python test/openai_chat_client_function_calling.py \
    --model gpt-oss-20b \
    --prompt "What is the weather like in SF?"
```

Or use the test script:

```bash
bash test/test.sh
```

### Python Example - OpenAI Compatible API

```python
from openai import OpenAI

# Create OpenAI client pointing to the TensorRT-LLM server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="tensorrt_llm",
)

# Simple chat completion
messages = [
    {
        "role": "user",
        "content": "What is the weather like in San Francisco?",
    },
]

chat_completion = client.chat.completions.create(
    model="gpt-oss-20b",
    messages=messages,
    max_completion_tokens=500,
)

response = chat_completion.choices[0].message.content
print(response)
```

### Python Example - Function Calling

```python
import json
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="tensorrt_llm",
)

# Define function tool
tool_get_current_weather = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Gets the current weather in the provided location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "format": {
                    "type": "string",
                    "description": "default: celsius",
                    "enum": ["celsius", "fahrenheit"],
                },
            },
            "required": ["location"],
        }
    }
}

messages = [
    {
        "role": "user",
        "content": "What is the weather like in SF?",
    },
]

# Chat completion with function calling
chat_completion = client.chat.completions.create(
    model="gpt-oss-20b",
    messages=messages,
    max_completion_tokens=500,
    tools=[tool_get_current_weather],
)

message = chat_completion.choices[0].message
if message.tool_calls:
    tool_call = message.tool_calls[0]
    func_name = tool_call.function.name
    kwargs = json.loads(tool_call.function.arguments)
    
    print(f"Function: {func_name}")
    print(f"Arguments: {kwargs}")
    
    # Execute function and get result
    # result = your_function(**kwargs)
    
    # Send function result back to model
    messages.extend([{
        "role": "assistant",
        "tool_calls": [tool_call],
    }, {
        "role": "tool",
        "content": json.dumps(result),
        "tool_call_id": tool_call.id
    }])
    
    # Get final response
    final_completion = client.chat.completions.create(
        model="gpt-oss-20b",
        messages=messages,
        max_completion_tokens=500,
    )
    
    response = final_completion.choices[0].message.content
    print(response)
```

### Server Configuration

The model is served using TensorRT-LLM with the following default settings:
- **Base URL**: `http://localhost:8000/v1`
- **API Key**: `tensorrt_llm` (for authentication)
- **Port**: 8000
- **API**: OpenAI-compatible API

Update the GPU device in `serve.sh`:
```bash
--gpus '"device=2"'  # Change device ID as needed
```

### Inputs
- **Messages**: Array of chat messages with roles (user/assistant/system/tool)
- **Tools** (optional): Array of function definitions for function calling
- **Max Completion Tokens**: Maximum tokens in the response

### Outputs
- **Content**: Generated text response
- **Tool Calls** (if applicable): Function calls with arguments
- **Reasoning** (if available): Chain-of-thought reasoning

### Requirements
Install the required Python packages:
```bash
pip install -r test/requirement.txt
```

Main dependency:
```bash
pip install openai
```

### Performance Evaluation

Run performance benchmarking with different request rates:

```bash
bash test/perf.sh
```

This script will:
- Test multiple request rates (1, 5, 10, 15, 20 requests/second)
- Generate 100 requests per test with 128 input tokens and 128 output tokens
- Save results to `./request_rate_results/` directory
- Provide metrics including throughput, latency, and token generation statistics

#### Manual Performance Testing

You can also run genai-perf directly with custom parameters:

```bash
genai-perf profile \
    -m gpt-oss-20b \
    --tokenizer /workspace/model/gpt-oss-20b \
    --endpoint-type chat \
    --random-seed 123 \
    --synthetic-input-tokens-mean 128 \
    --synthetic-input-tokens-stddev 0 \
    --output-tokens-mean 128 \
    --output-tokens-stddev 0 \
    --request-rate 10 \
    --request-count 100 \
    --url localhost:8000 \
    --extra-inputs max_tokens:256 \
    --extra-inputs temperature:0.7
```

**Key Parameters:**
- `--request-rate`: Target requests per second (1, 5, 10, 15, 20)
- `--request-count`: Total number of requests to send (default: 100)
- `--synthetic-input-tokens-mean`: Average input length in tokens (default: 128)
- `--output-tokens-mean`: Average output length in tokens (default: 128)
- `--stability-percentage`: Percentage for stability detection (default: 999)

### Features
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI API
- **Function Calling**: Support for tool/function calling with reasoning
- **High Performance**: Optimized with TensorRT-LLM for fast inference
- **GPU Acceleration**: Leverages NVIDIA GPUs for efficient computation
- **Performance Benchmarking**: Built-in tools for throughput and latency testing
