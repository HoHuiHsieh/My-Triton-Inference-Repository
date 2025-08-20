import argparse
import json
import re

from openai import OpenAI


system_prompt = """You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: 2025-06-28

Reasoning: low

# Valid channels: analysis, commentary, final. Channel must be included for every message.
Calls to these tools must go to commentary channel, for example:
    commentary to=get_location
    commentary to=get_current_weather
    commentary to=get_multiple_weathers
"""


developer_prompt = """# Instructions

Use a friendly tone.

# Tools

// Gets the location of the user.
get_location = () => any;

// Gets the current weather in the provided location.
get_current_weather = (_: {
    location: string, // The city and state, e.g. San Francisco, CA
    unit: "celsius" | "fahrenheit", // default: celsius
}) => any;

// Gets the current weather in multiple locations.
get_multiple_weathers = (_: {
    locations: string[], // List of city and state, e.g. ["San Francisco, CA", "New York, NY"]
    unit: "celsius" | "fahrenheit", // default: celsius
}) => any;

"""


schema_get_current_weather = {
    "type": "object",
    "properties": {
        "location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA",
        },
        "unit": {
            "type": "string",
            "description": "default: celsius",
            "enum": ["celsius", "fahrenheit"],
        },
    },
    "required": ["location"],
}

schema_get_multiple_weathers = {
    "type": "object",
    "properties": {
        "locations": {
            "type":
            "array",
            "items": {
                "type": "string"
            },
            "description":
            'List of city and state, e.g. ["San Francisco, CA", "New York, NY"]',
        },
        "unit": {
            "type": "string",
            "description": "default: celsius",
            "enum": ["celsius", "fahrenheit"],
        },
    },
    "required": ["locations"],
}


def get_current_weather(location: str, unit: str = "celsius") -> dict:
    print(f"\n\n*** Fetching current weather for {location} in {unit}... ***\n\n")
    return {"location": location ,"sunny": True, "temperature": 20 if unit == "celsius" else 68}


def get_multiple_weathers(locations: list[str],
                          unit: str = "celsius") -> list[dict]:
    return [get_current_weather(location, unit) for location in locations]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--prompt",
                        type=str,
                        default="What is the weather like in SF?")
    args = parser.parse_args()

    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="tensorrt_llm",
    )

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "developer",
            "content": developer_prompt,
        },
        {
            "role": "user",
            "content": args.prompt,
        },
    ]

    response_format = {
        "type":
            "structural_tag",
            "structures": [{
                "begin":
                "<|channel|>commentary to=get_current_weather <|constrain|>json<|message|>",
                "schema": schema_get_current_weather,
                "end": "<|call|>",
            }, {
                "begin":
                "<|channel|>commentary to=get_multiple_weathers <|constrain|>json<|message|>",
                "schema": schema_get_multiple_weathers,
                "end": "<|call|>",
            }],
        "triggers": ["<|channel|>commentary to="],
    }

    print(f"[USER PROMPT] {args.prompt}")

    # Stream the first response
    print("[RESPONSE 1] ", end="", flush=True)
    response_text = ""
    is_streaming = True
    response = client.chat.completions.create(
        model=args.model,
        messages=messages,
        max_completion_tokens=500,
        stream=is_streaming,
        stop=["<|call|>"],
        response_format=response_format,
        extra_body={
            "skip_special_tokens": False,
            "include_stop_str_in_output": True,
        },
    )
    if is_streaming:
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                response_text += content
    else:
        response_text = response.choices[0].message.content
        print(response_text)

    for regex, tool in [
        (r"(<\|channel\|>commentary to=get_current_weather <\|constrain\|>json<\|message\|>)([\S\s]+)(<\|call\|>)",
         get_current_weather),
        (r"(<\|channel\|>commentary to=get_multiple_weathers <\|constrain\|>json<\|message\|>)([\S\s]+)(<\|call\|>)",
         get_multiple_weathers)
    ]:
        match = re.search(regex, response_text)
        if match is not None:
            break
    else:
        print("Failed to call functions, exiting...")
        return

    kwargs = json.loads(match.group(2))
    print(f"\n[FUNCTION CALL] {tool.__name__}(**{kwargs})")
    answer = tool(**kwargs)

    print("\n----------------------------------------------------\n")

    messages.extend([{
        "role": "assistant",
        "content": match.group(0),
    }, {
        "role": f"{tool.__name__} to=assistant",
        "content": json.dumps(answer, ensure_ascii=False),
    }])
    print(json.dumps(messages, indent=2, ensure_ascii=False))

    # Stream the second response
    print("[RESPONSE 2] ", end="", flush=True)

    response = client.chat.completions.create(
        model=args.model,
        messages=messages,
        max_completion_tokens=500,
        stream=is_streaming ,
        stop=["<|call|>"],
        extra_body={
            "skip_special_tokens": False,
            "include_stop_str_in_output": True,
            # "response_format": response_format,
        },
    )

    response_text = ""
    if is_streaming:
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                response_text += content
    else:
        response_text = response.choices[0].message.content
        print(response_text)

    print()  # Add newline after streaming
    print(f"[FINAL RESPONSE] {response_text}")


if __name__ == "__main__":
    main()
