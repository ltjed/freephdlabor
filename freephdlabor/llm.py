import json
import os
import re
import base64
from typing import List, Union, Optional, Dict, Any

import anthropic
import backoff
import openai

MAX_NUM_TOKENS = 4096

AVAILABLE_LLMS = [
    "claude-3-5-sonnet-20240620",
    "claude-3-5-sonnet-20241022",
    "claude-3-7-sonnet-20250219",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o-mini",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "o1-preview-2024-09-12",
    "o1-mini-2024-09-12",
    "o1-2024-12-17",
    "o3-mini-2025-01-31",
    "o3-2025-01-31",
    "o4-mini-2025-04-16",
    "deepseek-coder",
    "deepseek-reasoner",
    "deepseek/deepseek-r1:nitro",
    "llama3.1-405b",
    # Anthropic Claude models via Amazon Bedrock
    "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
    "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
    "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
    "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
    "bedrock/anthropic.claude-3-opus-20240229-v1:0",
    # Anthropic Claude models Vertex AI
    "vertex_ai/claude-3-opus@20240229",
    "vertex_ai/claude-3-5-sonnet@20240620",
    "vertex_ai/claude-3-5-sonnet-v2@20241022",
    "vertex_ai/claude-3-sonnet@20240229",
    "vertex_ai/claude-3-haiku@20240307",
]


# Get N responses from a single message, used for ensembling.
@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APITimeoutError))
def get_batch_responses_from_llm(
        msg,
        client,
        model,
        system_message,
        print_debug=False,
        msg_history=None,
        temperature=0.75,
        n_responses=1,
):
    if msg_history is None:
        msg_history = []

    if model in [
        "gpt-4o-2024-05-13",
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-08-06",
    ]:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=n_responses,
            stop=None,
            seed=0,
        )
        content = [r.message.content for r in response.choices]
        new_msg_history = [
            new_msg_history + [{"role": "assistant", "content": c}] for c in content
        ]
    elif model == "deepseek-coder":
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model="deepseek-coder",
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=n_responses,
            stop=None,
        )
        content = [r.message.content for r in response.choices]
        new_msg_history = [
            new_msg_history + [{"role": "assistant", "content": c}] for c in content
        ]
    elif model == "deepseek-reasoner":
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=n_responses,
            stop=None,
        )
        reasoning_content = [r.message.reasoning_content for r in response.choices]
        content = [r.message.content for r in response.choices]
        new_msg_history = [
            new_msg_history + [{"role": "assistant", "content": c}] for c in content
        ]
    elif model == "llama-3-1-405b-instruct":
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model="meta-llama/llama-3.1-405b-instruct",
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=n_responses,
            stop=None,
        )
        content = [r.message.content for r in response.choices]
        new_msg_history = [
            new_msg_history + [{"role": "assistant", "content": c}] for c in content
        ]
    else:
        content, new_msg_history = [], []
        for _ in range(n_responses):
            c, hist = get_response_from_llm(
                msg,
                client,
                model,
                system_message,
                print_debug=False,
                msg_history=None,
                # temperature=temperature,
            )
            content.append(c)
            new_msg_history.append(hist)

    if print_debug:
        # Just print the first one.
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_history[0]):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(content)
        print("*" * 21 + " LLM END " + "*" * 21)
        print()

    return content, new_msg_history


@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APITimeoutError))
def get_response_from_llm(
        msg,
        client,
        model,
        system_message,
        print_debug=False,
        msg_history=None,
        temperature=0.75,
):
    if msg_history is None:
        msg_history = []

    if "claude" in model:
        new_msg_history = msg_history + [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": msg,
                    }
                ],
            }
        ]
        response = client.messages.create(
            model=model,
            max_tokens=MAX_NUM_TOKENS,
            temperature=temperature,
            system=system_message,
            messages=new_msg_history,
        )
        content = response.content[0].text
        new_msg_history = new_msg_history + [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": content,
                    }
                ],
            }
        ]
    elif model in [
        "gpt-4o-2024-05-13",
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-08-06",
    ]:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=1,
            stop=None,
            seed=0,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif model in ["o1-preview-2024-09-12", "o1-mini-2024-09-12", "o1-2024-12-17", "o3-mini-2025-01-31",]:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        print("temp?")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": system_message},
                *new_msg_history,
            ],
            temperature=1,
            max_completion_tokens=MAX_NUM_TOKENS,
            n=1,
            #stop=None,
            seed=0,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif model == "deepseek-coder":
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model="deepseek-coder",
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=1,
            stop=None,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif model in ["deepseek-reasoner", "deepseek/deepseek-r1:nitro"]:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=1,
            stop=None,
        )
        # reasoning_content = response.choices[0].message.reasoning_content
        # print(f"@@@\n reasoning_content is {reasoning_content}")
        content = response.choices[0].message.content
        print(f"@@@\n content is {content}")
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif model in ["meta-llama/llama-3.1-405b-instruct", "llama-3-1-405b-instruct"]:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model="meta-llama/llama-3.1-405b-instruct",
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=1,
            stop=None,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    # elif model == "deepseek/deepseek-reasoner":
    #     new_msg_history = msg_history + [{"role": "user", "content": msg}]
    #     response = client.chat.completions.create(
    #         model=model,
    #         messages=[
    #             {"role": "system", "content": system_message},
    #             *new_msg_history,
    #         ],
    #         temperature=temperature,
    #         max_tokens=MAX_NUM_TOKENS,
    #         n=1,
    #         stop=None,
    #     )
    #     content = response.choices[0].message.content
    #     new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    else:
        raise ValueError(f"Model {model} not supported.")

    if print_debug:
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_history):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(content)
        print("*" * 21 + " LLM END " + "*" * 21)
        print()

    return content, new_msg_history


def extract_json_between_markers(llm_output):
    # Regular expression pattern to find JSON content between ```json and ```
    for _ in range(10):
        print("!!!\n")
    print(llm_output)
    for _ in range(10):
        print("@@@\n")
    json_pattern = r"```json(.*?)```"
    matches = re.findall(json_pattern, llm_output, re.DOTALL)

    if not matches:
        # Fallback: Try to find any JSON-like content in the output
        json_pattern = r"\{.*?\}"
        matches = re.findall(json_pattern, llm_output, re.DOTALL)

    for json_string in matches:
        json_string = json_string.strip()
        try:
            parsed_json = json.loads(json_string)
            return parsed_json
        except json.JSONDecodeError:
            # Attempt to fix common JSON issues
            try:
                # Remove invalid control characters
                json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
                parsed_json = json.loads(json_string_clean)
                return parsed_json
            except json.JSONDecodeError:
                continue  # Try next match
    return None  # No valid JSON found

def extract_json_between_markers(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from text between ```json and ``` markers.
    
    Args:
        text: Text containing JSON
        
    Returns:
        Extracted JSON as a dictionary or None if extraction fails
    """
    # Try to find JSON between ```json and ``` markers
    matches = re.findall(r"```json\s*([\s\S]*?)\s*```", text)
    if matches:
        try:
            return json.loads(matches[0])
        except json.JSONDecodeError:
            return None
    
    # Try to find JSON between ``` and ``` markers without json specifier
    matches = re.findall(r"```\s*([\s\S]*?)\s*```", text)
    if matches:
        try:
            return json.loads(matches[0])
        except json.JSONDecodeError:
            return None
    
    return None

def encode_image_to_base64(image_data: Union[str, bytes, List[bytes]]) -> str:
    """
    Encode image data to base64 string for VLM usage.
    
    Args:
        image_data: Can be file path (str), raw bytes, or list of bytes
        
    Returns:
        Base64 encoded string
    """
    if isinstance(image_data, str):
        # File path
        with open(image_data, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    elif isinstance(image_data, list):
        # List of bytes (take first element)
        return base64.b64encode(image_data[0]).decode("utf-8")
    elif isinstance(image_data, bytes):
        # Raw bytes
        return base64.b64encode(image_data).decode("utf-8")
    else:
        raise TypeError(f"Unsupported image data type: {type(image_data)}")


@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APITimeoutError))
def get_response_from_vlm(
    prompt: str,
    images: List[str],
    client,
    model: str,
    system_message: str = "",
    print_debug: bool = False,
    msg_history: Optional[List[Dict]] = None,
    temperature: float = 0.75,
) -> tuple[str, List[Dict]]:
    """
    Get response from Vision-Language Model with image inputs.
    
    Args:
        prompt: Text prompt for the VLM
        images: List of image file paths
        client: OpenAI client instance
        model: Model name (should be vision-capable like gpt-4o)
        system_message: System message for the conversation
        print_debug: Whether to print debug information
        msg_history: Previous conversation history
        temperature: Sampling temperature
        
    Returns:
        Tuple of (response_content, updated_message_history)
    """
    if msg_history is None:
        msg_history = []
    
    # Prepare message content with text and images
    content = [{"type": "text", "text": prompt}]
    
    # Add images to content
    for image_path in images:
        try:
            base64_image = encode_image_to_base64(image_path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
        except Exception as e:
            print(f"Warning: Failed to encode image {image_path}: {e}")
            continue
    
    # Build message history
    new_msg_history = msg_history + [{"role": "user", "content": content}]
    
    # Prepare messages for API call
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.extend(new_msg_history)
    
    # Make API call (currently only supports OpenAI-compatible VLMs)
    if "gpt-4o" in model or "gpt-4-vision" in model:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=1,
            seed=0,
        )
        content_response = response.choices[0].message.content
        
        # Update message history
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content_response}]
        
    else:
        raise ValueError(f"VLM model {model} not supported. Currently only supports GPT-4 Vision models.")
    
    if print_debug:
        print()
        print("*" * 20 + " VLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_history):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(content_response)
        print("*" * 21 + " VLM END " + "*" * 21)
        print()
    
    return content_response, new_msg_history


def create_vlm_client(model: str = "gpt-4o-2024-05-13"):
    """
    Create a VLM client for vision tasks.
    
    Args:
        model: VLM model name (defaults to GPT-4o)
        
    Returns:
        Tuple of (client, model_name)
    """
    if "gpt-4o" in model or "gpt-4-vision" in model:
        print(f"Using OpenAI VLM API with model {model}.")
        return openai.OpenAI(), model
    else:
        # Default to GPT-4o if unsupported model
        print(f"Model {model} not supported for VLM. Defaulting to gpt-4o-2024-05-13.")
        return openai.OpenAI(), "gpt-4o-2024-05-13"


def create_client(model):
    if model.startswith("claude-"):
        print(f"Using Anthropic API with model {model}.")
        return anthropic.Anthropic(), model
    elif model.startswith("bedrock") and "claude" in model:
        client_model = model.split("/")[-1]
        print(f"Using Amazon Bedrock with model {client_model}.")
        return anthropic.AnthropicBedrock(), client_model
    elif model.startswith("vertex_ai") and "claude" in model:
        client_model = model.split("/")[-1]
        print(f"Using Vertex AI with model {client_model}.")
        return anthropic.AnthropicVertex(), client_model
    elif 'gpt' in model:
        print(f"Using OpenAI API with model {model}.")
        return openai.OpenAI(), model
    elif model in ["o1-preview-2024-09-12", "o1-mini-2024-09-12", "o1-2024-12-17", "o3-mini-2025-01-31",]:
        print(f"Using OpenAI API with model {model}.")
        return openai.OpenAI(), model
    elif model in ["deepseek-coder", "deepseek-reasoner"]:
        print(f"Using OpenAI API with {model}.")
        return openai.OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com"
        ), model
    elif model == "llama3.1-405b":
        print(f"Using OpenAI API with {model}.")
        return openai.OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1"
        ), "meta-llama/llama-3.1-405b-instruct"
    elif model == "deepseek/deepseek-r1:nitro":
        print("Using OpenRouter API with DeepSeek Reasoner.")
        return openai.OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1"
        ), "deepseek/deepseek-r1:nitro"
    else:
        raise ValueError(f"Model {model} not supported.")
