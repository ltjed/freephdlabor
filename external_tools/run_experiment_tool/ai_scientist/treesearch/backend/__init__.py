import os
from . import backend_anthropic, backend_openai
from .utils import FunctionSpec, OutputType, PromptType, compile_prompt_to_md


def query(
    system_message: PromptType | None,
    user_message: PromptType | None,
    model: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> OutputType:
    """
    General LLM query for various backends with a single system and user message.
    Supports function calling for some backends.

    Args:
        system_message (PromptType | None): Uncompiled system message (will generate a message following the OpenAI/Anthropic format)
        user_message (PromptType | None): Uncompiled user message (will generate a message following the OpenAI/Anthropic format)
        model (str): string identifier for the model to use (e.g. "gpt-4-turbo")
        temperature (float | None, optional): Temperature to sample at. Defaults to the model-specific default.
        max_tokens (int | None, optional): Maximum number of tokens to generate. Defaults to the model-specific max tokens.
        func_spec (FunctionSpec | None, optional): Optional FunctionSpec object defining a function call. If given, the return value will be a dict.

    Returns:
        OutputType: A string completion if func_spec is None, otherwise a dict with the function call details.
    """

    model_kwargs = model_kwargs | {
        "model": model,
        "temperature": temperature,
    }

    # Handle models with different API requirements
    if model.startswith("o1"):
        # o1 models: Don't support system messages (beta limitations)
        # ref: https://platform.openai.com/docs/guides/reasoning/beta-limitations
        if system_message and user_message is None:
            user_message = system_message
        elif system_message is None and user_message:
            pass
        elif system_message and user_message:
            system_message["Main Instructions"] = {}
            system_message["Main Instructions"] |= user_message
            user_message = system_message
        system_message = None
        model_kwargs["reasoning_effort"] = "high"
        model_kwargs.pop("temperature", None)  # o1 doesn't support temperature
    elif model.startswith("gpt-5"):
        # GPT-5 models: Support system messages but not temperature
        # Just set reasoning_effort and remove unsupported params
        # Check environment for reasoning_effort setting (from .llm_config.yaml)
        reasoning_effort = os.environ.get('RUN_EXPERIMENT_REASONING_EFFORT', 'high')
        model_kwargs["reasoning_effort"] = reasoning_effort
        model_kwargs.pop("temperature", None)  # GPT-5 doesn't support temperature
        # Note: max_completion_tokens is optional, defaults handled by API
    else:
        model_kwargs["max_tokens"] = max_tokens

    query_func = backend_anthropic.query if "claude-" in model else backend_openai.query
    output, req_time, in_tok_count, out_tok_count, info = query_func(
        system_message=compile_prompt_to_md(system_message) if system_message else None,
        user_message=compile_prompt_to_md(user_message) if user_message else None,
        func_spec=func_spec,
        **model_kwargs,
    )

    return output
