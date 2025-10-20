"""
Model utilities for tools to ensure they receive raw LiteLLMModel instances.

This module provides utilities to extract raw LiteLLMModel from LoggingLiteLLMModel
wrappers, ensuring tools always work with the raw model interface.
"""


def get_raw_model(model):
    """
    Extract raw LiteLLMModel from LoggingLiteLLMModel wrapper if needed.
    
    LoggingLiteLLMModel is designed for agent LLM call logging and expects
    ChatMessage objects. Tools typically work with dict messages and should
    use the raw LiteLLMModel interface.
    
    Args:
        model: Either a raw LiteLLMModel or LoggingLiteLLMModel wrapper
        
    Returns:
        Raw LiteLLMModel instance
    """
    if model is None:
        return None
        
    # Check if this is a LoggingLiteLLMModel wrapper
    if hasattr(model, 'model') and hasattr(model, 'agent_context'):
        return model.model  # Return the wrapped LiteLLMModel
        
    return model  # Already raw model