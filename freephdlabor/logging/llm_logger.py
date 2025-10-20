"""
LLM Logging Infrastructure for Multi-Agent System

This module provides a logging wrapper for LiteLLMModel that captures complete
agent LLM call context for debugging and prompt improvement analysis.
"""

import json
import uuid
import os
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from smolagents.models import ChatMessage


class LoggingLiteLLMModel:
    """
    A wrapper around LiteLLMModel that logs all agent LLM calls with complete context.
    
    This wrapper intercepts generate() calls to capture:
    - Complete input messages (including system prompts and agent instructions)
    - Agent context (which agent made the call)
    - Response content and metadata
    - Token usage and timing information
    
    The logs are designed to be easily analyzable by AI assistants for prompt
    engineering improvements and multi-agent coordination debugging.
    """
    
    def __init__(self, base_model, agent_context: Dict[str, str], log_file_path: str):
        """
        Initialize the logging wrapper.
        
        Args:
            base_model: The LiteLLMModel instance to wrap
            agent_context: Dict with 'agent_type' and 'agent_name' keys
            log_file_path: Path to the JSONL log file
        """
        self.model = base_model
        self.agent_context = agent_context
        self.log_file_path = log_file_path
        
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    
    def generate(self, messages: List[ChatMessage], **kwargs) -> ChatMessage:
        """
        Wrap the base model's generate() method with logging.
        
        Args:
            messages: List of ChatMessage objects
            **kwargs: Additional parameters passed to the model
            
        Returns:
            ChatMessage response from the base model
        """
        call_id = str(uuid.uuid4())
        start_time = time.time()
        timestamp = datetime.now().isoformat()
        
        # Create log entry with input data
        log_entry = {
            "call_id": call_id,
            "timestamp": timestamp,
            "agent_type": self.agent_context.get("agent_type", "unknown"),
            "agent_name": self.agent_context.get("agent_name", "unknown"),
            "workspace_run": self._get_workspace_run_id(),
            "input": {
                "messages": self._serialize_messages(messages),
                "parameters": kwargs
            }
        }
        
        # Quota-aware retry logic for rate limiting
        max_retries = 5  # Balanced retry coverage without excessive delays
        base_delay = 9  # seconds, as suggested by Gemini API
        response = None
        
        try:
            for attempt in range(max_retries + 1):
                try:
                    # Call the actual model
                    response = self.model.generate(messages, **kwargs)
                    
                    # Calculate timing
                    end_time = time.time()
                    duration_ms = int((end_time - start_time) * 1000)
                    
                    # Add response data to log entry
                    log_entry["output"] = {
                        "content": response.content if response.content else "",
                        "token_usage": self._extract_token_usage(response),
                        "duration_ms": duration_ms
                    }
                    log_entry["status"] = "success"
                    if attempt > 0:
                        log_entry["retry_attempt"] = attempt
                    
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    # Check if this is a quota/rate limit error (429)
                    is_quota_error = (
                        "429" in str(e) or 
                        "RateLimitError" in str(e) or
                        "RESOURCE_EXHAUSTED" in str(e) or
                        "quota" in str(e).lower()
                    )
                    
                    # Calculate timing for this attempt
                    end_time = time.time()
                    duration_ms = int((end_time - start_time) * 1000)
                    
                    if is_quota_error and attempt < max_retries:
                        # Calculate exponential backoff delay
                        delay = base_delay * (2 ** attempt)
                        
                        # Log the retry attempt
                        retry_entry = log_entry.copy()
                        retry_entry["output"] = {
                            "error": f"Quota limit hit, retrying in {delay}s (attempt {attempt + 1}/{max_retries + 1}): {str(e)}",
                            "duration_ms": duration_ms
                        }
                        retry_entry["status"] = "quota_retry"
                        retry_entry["retry_attempt"] = attempt
                        retry_entry["retry_delay_seconds"] = delay
                        self._write_log_entry(retry_entry)
                        
                        # Wait before retrying
                        time.sleep(delay)
                        continue
                    else:
                        # Final failure or non-quota error
                        log_entry["output"] = {
                            "error": str(e),
                            "duration_ms": duration_ms
                        }
                        log_entry["status"] = "error"
                        if attempt > 0:
                            log_entry["final_retry_attempt"] = attempt
                        
                        # Re-raise the exception
                        raise
        
        finally:
            # Write log entry (if not already written during retry)
            if log_entry.get("status") != "quota_retry":
                self._write_log_entry(log_entry)
        
        return response
    
    def _serialize_messages(self, messages) -> List[Dict[str, Any]]:
        """
        Convert ChatMessage objects or dict messages to serializable dictionaries.
        
        Args:
            messages: List of ChatMessage objects or dict objects with 'role'/'content' keys
            
        Returns:
            List of dictionaries representing the messages
        """
        serialized = []
        for msg in messages:
            # Handle both ChatMessage objects and dict format
            if hasattr(msg, 'role') and hasattr(msg, 'content'):
                # ChatMessage object
                msg_dict = {
                    "role": msg.role,
                    "content": msg.content
                }
                # Include tool calls if present
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    msg_dict["tool_calls"] = [
                        {
                            "id": tc.id if hasattr(tc, 'id') else None,
                            "type": tc.type if hasattr(tc, 'type') else None,
                            "function": {
                                "name": tc.function.name if hasattr(tc, 'function') and hasattr(tc.function, 'name') else None,
                                "arguments": tc.function.arguments if hasattr(tc, 'function') and hasattr(tc.function, 'arguments') else None
                            } if hasattr(tc, 'function') else None
                        }
                        for tc in msg.tool_calls
                    ]
            elif isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                # Dict format from tools
                msg_dict = {
                    "role": msg["role"],
                    "content": msg["content"]
                }
                # Include tool calls if present in dict
                if 'tool_calls' in msg and msg['tool_calls']:
                    msg_dict["tool_calls"] = msg['tool_calls']
            else:
                # Fallback for unknown format
                msg_dict = {
                    "role": "unknown",
                    "content": str(msg)
                }
            
            serialized.append(msg_dict)
        
        return serialized
    
    def _extract_token_usage(self, response: ChatMessage) -> Optional[Dict[str, int]]:
        """
        Extract token usage information from the response.
        
        Args:
            response: ChatMessage response from the model
            
        Returns:
            Dictionary with token usage info or None
        """
        if hasattr(response, 'token_usage') and response.token_usage:
            return {
                "prompt_tokens": getattr(response.token_usage, 'prompt_tokens', 0),
                "completion_tokens": getattr(response.token_usage, 'completion_tokens', 0),
                "total_tokens": getattr(response.token_usage, 'total_tokens', 0)
            }
        return None
    
    def _get_workspace_run_id(self) -> str:
        """
        Extract the workspace run ID from the log file path.
        
        Returns:
            Workspace run identifier (e.g., "20250715_103000_adaptive_lr_cnn")
        """
        try:
            # Extract from path like: .../results/20250715_103000_adaptive_lr_cnn/agent_llm_calls.jsonl
            path_parts = os.path.normpath(self.log_file_path).split(os.sep)
            for part in path_parts:
                if part.startswith("202") and "_" in part:  # Look for timestamp pattern
                    return part
        except:
            pass
        return "unknown_run"
    
    def _write_log_entry(self, log_entry: Dict[str, Any]) -> None:
        """
        Write a log entry to the JSONL file.
        
        Args:
            log_entry: Dictionary containing the log data
        """
        try:
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            # Don't let logging failures break the system
            print(f"Warning: Failed to write LLM log entry: {e}")
    
    # Forward other attributes to the base model
    def __getattr__(self, name):
        """Forward attribute access to the base model."""
        return getattr(self.model, name)


def create_agent_logging_model(base_model, agent_type: str, agent_name: str, workspace_dir: str):
    """
    Convenience function to create a logging model wrapper for an agent.
    
    Args:
        base_model: The LiteLLMModel instance to wrap
        agent_type: Type of agent (e.g., "ManagerAgent", "IdeationAgent")
        agent_name: Name of the agent instance (e.g., "manager_agent", "ideation_agent")
        workspace_dir: Workspace directory path
        
    Returns:
        LoggingLiteLLMModel instance configured for the agent
    """
    agent_context = {
        "agent_type": agent_type,
        "agent_name": agent_name
    }
    
    log_file_path = os.path.abspath(os.path.join(workspace_dir, "agent_llm_calls.jsonl"))
    
    return LoggingLiteLLMModel(
        base_model=base_model,
        agent_context=agent_context,
        log_file_path=log_file_path
    )