"""
Workspace-aware Python executor for agent code execution.

This module provides a custom Python executor that runs code in a specified
workspace directory while maintaining full compatibility with smolagents.
"""

import os
import logging
from typing import Any, Optional, Dict, List, Tuple
from smolagents import LocalPythonExecutor, Tool


class WorkspacePythonExecutor(LocalPythonExecutor):
    """
    A Python executor that runs code in a workspace directory context.
    
    This executor wraps LocalPythonExecutor to ensure all agent-generated code
    runs in the workspace directory, while maintaining complete compatibility
    with the base executor's behavior.
    
    Args:
        workspace_dir: Directory where code should be executed
        additional_authorized_imports: List of additional imports to allow
        max_print_outputs_length: Maximum length of print outputs
        additional_functions: Additional Python functions to make available
    """
    
    def __init__(
        self,
        workspace_dir: str,
        additional_authorized_imports: Optional[List[str]] = None,
        max_print_outputs_length: Optional[int] = None,
        additional_functions: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the workspace executor with a specific directory."""
        # Store workspace directory as absolute path
        self.workspace_dir = os.path.abspath(workspace_dir)
        
        # Ensure workspace directory exists
        os.makedirs(self.workspace_dir, exist_ok=True)
        
        # Initialize parent with all the same parameters
        # Ensure additional_authorized_imports is always a list
        if additional_authorized_imports is None:
            additional_authorized_imports = []
            
        super().__init__(
            additional_authorized_imports=additional_authorized_imports,
            max_print_outputs_length=max_print_outputs_length,
            additional_functions=additional_functions,
        )
        
        # Log the setup for debugging
        logging.debug(f"WorkspacePythonExecutor initialized with workspace: {self.workspace_dir}")
        
    def __call__(self, code_action: str) -> Tuple[Any, str, bool]:
        """
        Execute Python code in the workspace directory context.
        
        Args:
            code_action: Python code to execute
            
        Returns:
            Tuple of (result, output, is_final_answer)
            - result: The execution result
            - output: Captured print outputs
            - is_final_answer: Whether this is a final answer
        """
        # Save current directory
        original_dir = os.getcwd()
        
        try:
            # Change to workspace directory
            os.chdir(self.workspace_dir)
            
            # Execute code using parent's implementation
            result = super().__call__(code_action)
            
            return result
            
        except Exception as e:
            # Enhanced error reporting: ensure critical import errors are visible to agent
            error_msg = str(e)
            if "Forbidden access to module" in error_msg or "InterpreterError" in str(type(e)):
                # For import/security errors, make sure they're reported clearly
                enhanced_error = f"CRITICAL EXECUTION ERROR: {error_msg}\nCode that failed: {code_action[:200]}..."
                logging.error(enhanced_error)
                print(f"🚨 {enhanced_error}")  # Also print to stdout for agent visibility
            
            # Re-raise the exception to maintain normal error handling flow
            raise
            
        finally:
            # Always restore original directory
            os.chdir(original_dir)
            
    def send_variables(self, variables: Dict[str, Any]) -> None:
        """
        Update state with variables (delegated to parent).
        
        Args:
            variables: Dictionary of variables to add to state
        """
        super().send_variables(variables)
        
    def send_tools(self, tools: Dict[str, Tool]) -> None:
        """
        Register tools (delegated to parent).
        
        Args:
            tools: Dictionary of tools to register
        """
        super().send_tools(tools)
        
    def __getattr__(self, name: str) -> Any:
        """
        Delegate any missing attributes to the parent class.
        
        This ensures complete compatibility with LocalPythonExecutor
        even if new methods or properties are added in future versions.
        """
        return getattr(super(), name)
        
    def __repr__(self) -> str:
        """String representation of the executor."""
        return f"WorkspacePythonExecutor(workspace_dir='{self.workspace_dir}')"