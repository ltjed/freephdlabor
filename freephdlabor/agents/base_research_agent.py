"""
BaseResearchAgent - Minimal base class for all research agents.

This class eliminates code duplication by providing common functionality
that all research agents need: workspace setup, logging, and workspace-aware
Python code execution.
"""

import os
import json
from smolagents import CodeAgent
from smolagents.agents import PlanningPromptTemplate
from ..interpreters import WorkspacePythonExecutor
from ..logging.llm_logger import create_agent_logging_model
from ..prompts.research_planning_templates import get_research_planning_templates
from .context_compaction import ContextMonitoringCallback


class BaseResearchAgent(CodeAgent):
    """
    Base class for all research agents in the multi-agent system.
    
    Provides minimal common functionality:
    - Workspace directory setup and agent folder creation
    - Workspace-aware Python executor (WorkspacePythonExecutor)
    - Standardized LLM call logging
    
    This eliminates code duplication across IdeationAgent, ExperimentationAgent,
    WriteupAgent, and ManagerAgent while maintaining identical behavior.
    """
    
    def __init__(self, model, agent_name, workspace_dir=None, planning_interval=None, **kwargs):
        """
        Initialize base research agent with common functionality.

        Args:
            model: The LLM model to use for the agent
            agent_name: Name identifier for this agent (e.g., "ideation_agent")
            workspace_dir: Optional workspace directory for file operations
            planning_interval: Optional interval for planning steps (e.g., 5 = plan every 5 steps)
            **kwargs: Additional arguments passed to CodeAgent (including system_prompt)
        """
        # Store agent identification and workspace
        self.agent_name = agent_name
        self.back_up_agent_name = agent_name
        self.workspace_dir = workspace_dir
        
        # Create agent-specific folder in workspace
        if workspace_dir:
            os.makedirs(os.path.join(workspace_dir, self.agent_name), exist_ok=True)
        
        # Wrap model with logging for agent LLM calls
        if workspace_dir:
            logged_model = create_agent_logging_model(
                base_model=model,
                agent_type=self.__class__.__name__,
                agent_name=self.agent_name,
                workspace_dir=workspace_dir
            )
        else:
            logged_model = model
        
        # Set up automatic context compaction if enabled
        enable_auto_compaction = kwargs.pop('enable_auto_compaction', True)  # Default enabled
        token_threshold = kwargs.pop('token_threshold', None)  # Auto-calculate from model if None
        keep_recent_steps = kwargs.pop('keep_recent_steps', 3)  # Keep last 3 steps
        safety_margin = kwargs.pop('safety_margin', 0.75)  # Use 75% of model context before compaction
        
        # Add context monitoring callback if auto compaction is enabled
        if enable_auto_compaction:
            context_callback = ContextMonitoringCallback(
                model=model,  # Use original model for compaction and context limit detection
                token_threshold=token_threshold,  # Auto-calculated if None
                keep_recent_steps=keep_recent_steps,
                safety_margin=safety_margin
            )
            # Add to existing callbacks or create new list
            step_callbacks = kwargs.get('step_callbacks', [])
            step_callbacks.append(context_callback)
            kwargs['step_callbacks'] = step_callbacks

        # Handle planning templates setup if planning is enabled
        if planning_interval is not None:
            planning_templates = get_research_planning_templates()
            # If prompt_templates not already provided, create with our planning templates
            if 'prompt_templates' not in kwargs:
                from smolagents.agents import PromptTemplates, ManagedAgentPromptTemplate, FinalAnswerPromptTemplate
                kwargs['prompt_templates'] = PromptTemplates(
                    system_prompt="",  # Will be set by individual agents after init
                    planning=PlanningPromptTemplate(
                        initial_plan=planning_templates['initial_plan'],
                        update_plan_pre_messages=planning_templates['update_plan_pre_messages'],
                        update_plan_post_messages=planning_templates['update_plan_post_messages']
                    ),
                    managed_agent=ManagedAgentPromptTemplate(
                        task="Task for {{name}}: {{task}}",
                        report="Report from {{name}}: {{final_answer}}"
                    ),
                    final_answer=FinalAnswerPromptTemplate(
                        pre_messages="Please provide a final answer based on the above analysis:",
                        post_messages="Task: {{task}}"
                    )
                )
            else:
                # Update existing prompt_templates with planning templates
                if 'planning' not in kwargs['prompt_templates']:
                    kwargs['prompt_templates']['planning'] = PlanningPromptTemplate(
                        initial_plan=planning_templates['initial_plan'],
                        update_plan_pre_messages=planning_templates['update_plan_pre_messages'],
                        update_plan_post_messages=planning_templates['update_plan_post_messages']
                    )
        # When planning is disabled, don't create any prompt_templates
        # Individual agents will create them after super().__init__() as before
        
        # Initialize CodeAgent with logged model, markdown code blocks, and planning support
        # Individual agents will add their specific tools and instructions via system_prompt kwarg
        super().__init__(
            model=logged_model,
            code_block_tags="markdown",  # Use markdown format for better Gemini compatibility
            planning_interval=planning_interval,  # Enable planning if specified
            **kwargs
        )
        
    def create_python_executor(self):
        """
        Override to use WorkspacePythonExecutor that runs code in workspace directory.
        
        This ensures all agent-generated Python code executes in the workspace
        directory context, solving working directory confusion issues.
        """
        if hasattr(self, 'workspace_dir') and self.workspace_dir:
            return WorkspacePythonExecutor(
                workspace_dir=self.workspace_dir,
                additional_authorized_imports=self.additional_authorized_imports
            )
        else:
            # Fallback to default behavior if no workspace configured
            return super().create_python_executor()

    def save_memory(self):
        """
        Save the agent's memory to a JSONL file in the agent's workspace directory.
        Uses high-level memory API (get_full_steps) to retrieve memory state as dictionaries.
        Each memory step is saved as one JSON object per line.
        """
        if not self.workspace_dir:
            return  # No workspace configured; nothing to do
        
        # Define the memory file path: <workspace_dir>/<agent_name>/<agent_name>_memory.jsonl
        memory_path = os.path.join(self.workspace_dir, self.back_up_agent_name, f"{self.back_up_agent_name}_memory.jsonl")
        
        try:
            # Import make_json_serializable from smolagents
            from smolagents.utils import make_json_serializable
            
            with open(memory_path, 'w') as f:
                # Save system prompt if available (system_prompt is a SystemPromptStep object)
                if hasattr(self.memory, 'system_prompt') and self.memory.system_prompt is not None:
                    system_prompt_data = {
                        "type": "system_prompt",
                        "content": self.memory.system_prompt.system_prompt  # Access the string attribute
                    }
                    json.dump(system_prompt_data, f)
                    f.write("\n")
                
                # Get all steps as dictionaries using the high-level API
                # get_full_steps() returns [step.dict() for step in self.steps]
                full_steps = self.memory.get_full_steps()
                
                for step_data in full_steps:
                    # Remove image data which can't be serialized
                    if "observations_images" in step_data:
                        step_data["observations_images"] = None
                    if "task_images" in step_data:
                        step_data["task_images"] = None
                    
                    # Apply make_json_serializable to ensure everything is JSON-safe
                    # This handles any remaining ChatMessage or other objects
                    serializable_data = make_json_serializable(step_data)
                    
                    # Write the step_data as JSON
                    json.dump(serializable_data, f)
                    f.write("\n")
                    
        except Exception as e:
            print(f"Failed to save memory!!! {e}")
            import traceback
            traceback.print_exc()


    def resume_memory(self):
        """
        Load (resume) the agent's memory from a JSONL file in the agent's workspace.
        Reconstructs the memory using the saved steps.
        If no file is found, nothing happens.
        """
        if not self.workspace_dir:
            return  # No workspace configured; cannot load memory
        
        memory_path = os.path.join(self.workspace_dir, self.back_up_agent_name, f"{self.back_up_agent_name}_memory.jsonl")
        if not os.path.isfile(memory_path):
            return  # Memory file doesn't exist; skip loading
        
        try:
            # Import necessary classes
            try:
                from smolagents.memory import (
                    AgentMemory, SystemPromptStep, TaskStep, 
                    ActionStep, PlanningStep, FinalAnswerStep
                )
            except ImportError:
                from smolagents import AgentMemory
                from smolagents.memory import ActionStep, PlanningStep, TaskStep
                try:
                    from smolagents.memory import FinalAnswerStep
                except ImportError:
                    FinalAnswerStep = None
            
            try:
                from smolagents.models import ChatMessage
            except ImportError:
                # Define a minimal ChatMessage if not available
                class ChatMessage:
                    def __init__(self, role, content):
                        self.role = role
                        self.content = content
                    
                    @classmethod
                    def from_dict(cls, d):
                        return cls(d.get("role", ""), d.get("content", ""))
            
            # Read all lines from the memory file
            with open(memory_path, 'r') as f:
                lines = f.read().splitlines()
            
            if not lines:
                return  # Empty file, nothing to load
            
            # Parse entries and reconstruct memory
            system_prompt_str = None
            steps_to_load = []
            
            for line in lines:
                entry = json.loads(line)
                
                # Check if this is the system prompt
                if entry.get("type") == "system_prompt":
                    system_prompt_str = entry.get("content")
                    continue
                
                # Otherwise, it's a step - reconstruct ChatMessage objects if present
                # The .dict() method serializes messages properly, so we need to reconstruct them
                if "model_input_messages" in entry and entry["model_input_messages"]:
                    msgs = []
                    for msg in entry["model_input_messages"]:
                        if isinstance(msg, dict):
                            # Reconstruct ChatMessage from dict
                            msgs.append(ChatMessage(
                                role=msg.get("role", ""),
                                content=msg.get("content", "")
                            ))
                        else:
                            msgs.append(msg)
                    entry["model_input_messages"] = msgs
                
                if "model_output_message" in entry and entry["model_output_message"] is not None:
                    msg = entry["model_output_message"]
                    if isinstance(msg, dict):
                        entry["model_output_message"] = ChatMessage(
                            role=msg.get("role", ""),
                            content=msg.get("content", "")
                        )
                
                # Reconstruct ToolCall objects if present
                if "tool_calls" in entry and entry["tool_calls"]:
                    from smolagents.memory import ToolCall
                    tool_calls = []
                    for tc in entry["tool_calls"]:
                        if isinstance(tc, dict) and "function" in tc:
                            # Reconstruct from the dict format
                            func = tc["function"]
                            tool_calls.append(ToolCall(
                                name=func.get("name", ""),
                                arguments=func.get("arguments", {}),
                                id=tc.get("id", "")
                            ))
                        else:
                            tool_calls.append(tc)
                    entry["tool_calls"] = tool_calls
                
                # Leave Timing, TokenUsage, and AgentError as dicts
                # They don't need to be reconstructed as objects for the memory to work
                # The step classes can handle them as dicts or will reconstruct them internally
                
                # Determine step type and instantiate
                step_obj = None
                try:
                    if "task" in entry:
                        step_obj = TaskStep(**entry)
                    elif "plan" in entry:
                        step_obj = PlanningStep(**entry)
                    elif "step_number" in entry:
                        step_obj = ActionStep(**entry)
                    elif "output" in entry and FinalAnswerStep is not None:
                        step_obj = FinalAnswerStep(**entry)
                except Exception as e:
                    print(f"Failed to reconstruct step: {e}")
                    continue
                
                if step_obj is not None:
                    steps_to_load.append(step_obj)
            
            # Get current system prompt as fallback
            current_system_prompt = None
            if hasattr(self.memory, 'system_prompt') and self.memory.system_prompt is not None:
                current_system_prompt = self.memory.system_prompt.system_prompt
            
            # Create new memory with loaded system prompt (AgentMemory takes a string)
            new_memory = AgentMemory(
                system_prompt=system_prompt_str if system_prompt_str is not None else current_system_prompt
            )
            
            # Add all loaded steps to the new memory
            new_memory.steps = steps_to_load
            
            # Replace the agent's current memory with the reconstructed memory
            self.memory = new_memory
        except Exception as e:
            print(f"Memory load failed!!! {e}")
            import traceback
            traceback.print_exc()