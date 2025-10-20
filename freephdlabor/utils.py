import json
import os
import re
from typing import Optional
from smolagents import LiteLLMModel

# Available models
AVAILABLE_MODELS = [
    # OpenAI GPT-5 models (latest)
    "gpt-5",
    "gpt-5-mini", 
    "gpt-5-nano",
    # OpenAI models
    "gpt-4o",
    "gpt-4.1-mini-2025-04-14",
    # OpenAI reasoning models
    "o4-mini-2025-04-16",
    "o3-2025-04-16",
    "o3-pro-2025-06-10",
    # Claude models (Anthropic)
    "claude-opus-4-20250514",
    "claude-sonnet-4-20250514",
    "claude-sonnet-4-5",
    "claude-sonnet-4-5-20250929",
    # DeepSeek models
    "deepseek-chat",
    "deepseek-coder",
    # grok models
    "grok-4-0709",
    # Google Gemini models
    "gemini-2.5-pro",
    "gemini-2.5-flash",
]


def extract_content_between_markers(response: str, start_marker: str, end_marker: str) -> Optional[str]:
    """
    Extract content between specified start and end markers from a response string.
    
    Args:
        response: The raw response string to extract content from
        start_marker: The starting marker/delimiter
        end_marker: The ending marker/delimiter
        
    Returns:
        Extracted content as string, or None if not found
    """
    try:
        # Build regex pattern to find content between markers
        # Escape special regex characters in markers
        start_escaped = re.escape(start_marker)
        end_escaped = re.escape(end_marker)
        
        # Pattern to match content between start and end markers
        pattern = f"{start_escaped}(.*?){end_escaped}"
        matches = re.findall(pattern, response, re.DOTALL)
        
        if matches:
            # Return the first match, stripped of whitespace
            return matches[0].strip()
        
        return None
        
    except Exception as e:
        return None

def create_model(model_name, reasoning_effort="medium", verbosity="medium", budget_tokens=None):
    """Create a smolagents model based on the model name using API keys from environment variables.

    Args:
        model_name: Name of the model to create
        reasoning_effort: GPT-5 reasoning effort level (minimal, low, medium, high)
        verbosity: GPT-5 verbosity level (low, medium, high)
        budget_tokens: Claude Extended Thinking token budget (min: 1024, recommended: 16k+)
    """
    
    # Model context limits for fallback detection
    # These will be passed as custom attributes to LiteLLMModel via **kwargs
    model_context_limits = {
        # Claude models
        "claude-3-5-sonnet-20241022": 200000,
        "claude-3-5-sonnet-20240620": 200000,
        "claude-3-5-haiku-20241022": 200000,
        "claude-3-opus-20240229": 200000,
        "claude-3-sonnet-20240229": 200000,
        "claude-3-haiku-20240307": 200000,
        "claude-sonnet-4-20250514": 200000,
        "claude-opus-4-20250514": 200000,
        "claude-sonnet-4-5": 200000,
        "claude-sonnet-4-5-20250929": 200000,
        
        # OpenAI GPT-5 models (context window: 256K input, 128K output including reasoning tokens)
        "gpt-5": 256000,
        "gpt-5-mini": 256000, 
        "gpt-5-nano": 256000,
        
        # OpenAI models
        "gpt-4o": 128000,
        "gpt-4o-mini": 128000,
        "gpt-4-turbo": 128000,
        "gpt-4": 8192,
        "gpt-3.5-turbo": 16385,
        "o1-preview": 128000,
        "o1-mini": 128000,
        "o3-mini": 128000,
        "o3-2025-04-16": 200000,
        "o4-mini": 128000,
        "gpt-4.1-mini-2025-04-14": 128000,
        "o4-mini-2025-04-16": 128000,
        
        # Gemini models
        "gemini-1.5-pro": 1000000,
        "gemini-1.5-flash": 1000000,
        "gemini-2.5-pro": 1000000,
        "gemini-2.5-flash": 1000000,
        
        # DeepSeek models
        "deepseek-chat": 64000,
        "deepseek-coder": 64000,
        "deepseek-reasoner": 64000,
        
        # Grok models
        "grok-4-0709": 128000,
    }
    
    # Get context limit for this model (fallback for runtime detection)
    context_limit = model_context_limits.get(model_name, 128000)  # Default 128k
    
    if "claude" in model_name:
        # Claude models with optional Extended Thinking support
        # Note: context_limit is NOT passed to API, it's just metadata

        # Add Extended Thinking if budget_tokens is specified
        extra_kwargs = {}
        if budget_tokens is not None:
            extra_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": budget_tokens
            }

        return LiteLLMModel(
            model_id=f"anthropic/{model_name}",  # Explicit provider prefix for LiteLLM routing
            api_key=os.environ["ANTHROPIC_API_KEY"],
            **extra_kwargs
        )
    elif model_name.startswith("gpt-5"):
        # GPT-5 models with reasoning_effort and verbosity support
        # Global litellm.drop_params = True handles unsupported parameter filtering
        extra_kwargs = {
            "reasoning_effort": reasoning_effort,  # Required for GPT-5
            "verbosity": verbosity,  # Required for GPT-5
        }
        
        return LiteLLMModel(
            model=model_name,
            model_id=model_name,
            api_key=os.environ["OPENAI_API_KEY"],
            **extra_kwargs
        )
    elif "gpt" in model_name or model_name.startswith(("o1-", "o3-", "o4-")):
        return LiteLLMModel(
            model=model_name,
            model_id=model_name,
            api_key=os.environ["OPENAI_API_KEY"],
            context_limit=context_limit,
        )
    elif "deepseek" in model_name:
        return LiteLLMModel(
            model=model_name,
            model_id=model_name,
            api_key=os.environ["DEEPSEEK_API_KEY"],
            api_base="https://api.deepseek.com",
            context_limit=context_limit,
        )
    elif "llama" in model_name:
        return LiteLLMModel(
            model=f"openrouter/{model_name}",
            model_id=model_name,
            api_key=os.environ["OPENROUTER_API_KEY"],
            context_limit=context_limit,
        )
    elif "gemini" in model_name:
        # Add thinking budget for Gemini 2.5 models
        extra_kwargs = {"context_limit": context_limit}
        if "gemini-2.5-pro" in model_name:
            extra_kwargs["thinking_budget"] = 32768
        
        return LiteLLMModel(
            model=f"gemini/{model_name}", 
            model_id=f"gemini/{model_name}", 
            api_key=os.environ["GOOGLE_API_KEY"],
            **extra_kwargs
        )
    else:
        # Default to OpenAI-compatible
        return LiteLLMModel(
            model=model_name,
            model_id=model_name,
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            context_limit=context_limit,
        )

from freephdlabor.agents.manager_agent import ManagerAgent
from freephdlabor.agents.ideation_agent import IdeationAgent
from freephdlabor.agents.experimentation_agent import ExperimentationAgent
from freephdlabor.agents.writeup_agent import WriteupAgent
from freephdlabor.interpreters import WorkspacePythonExecutor
from freephdlabor.agents.reviewer_agent import ReviewerAgent
from freephdlabor.agents.proofreading_agent import ProofreadingAgent

def initialize_agent_system(model, workspace_dir, workspace_interpreter, essential_imports, enable_planning=False, planning_interval=3, interrupt_callback=None):
    """
    Initialize the complete multi-agent system with consistent configuration.

    This function ensures all agents get the same workspace interpreter,
    configuration, and imports, solving the working directory confusion.

    Args:
        model: The LLM model instance
        workspace_dir: Directory where all agents will operate
        workspace_interpreter: Custom interpreter that runs code in workspace
        essential_imports: List of authorized Python imports
        enable_planning: Enable planning feature for research agents
        planning_interval: Interval for planning steps (e.g., 3 = replan every 3 steps)
        interrupt_callback: Setup Interrupt Callback

    Returns:
        ManagerAgent: Configured with pre-initialized specialist agents
    """
    print("🔧 Initializing multi-agent system...")

    # Determine planning configuration
    planning_config = {}
    if enable_planning:
        planning_config = {"planning_interval": planning_interval}
        print(f"📋 Planning enabled: agents will replan every {planning_interval} steps")

    # Create all agents with workspace-aware configuration
    # Each agent overrides create_python_executor() to use WorkspacePythonExecutor
    ideation_agent = IdeationAgent(
        model=model,
        workspace_dir=workspace_dir,
        name="ideation_agent",
        description="A specialist agent for generating, refining, and evaluating research ideas.",
        additional_authorized_imports=essential_imports,
        step_callbacks=[interrupt_callback],
        **planning_config
    )
    print("✅ IdeationAgent initialized")
    
    experimentation_agent = ExperimentationAgent(
        model=model,
        workspace_dir=workspace_dir,
        name="experimentation_agent",
        description="A specialist agent for running experiments and analyzing results using RunExperimentTool.",
        additional_authorized_imports=essential_imports,
        step_callbacks=[interrupt_callback],
        **planning_config
    )
    print("✅ ExperimentationAgent initialized")

    # Initialize ResourcePreparationAgent (NEW - handles heavy preparatory work)
    from freephdlabor.agents.resource_preparation_agent import ResourcePreparationAgent
    resource_preparation_agent = ResourcePreparationAgent(
        model=model,
        workspace_dir=workspace_dir,
        name="resource_preparation_agent",
        description="""A comprehensive resource organization agent that prepares complete experimental documentation for WriteupAgent.

Key Functions: Locates experiment results folders, creates writeup_subspace/ workspace, links experiment data using symlinks/copies, generates complete file structure analysis with descriptions of EVERY file found, creates comprehensive bibliography based on full experimental understanding.

Key Tools: ExperimentLinkerTool, CitationSearchTool, VLMDocumentAnalysisTool, file editing tools.

Approach: Comprehensive documentation of all experimental artifacts without selectivity. Creates complete file tree structure, reads actual content of every file (VLM for images), and provides complete resource inventory. WriteupAgent can then selectively choose what to use from the comprehensive documentation.""",
        additional_authorized_imports=essential_imports,
        step_callbacks=[interrupt_callback],
        **planning_config
    )
    print("✅ ResourcePreparationAgent initialized")

    writeup_agent = WriteupAgent(
        model=model,
        workspace_dir=workspace_dir,
        name="writeup_agent",
        description="A SPECIALIZED agent for LaTeX writing and compilation that expects pre-organized resources from ResourcePreparationAgent.",
        additional_authorized_imports=essential_imports,
        step_callbacks=[interrupt_callback],
        **planning_config
    )
    print("✅ WriteupAgent initialized")

    reviewer_agent = ReviewerAgent(
        model=model,
        workspace_dir=workspace_dir,
        name="reviewer_agent",
        description="A specialist agent for peer-reviewing AI research paper.",
        additional_authorized_imports=essential_imports,
        step_callbacks=[interrupt_callback],
        **planning_config
    )
    print("✅ Reviewer initialized")

    proofreading_agent = ProofreadingAgent(
        model=model,
        workspace_dir=workspace_dir,
        name="proofreading_agent",
        description="A specialist agent for proofreading and quality assurance of LaTeX files in academic papers.",
        additional_authorized_imports=essential_imports,
        step_callbacks=[interrupt_callback],
        **planning_config
    )
    print("✅ ProofreadingAgent initialized")

    # Create ManagerAgent with pre-initialized agents (including NEW ResourcePreparationAgent)
    managed_agents = [ideation_agent, experimentation_agent, resource_preparation_agent, writeup_agent, reviewer_agent, proofreading_agent]
    manager = ManagerAgent(
        model=model,
        interpreter=workspace_interpreter,
        workspace_dir=workspace_dir,
        managed_agents=managed_agents,
        additional_authorized_imports=essential_imports,
        step_callbacks=[interrupt_callback],
    )
    print("✅ ManagerAgent initialized with specialist agents")
    
    return manager

def save_agent_memory(manager):
    # Save All agents' memories
    manager.save_memory()
    if hasattr(manager, 'managed_agents') and isinstance(manager.managed_agents, dict):
        for agent_name, agent in manager.managed_agents.items():
            agent.save_memory()
