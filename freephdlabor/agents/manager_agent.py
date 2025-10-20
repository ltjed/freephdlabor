"""
ManagerAgent implementation using smolagents framework.
Orchestrates IdeationAgent and other agents in the multi-agent system.
"""

import os
from typing import List
from .base_research_agent import BaseResearchAgent

from .reviewer_agent import ReviewerAgent
from .ideation_agent import IdeationAgent
from .experimentation_agent import ExperimentationAgent
from .resource_preparation_agent import ResourcePreparationAgent
from .writeup_agent import WriteupAgent
from ..toolkits.general_tools.file_editing.file_editing_tools import (
    SeeFile,
    CreateFileWithContent,
    ModifyFile,
    ListDir,
    SearchKeyword,
    DeleteFileOrFolder,
)
from ..prompts.manager_instructions import get_manager_system_prompt


class ManagerAgent(BaseResearchAgent):
    """
    A manager agent that orchestrates other agents in the AI scientist system.
    This agent decides which specialist agent to use to accomplish a task.

    Workflow Intelligence:
    - Automatically delegates idea generation to IdeationAgent
    - Delegates experiment execution to ExperimentationAgent
    - Delegates paper writing to WriteupAgent
    - Handles failures gracefully without generating synthetic results
    - Manages workspace coordination between agents
    """

    def __init__(
        self, model, interpreter, workspace_dir=None, managed_agents=None, **kwargs
    ):
        """
        Initialize the ManagerAgent.

        Args:
            model: The LLM model to use for the agent.
            interpreter: Code interpreter for the agent.
            workspace_dir: Optional workspace directory for agent coordination.
            managed_agents: Optional list of pre-initialized agents to manage.
                          If None, will create default agents internally.
            **kwargs: Additional arguments passed to BaseResearchAgent.
        """
        # Store the interpreter for later use (BaseResearchAgent will handle workspace executor)
        self.interpreter = interpreter

        # Create inter-agent messages folder (specific to ManagerAgent)
        if workspace_dir:
            os.makedirs(
                os.path.join(workspace_dir, "inter_agent_messages"), exist_ok=True
            )

        # Use provided managed agents or create them if not provided
        if managed_agents is not None:
            # Use pre-initialized agents (recommended approach)
            self.managed_agents = managed_agents
        else:
            # Fallback: Create agents internally (legacy behavior)
            # Essential imports for tool-centric agents (shared across all agents)
            essential_imports = kwargs.get("additional_authorized_imports", [])

            # Create managed agents for delegation - they will initialize their own file editing tools
            # Note: Managed agents will get their own logging wrappers in their constructors
            from ..prompts.ideation_instructions import get_ideation_system_prompt
            ideation_agent = IdeationAgent(
                model=model,  # Pass original model, they'll wrap it themselves
                workspace_dir=workspace_dir,
                name="ideation_agent",
                description=f"""A specialist agent for generating, refining, and evaluating research ideas.

--- SYSTEM INSTRUCTIONS ---
{get_ideation_system_prompt()}
--- END SYSTEM INSTRUCTIONS ---""",
                additional_authorized_imports=essential_imports,
            )

            from ..prompts.experimentation_instructions import get_experimentation_system_prompt
            experimentation_agent = ExperimentationAgent(
                model=model,  # Pass original model, they'll wrap it themselves
                workspace_dir=workspace_dir,
                name="experimentation_agent",
                description=f"""A specialist agent for running experiments and analyzing results using RunExperimentTool.

--- SYSTEM INSTRUCTIONS ---
{get_experimentation_system_prompt()}
--- END SYSTEM INSTRUCTIONS ---""",
                additional_authorized_imports=essential_imports,
            )

            from ..prompts.writeup_instructions import get_writeup_system_prompt
            writeup_agent = WriteupAgent(
                model=model,  # Pass original model, they'll wrap it themselves
                workspace_dir=workspace_dir,
                name="writeup_agent",
                description=f"""A specialist agent for academic paper writing that works with pre-organized resources from ResourcePreparationAgent.

--- SYSTEM INSTRUCTIONS ---
{get_writeup_system_prompt(tools=[], managed_agents=None)}
--- END SYSTEM INSTRUCTIONS ---""",
                additional_authorized_imports=essential_imports,
            )

            from ..prompts.resource_preparation_instructions import get_resource_preparation_system_prompt
            resource_preparation_agent = ResourcePreparationAgent(
                model=model,  # Pass original model, they'll wrap it themselves
                workspace_dir=workspace_dir,
                name="resource_preparation_agent",
                description=f"""A comprehensive resource organization agent that prepares complete experimental documentation for WriteupAgent.

Key Functions: Locates experiment results folders, creates paper_workspace/ workspace, links experiment data using symlinks/copies, generates complete file structure analysis with descriptions of EVERY file found, creates comprehensive bibliography based on full experimental understanding.

Key Tools: ExperimentLinkerTool, CitationSearchTool, VLMDocumentAnalysisTool, file editing tools.

Approach: Comprehensive documentation of all experimental artifacts without selectivity. Creates complete file tree structure, reads actual content of every file (VLM for images), and provides complete resource inventory. WriteupAgent can then selectively choose what to use from the comprehensive documentation.

--- SYSTEM INSTRUCTIONS ---
{get_resource_preparation_system_prompt()}
--- END SYSTEM INSTRUCTIONS ---""",
                additional_authorized_imports=essential_imports,
            )

            from ..prompts.reviewer_instructions import get_reviewer_system_prompt
            reviewer_agent = ReviewerAgent(
                model=model,  # Pass original model, they'll wrap it themselves
                workspace_dir=workspace_dir,
                name="reviewer_agent",
                description=f"""A specialist agent for peer-reviewing AI research paper.

--- SYSTEM INSTRUCTIONS ---
{get_reviewer_system_prompt()}
--- END SYSTEM INSTRUCTIONS ---""",
                additional_authorized_imports=essential_imports,
            )

            self.managed_agents = [ideation_agent, experimentation_agent, resource_preparation_agent, writeup_agent, reviewer_agent]

        # Build dynamic agent list for prompt
        available_agents = [agent.name for agent in self.managed_agents]

        # Initialize file editing tools for ManagerAgent
        file_editing_tools = []
        if workspace_dir:
            file_editing_tools = [
                SeeFile(working_dir=workspace_dir),
                CreateFileWithContent(working_dir=workspace_dir),
                ModifyFile(working_dir=workspace_dir),
                ListDir(working_dir=workspace_dir),
                SearchKeyword(working_dir=workspace_dir),
                DeleteFileOrFolder(working_dir=workspace_dir),
            ]

        tools: List = file_editing_tools

        # Generate complete system prompt using template
        system_prompt = get_manager_system_prompt(
            tools=tools, managed_agents=self.managed_agents
        )

        super().__init__(
            model=model,  # Pass original model, BaseResearchAgent will handle logging
            agent_name="manager_agent",
            workspace_dir=workspace_dir,
            tools=tools,
            managed_agents=self.managed_agents,
            **kwargs
        )

        # Set system prompt after initialization (correct smolagents pattern)
        self.prompt_templates["system_prompt"] = system_prompt

    # The run method is inherited from the parent CodeAgent and is what should be called
    # to execute a task. It uses the LLM to reason and create a plan.
    
        # Resume memory if possible
        self.resume_memory()
