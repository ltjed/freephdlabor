"""
ReviewerAgent implementation using smolagents framework.
Minimal implementation focused on running experiments via VLMDocumentAnalysisTool.
Designed to be managed by ManagerAgent for delegation-based workflow.
"""

import os
from typing import Optional
from .base_research_agent import BaseResearchAgent
from ..toolkits.writeup.vlm_document_analysis_tool import VLMDocumentAnalysisTool
from ..prompts.reviewer_instructions import get_reviewer_system_prompt
from ..toolkits.paper_search_tool import PaperSearchTool
from ..toolkits.general_tools.file_editing.file_editing_tools import (
    SeeFile, CreateFileWithContent, ModifyFile, ListDir, SearchKeyword, DeleteFileOrFolder
)

class ReviewerAgent(BaseResearchAgent):
    """
    An agent specialized in running experiments and analyzing results.

    Design Philosophy:
    - Minimal implementation focused on experiment execution
    - Uses RunExperimentTool as primary capability
    - Workspace-aware for file-based collaboration
    - Designed to be managed by ManagerAgent

    Workflow Process:
    1. Receive experiment task from ManagerAgent
    2. Read research idea from workspace files if needed
    3. Execute experiments using RunExperimentTool
    4. Save results to workspace for other agents
    5. Report completion back to ManagerAgent
    """

    def __init__(self, model, workspace_dir: Optional[str] = None, **kwargs):
        """
        Initialize the ReviewerAgent.

        Args:
            model: The LLM model to use for the agent
            workspace_dir: Directory for workspace operations
            **kwargs: Additional arguments passed to BaseResearchAgent
        """
        # Convert workspace_dir to absolute path immediately to prevent nested directory issues
        if workspace_dir:
            workspace_dir = os.path.abspath(workspace_dir)
            
        # Legacy compatibility: set agent_folder for any code that might reference it
        if workspace_dir:
            self.agent_folder = os.path.join(workspace_dir, "reviewer_agent")

        # Initialize tools - minimal set focused on experimentation
        # NOTE: Tools get raw model for efficiency, agents use LoggingLiteLLMModel for decision tracking
        from ..toolkits.model_utils import get_raw_model
        raw_model = get_raw_model(model)
        
        tools = [
            VLMDocumentAnalysisTool(
                model=raw_model, working_dir=workspace_dir
            ),  # Primary tool for analyzing research paper (use raw model)
        ]

        file_editing_tools = [
            SeeFile(working_dir=workspace_dir),
            CreateFileWithContent(working_dir=workspace_dir),
            ModifyFile(working_dir=workspace_dir),
            ListDir(working_dir=workspace_dir),
            SearchKeyword(working_dir=workspace_dir),
            DeleteFileOrFolder(working_dir=workspace_dir),
        ]
        tools.extend(file_editing_tools)

        # Generate complete system prompt using template
        system_prompt = get_reviewer_system_prompt(
            tools=tools,
            managed_agents=None,  # ReviewerAgent typically doesn't manage other agents
        )

        # Initialize BaseResearchAgent with specialized tools
        super().__init__(
            model=model,  # Pass original model, BaseResearchAgent will handle logging
            agent_name="reviewer_agent",
            workspace_dir=workspace_dir,
            tools=tools,
            **kwargs
        )

        # Set system prompt after initialization (correct smolagents pattern)
        self.prompt_templates["system_prompt"] = system_prompt
        
        # Resume memory if possible
        self.resume_memory()
