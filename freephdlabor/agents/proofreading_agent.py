"""
ProofreadingAgent implementation using smolagents framework.
"""

import os
from typing import Optional
from .base_research_agent import BaseResearchAgent
from ..prompts.proofreading_instructions import get_proofreading_system_prompt
from ..toolkits.general_tools.file_editing.file_editing_tools import (
    SeeFile, ModifyFile, ListDir
)
from ..toolkits.writeup.latex_compiler_tool import LaTeXCompilerTool
from ..toolkits.writeup.vlm_document_analysis_tool import VLMDocumentAnalysisTool



class ProofreadingAgent(BaseResearchAgent):
    """
    An agent specialized in proofreading and quality assurance for research documents.

    Design Philosophy:
    - Minimal implementation focused on proofreading and typo correction in research documents
    - Uses specialized tools for document analysis and proofreading
    - Workspace-aware for file-based collaboration
    - Designed to be managed by ManagerAgent

    Workflow Process:
    1. Receive proofreading task from ManagerAgent
    2. Read research document from workspace files if needed
    3. Analyze document for errors using VLMDocumentAnalysisTool
    4. Apply corrections using Document Editing Tools
    5. Report completion back to ManagerAgent
    """

    def __init__(self, model, workspace_dir: Optional[str] = None, **kwargs):
        """
        Initialize the ProofreadingAgent.

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
            self.agent_folder = os.path.join(workspace_dir, "proofreading_agent")

        # Initialize tools - minimal set focused on experimentation
        # NOTE: Tools get raw model for efficiency, agents use LoggingLiteLLMModel for decision tracking
        from ..toolkits.model_utils import get_raw_model
        raw_model = get_raw_model(model)
        
        tools = [
            LaTeXCompilerTool(working_dir=workspace_dir, model=raw_model),  # Primary tool for regenerating PDF (use raw model)
            VLMDocumentAnalysisTool(working_dir=workspace_dir, model=raw_model),  # Primary tool for document analysis (use raw model)
        ]

        # file editing tools for typo correction
        file_editing_tools = [
            SeeFile(working_dir=workspace_dir),
            ModifyFile(working_dir=workspace_dir),
            ListDir(working_dir=workspace_dir),
        ]
        tools.extend(file_editing_tools)

        # Generate complete system prompt using template
        system_prompt = get_proofreading_system_prompt(
            tools=tools,
            managed_agents=None,  # ProofreadingAgent typically doesn't manage other agents
        )

        # Initialize BaseResearchAgent with specialized tools
        super().__init__(
            model=model,  # Pass original model, BaseResearchAgent will handle logging
            agent_name="proofreading_agent",
            workspace_dir=workspace_dir,
            tools=tools,
            **kwargs
        )

        # Set system prompt after initialization (correct smolagents pattern)
        self.prompt_templates["system_prompt"] = system_prompt
        
        # Resume memory if possible
        self.resume_memory()
