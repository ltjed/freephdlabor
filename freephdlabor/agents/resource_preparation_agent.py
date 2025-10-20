"""
ResourcePreparationAgent implementation using smolagents framework.
Simplified agent for comprehensive experimental resource organization.
"""

import os
from typing import Optional
from .base_research_agent import BaseResearchAgent

from ..toolkits.writeup.citation_search_tool import CitationSearchTool
from ..toolkits.writeup.vlm_document_analysis_tool import VLMDocumentAnalysisTool
from ..toolkits.general_tools.file_editing.file_editing_tools import (
    SeeFile, CreateFileWithContent, ModifyFile, ListDir, SearchKeyword, DeleteFileOrFolder
)
from ..toolkits.general_tools.file_editing.experiment_linker_tool import ExperimentLinkerTool

from ..prompts.resource_preparation_instructions import get_resource_preparation_system_prompt


class ResourcePreparationAgent(BaseResearchAgent):
    """
    Agent for comprehensive experimental resource organization.
    Locates experiment results, creates complete file inventories, and prepares bibliography.
    """

    def __init__(self, model, workspace_dir: Optional[str] = None, **kwargs):
        """Initialize ResourcePreparationAgent with minimal essential tools."""
        if workspace_dir:
            workspace_dir = os.path.abspath(workspace_dir)

        if workspace_dir:
            self.agent_folder = os.path.join(workspace_dir, "resource_preparation_agent")
            os.makedirs(self.agent_folder, exist_ok=True)

        from ..toolkits.model_utils import get_raw_model
        raw_model = get_raw_model(model)

        tools = [
            ExperimentLinkerTool(working_dir=workspace_dir),
            CitationSearchTool(),
            VLMDocumentAnalysisTool(model=raw_model, working_dir=workspace_dir),
        ]

        if workspace_dir:
            file_editing_tools = [
                SeeFile(working_dir=workspace_dir),
                CreateFileWithContent(working_dir=workspace_dir),
                ModifyFile(working_dir=workspace_dir),
                ListDir(working_dir=workspace_dir),
                SearchKeyword(working_dir=workspace_dir),
                DeleteFileOrFolder(working_dir=workspace_dir),
            ]
            tools.extend(file_editing_tools)

        system_prompt = get_resource_preparation_system_prompt(
            tools=tools,
            managed_agents=None
        )

        default_imports = ['json', 'os', 'subprocess', 'tempfile', 'shutil', 'pathlib', 'glob', 'numpy', 'pandas']
        passed_imports = kwargs.pop('additional_authorized_imports', [])
        combined_imports = list(set(default_imports + passed_imports))

        super().__init__(
            model=model,
            agent_name="resource_preparation_agent",
            workspace_dir=workspace_dir,
            tools=tools,
            additional_authorized_imports=combined_imports,
            max_steps=50,
            **kwargs
        )

        # Set system prompt after initialization (correct smolagents pattern)
        self.prompt_templates["system_prompt"] = system_prompt
        
        # Resume memory if possible
        self.resume_memory()