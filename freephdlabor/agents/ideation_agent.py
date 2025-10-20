"""
IdeationAgent implementation using smolagents framework.
Based on ai_scientist/generate_ideas.py functionality.
Follows tool-centric design philosophy where tools are executors, not prompt generators.
"""

import json
import os
from typing import List, Dict, Optional, Any
from .base_research_agent import BaseResearchAgent

from ..toolkits.paper_search_tool import PaperSearchTool
from ..toolkits.general_tools.fetch_arxiv_papers.fetch_arxiv_papers_tools import FetchArxivPapersTool
from ..toolkits.general_tools.open_deep_search.ods_tool import OpenDeepSearchTool
from ..toolkits.writeup.vlm_document_analysis_tool import VLMDocumentAnalysisTool
from ..toolkits.generate_idea_tool import GenerateIdeaTool
from ..toolkits.check_idea_novelty_tool import CheckIdeaNoveltyTool
from ..toolkits.refine_idea_tool import RefineIdeaTool
from ..toolkits.general_tools.file_editing.file_editing_tools import (
    SeeFile, CreateFileWithContent, ModifyFile, ListDir, SearchKeyword, DeleteFileOrFolder
)
from ..prompts.ideation_instructions import get_ideation_system_prompt


class IdeationAgent(BaseResearchAgent):
    """
    An agent specialized in research idea generation, refinement, and novelty checking.
    
    Design Philosophy:
    - Pure tool orchestrator - no duplicate methods like generate_ideas()
    - Tools are executors that call LLM and return results, not prompts
    - Agent orchestrates through natural language conversation
    - Single responsibility: each tool has one clear purpose
    
    Workflow Process:
    1. Literature search using OpenDeepSearchTool and FetchArxivPapersTool
    2. Idea generation using GenerateIdeaTool based on literature gaps
    3. Idea refinement using RefineIdeaTool for feasibility and novelty
    4. Document analysis using TextInspectorTool for deeper understanding
    5. Return structured, validated research ideas ready for experimentation
    """
    
    def __init__(self, model, workspace_dir=None, **kwargs):
        """
        Initialize the IdeationAgent.
        
        Args:
            model: The LLM model to use for the agent
            workspace_dir: Optional workspace directory for file operations
            **kwargs: Additional arguments passed to BaseResearchAgent
        """
        # Convert workspace_dir to absolute path immediately to prevent nested directory issues
        if workspace_dir:
            workspace_dir = os.path.abspath(workspace_dir)
            
        # Legacy compatibility: set agent_folder for any code that might reference it
        if workspace_dir:
            self.agent_folder = os.path.join(workspace_dir, "ideation_agent")
        
        # Initialize tools - these are the primary executors
        # NOTE: Tools get raw model for efficiency, agents use LoggingLiteLLMModel for decision tracking
        from ..toolkits.model_utils import get_raw_model
        raw_model = get_raw_model(model)
        
        tools = [
            # comment out for now, could be modified to work later
            # PaperSearchTool(),
            # CheckIdeaNoveltyTool(model=raw_model),  # Pass raw model for efficiency
            OpenDeepSearchTool(model_name=model.model_id),
            FetchArxivPapersTool(working_dir=workspace_dir),  # Pass workspace_dir for proper file organization
            GenerateIdeaTool(model=raw_model),  # Tools use raw model for efficiency
            RefineIdeaTool(model=raw_model),  # Tools use raw model for efficiency
            VLMDocumentAnalysisTool(model=raw_model, working_dir=workspace_dir),  # Superior PDF analysis with visual understanding
        ]
        
        # Add file editing tools if workspace_dir is provided
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
        
        # Generate complete system prompt using template
        system_prompt = get_ideation_system_prompt(
            tools=tools,
            managed_agents=None  # IdeationAgent typically doesn't manage other agents
        )

        # Initialize BaseResearchAgent with specialized tools
        super().__init__(
            model=model,  # Pass original model, BaseResearchAgent will handle logging
            agent_name="ideation_agent",
            workspace_dir=workspace_dir,
            tools=tools,
            **kwargs
        )

        # Replace the system prompt template with our custom one for complete control
        self.prompt_templates["system_prompt"] = system_prompt 
        
        # Resume memory if possible
        self.resume_memory()