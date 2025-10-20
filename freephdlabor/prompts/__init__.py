"""
Prompt templates for freephdlabor package.

Organized by function rather than agent for maximum reusability.
"""

# Available instruction modules
from freephdlabor.prompts.ideation_instructions import IDEATION_INSTRUCTIONS
from freephdlabor.prompts.experimentation_instructions import EXPERIMENTATION_INSTRUCTIONS
from freephdlabor.prompts.manager_instructions import MANAGER_INSTRUCTIONS
from freephdlabor.prompts.writeup_instructions import WRITEUP_INSTRUCTIONS

# Workspace management functions
from freephdlabor.prompts.workspace_management import WORKSPACE_GUIDANCE

__all__ = [
    'IDEATION_INSTRUCTIONS',
    'EXPERIMENTATION_INSTRUCTIONS',
    'MANAGER_INSTRUCTIONS', 
    'WRITEUP_INSTRUCTIONS',
    'WORKSPACE_GUIDANCE'
]