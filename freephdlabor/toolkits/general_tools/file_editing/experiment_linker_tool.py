"""
Tool for linking or copying experiment results folders to writeup workspace.
"""

import os
import shutil
from pathlib import Path
from smolagents import Tool


class ExperimentLinkerTool(Tool):
    name = "experiment_linker_tool"
    description = """
    Link or copy experiment results folder to paper_workspace for organized access.

    This tool safely creates a symlink (preferred) or copies an experiment results folder
    into the paper_workspace directory. Handles path validation and provides fallback
    options if symlink creation fails.

    Usage: Call this tool to make experiment data accessible in paper_workspace
    before generating structure analysis or accessing experimental files.
    """

    inputs = {
        "source_path": {
            "type": "string",
            "description": "Full path to the experiment results folder to link/copy"
        },
        "target_name": {
            "type": "string",
            "description": "Name for the linked folder inside paper_workspace (e.g., 'experiment_data')",
            "nullable": True
        }
    }

    output_type = "string"

    def __init__(self, working_dir: str = None):
        """Initialize ExperimentLinkerTool with workspace directory."""
        super().__init__()
        self.working_dir = os.path.abspath(working_dir) if working_dir else None

    def forward(self, source_path: str, target_name: str = "experiment_data") -> str:
        """
        Link or copy experiment folder to paper_workspace.

        Args:
            source_path: Path to experiment results folder
            target_name: Name for folder in paper_workspace

        Returns:
            JSON string with operation result and target path
        """
        try:
            if not self.working_dir:
                return '{"success": false, "error": "No working directory configured"}'

            source_path = os.path.abspath(source_path)
            if not os.path.exists(source_path):
                return f'{{"success": false, "error": "Source path does not exist: {source_path}"}}'

            if not os.path.isdir(source_path):
                return f'{{"success": false, "error": "Source path is not a directory: {source_path}"}}'

            paper_workspace = os.path.join(self.working_dir, "paper_workspace")
            os.makedirs(paper_workspace, exist_ok=True)

            target_path = os.path.join(paper_workspace, target_name)

            if os.path.exists(target_path):
                return f'{{"success": false, "error": "Target already exists: {target_path}"}}'

            try:
                os.symlink(source_path, target_path)
                operation = "symlink"
            except (OSError, NotImplementedError):
                shutil.copytree(source_path, target_path)
                operation = "copy"

            return f'{{"success": true, "operation": "{operation}", "source": "{source_path}", "target": "{target_path}"}}'

        except Exception as e:
            return f'{{"success": false, "error": "Unexpected error: {str(e)}"}}'