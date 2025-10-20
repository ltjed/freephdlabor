"""
Knowledge Base Repo Addition Tools
This module contains tools for adding new content to a structured knowledge base.
Supports writing new files, copying files or folders from the working directory,
and appending content to existing files. All updates are automatically indexed
for semantic search.
"""

from smolagents.tools import Tool
import shutil
from pathlib import Path
from freephdlabor.toolkits.general_tools.kb_repo_management.repo_indexer import (
    RepoIndexer,
)


class WriteToKnowledgeBase(Tool):
    name = "write_to_knowledge_base"
    description = (
        "Create a new file in the knowledge base and write the given content into it. "
        "If overwrite=True, replaces any existing file. If overwrite=False, adds a numeric suffix to avoid conflict. "
        "Updates the semantic index automatically."
    )
    inputs = {
        "content": {
            "type": "string",
            "description": "Text or code to write into the file.",
        },
        "destination_path": {
            "type": "string",
            "description": "Relative path of the new file in the knowledge base.",
        },
        "overwrite": {
            "type": "boolean",
            "description": "Whether to overwrite if the file already exists.",
        },
    }
    output_type = "string"

    def __init__(self, repo_indexer: RepoIndexer):
        super().__init__()
        self.repo_indexer = repo_indexer
        self.root = Path(repo_indexer.root)

    def _get_unique_path(self, base_path: Path) -> Path:
        counter = 1
        new_path = base_path
        while new_path.exists():
            new_path = base_path.with_name(
                f"{base_path.stem}_{counter}{base_path.suffix}"
            )
            counter += 1
        return new_path

    def _safe_kb_path(self, path: str) -> Path:
        abs_root = self.root.resolve()
        abs_path = (self.root / path).resolve()
        if not str(abs_path).startswith(str(abs_root)):
            raise PermissionError(
                "Access outside the knowledge base root is not allowed."
            )
        return abs_path

    def forward(self, content: str, destination_path: str, overwrite: bool) -> str:
        try:
            dst = self._safe_kb_path(destination_path)
        except PermissionError as e:
            return str(e)

        if dst.exists() and not overwrite:
            dst = self._get_unique_path(dst)

        dst.parent.mkdir(parents=True, exist_ok=True)

        with open(dst, "w", encoding="utf-8") as f:
            f.write(content)

        self.repo_indexer.update_file(dst)

        return f"Wrote content to '{dst.relative_to(self.root)}'. File has been indexed for semantic search."


class CopyToKnowledgeBase(Tool):
    name = "copy_to_knowledge_base"
    description = (
        "Copy a file or folder from the working directory to the knowledge base. "
        "If overwrite=True, merges folders or replaces files. If overwrite=False, adds suffix to avoid conflict. "
        "All new or updated files are indexed for semantic search."
    )
    inputs = {
        "source_path": {
            "type": "string",
            "description": "Path in the working directory.",
        },
        "destination_path": {
            "type": "string",
            "description": "Target path in the knowledge base.",
        },
        "overwrite": {
            "type": "boolean",
            "description": "Whether to overwrite existing files or folders.",
        },
    }
    output_type = "string"

    def __init__(self, repo_indexer: RepoIndexer, working_dir: str):
        super().__init__()
        self.repo_indexer = repo_indexer
        self.working_dir = Path(working_dir)
        self.root = Path(repo_indexer.root)

    def _get_unique_path(self, base_path: Path) -> Path:
        counter = 1
        new_path = base_path
        while new_path.exists():
            new_path = base_path.with_name(
                f"{base_path.stem}_{counter}{base_path.suffix}"
            )
            counter += 1
        return new_path

    def _safe_working_path(self, path: str) -> Path:
        abs_root = self.working_dir.resolve()
        abs_path = (self.working_dir / path).resolve()
        if not str(abs_path).startswith(str(abs_root)):
            raise PermissionError(
                "Access outside the working directory is not allowed."
            )
        return abs_path

    def _safe_kb_path(self, path: str) -> Path:
        abs_root = self.root.resolve()
        abs_path = (self.root / path).resolve()
        if not str(abs_path).startswith(str(abs_root)):
            raise PermissionError(
                "Access outside the knowledge base root is not allowed."
            )
        return abs_path

    def forward(self, source_path: str, destination_path: str, overwrite: bool) -> str:
        try:
            src = self._safe_working_path(source_path)
            dst = self._safe_kb_path(destination_path)
        except PermissionError as e:
            return str(e)

        if not src.exists():
            return f"Error: source '{source_path}' does not exist in the working directory."

        if dst.exists() and not overwrite:
            dst = self._get_unique_path(dst)

        dst.parent.mkdir(parents=True, exist_ok=True)

        if src.is_dir():
            if dst.exists() and overwrite:
                dst.mkdir(parents=True, exist_ok=True)
                for item in src.rglob("*"):
                    target = dst / item.relative_to(src)
                    if item.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(item, target)
            else:
                shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

        updated_files = []
        if dst.is_dir():
            for file in dst.rglob("*.*"):
                self.repo_indexer.update_file(file)
                updated_files.append(str(file.relative_to(self.root)))
        else:
            self.repo_indexer.update_file(dst)
            updated_files.append(str(dst.relative_to(self.root)))

        return (
            f"Copied '{source_path}' to '{dst.relative_to(self.root)}'. "
            f"Indexed {len(updated_files)} file(s): {', '.join(updated_files)}."
        )


class AppendToKnowledgeBaseFile(Tool):
    name = "append_to_knowledge_base_file"
    description = (
        "Append new content to a plain text file in the knowledge base. "
        "You can insert at the end, or before/after a specific line using match_string. "
        "If match_string is not found, the content is added to the end. "
        "Automatically reindexes the file for semantic search."
    )
    inputs = {
        "target_file": {
            "type": "string",
            "description": "Relative path of the file in the knowledge base.",
        },
        "new_content": {
            "type": "string",
            "description": "Content to insert into the file.",
        },
        "insert_mode": {
            "type": "string",
            "description": "Content insertion position (default: 'end'):\n" +
                          "• 'end': Append content to the end of the file\n" +
                          "• 'before': Insert content before the line containing match_string\n" +
                          "• 'after': Insert content after the line containing match_string\n" +
                          "Note: 'before' and 'after' modes require match_string parameter",
            "nullable": True,  # Required since it has a default in function signature
        },
        "match_string": {
            "type": "string",
            "description": "String to locate insertion point for 'before' or 'after' modes.",
            "nullable": True,  # Required since it's optional (default=None)
        },
    }
    output_type = "string"

    def __init__(self, repo_indexer: RepoIndexer):
        super().__init__()
        self.root = Path(repo_indexer.root)
        self.repo_indexer = repo_indexer

    def _safe_kb_path(self, path: str) -> Path:
        abs_root = self.root.resolve()
        abs_path = (self.root / path).resolve()
        if not str(abs_path).startswith(str(abs_root)):
            raise PermissionError(
                "Access outside the knowledge base root is not allowed."
            )
        return abs_path

    def forward(
        self,
        target_file: str,
        new_content: str,
        insert_mode: str | None = None,
        match_string: str | None = None,
    ) -> str:
        try:
            filepath = self._safe_kb_path(target_file)
        except PermissionError as e:
            return str(e)

        if not filepath.exists():
            return f"Error: file '{target_file}' does not exist in the knowledge base."

        if not filepath.is_file():
            return f"Error: '{target_file}' is not a file."

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            return f"Error: Cannot read '{target_file}' — it may be a binary or non-text file."

        inserted = False
        new_lines = []

        if insert_mode is None:
            insert_mode = "end"

        if insert_mode == "end" or not match_string:
            lines.append(
                new_content if new_content.endswith("\n") else new_content + "\n"
            )
            inserted = True

        elif insert_mode in {"before", "after"}:
            for i, line in enumerate(lines):
                if match_string in line:
                    if insert_mode == "before":
                        new_lines = lines[:i] + [new_content + "\n"] + lines[i:]
                    else:  # after
                        new_lines = (
                            lines[: i + 1] + [new_content + "\n"] + lines[i + 1 :]
                        )
                    inserted = True
                    break

            if not inserted:
                lines.append(
                    new_content if new_content.endswith("\n") else new_content + "\n"
                )
                inserted = True
            else:
                lines = new_lines

        with open(filepath, "w", encoding="utf-8") as f:
            f.writelines(lines)

        self.repo_indexer.update_file(filepath)

        if insert_mode == "end" or not match_string:
            return f"Appended content to the end of '{target_file}'. File has been reindexed."
        elif inserted:
            return f"Inserted content {insert_mode} line matching '{match_string}' in '{target_file}'. File has been reindexed."
        else:
            return f"Match string '{match_string}' not found. Content appended to end of '{target_file}'. File has been reindexed."
