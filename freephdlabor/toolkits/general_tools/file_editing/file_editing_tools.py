from smolagents.tools import Tool
import os
from typing import Any
import importlib.util


class ListDir(Tool):
    name = "list_dir"
    description = (
        "List files in the chosen directory. Use this to explore the directory structure. "
        "Note: only files under the allowed working directory are accessible."
    )
    inputs = {"directory": {"type": "string", "description": "The directory to check."}}
    output_type = "string"

    def __init__(self, working_dir):
        super().__init__()
        # Always store working_dir as absolute path to prevent any path resolution issues
        self.working_dir = os.path.abspath(working_dir)

    def forward(self, directory: str) -> str:
        try:
            chosen_dir = self._safe_path(directory)
        except PermissionError as e:
            return str(e)
        except FileNotFoundError as e:
            return str(e)
        if not os.path.exists(chosen_dir):
            return f"The directory {directory} does not exist. Please start checking from the root directory."
        files = os.listdir(chosen_dir)
        if files == []:
            return f"The directory {directory} is empty."
        else:
            return '\n'.join(files)

    def _safe_path(self, path: str) -> str:
        """Convert path to absolute workspace path with clear error messages for agents."""
        if not self.working_dir:
            return path
            
        abs_working_dir = os.path.abspath(self.working_dir)
        
        # Check if input path is absolute or relative
        if os.path.isabs(path):
            # Absolute path handling
            abs_path = os.path.abspath(path)
            
            # Check if within workspace
            if abs_path.startswith(abs_working_dir):
                return abs_path
            else:
                # Provide actionable error for agent
                raise PermissionError(
                    f"Access denied: The absolute path '{path}' is outside the workspace. "
                    f"Please use a relative path or an absolute path within '{abs_working_dir}'. "
                    f"Example: Use 'subdirectory/' instead of the full path."
                )
        else:
            # Relative path - join with workspace
            abs_path = os.path.abspath(os.path.join(abs_working_dir, path))
            
            # For directory listing, directories should exist
            if not os.path.exists(abs_path):
                # Provide helpful error for agent
                parent_dir = os.path.dirname(abs_path)
                if os.path.exists(parent_dir):
                    raise FileNotFoundError(
                        f"Directory not found: '{path}' does not exist in the workspace. "
                        f"The parent directory exists. Please check the directory name."
                    )
                else:
                    raise FileNotFoundError(
                        f"Directory not found: '{path}' does not exist in the workspace. "
                        f"The parent directory '{os.path.dirname(path)}' was not found."
                    )
            
            return abs_path


class SeeFile(Tool):
    name = "see_file"
    description = (
        "Read workspace files quickly. Use for code files, configs, logs, and simple text files in your workspace. "
        "Returns clean file content without line numbers. "
        "For PDFs or complex documents, use inspect_file_as_text instead."
    )
    inputs = {"filename": {"type": "string", "description": "Name of the file to check."}}
    output_type = "string"

    def __init__(self, working_dir):
        super().__init__()
        # Always store working_dir as absolute path to prevent any path resolution issues
        self.working_dir = os.path.abspath(working_dir)

    def forward(self, filename: str) -> str:
        try:
            filepath = self._safe_path(filename)
        except PermissionError as e:
            return str(e)
        except FileNotFoundError as e:
            return str(e)
        if not os.path.exists(filepath):
            return f"The file {filename} does not exist."
        with open(filepath, "r") as file:
            content = file.read()
        return content

    def _safe_path(self, path: str) -> str:
        """Convert path to absolute workspace path with clear error messages for agents."""
        if not self.working_dir:
            return path
            
        abs_working_dir = os.path.abspath(self.working_dir)
        
        # Check if input path is absolute or relative
        if os.path.isabs(path):
            # Absolute path handling
            abs_path = os.path.abspath(path)
            
            # Check if within workspace
            if abs_path.startswith(abs_working_dir):
                return abs_path
            else:
                # Provide actionable error for agent
                raise PermissionError(
                    f"Access denied: The absolute path '{path}' is outside the workspace. "
                    f"Please use a relative path or an absolute path within '{abs_working_dir}'. "
                    f"Example: Use 'subdirectory/filename.txt' instead of the full path."
                )
        else:
            # Relative path - join with workspace
            abs_path = os.path.abspath(os.path.join(abs_working_dir, path))
            
            # Return path (existence checks handled by individual tools as needed)
            return abs_path

class ModifyFile(Tool):
    name = "modify_file"
    description = (
        "Modify a plain text file by replacing specific lines with new content. "
        "Only works with plain text files (e.g., .txt, .py, .md). Ensure correct indentation. "
        "Not applicable for binary files such as .pdf, .docx, or spreadsheets."
    )
    inputs = {
        "filename": {"type": "string", "description": "Name of the file to modify."},
        "start_line": {"type": "integer", "description": "Start line number to replace."},
        "end_line": {"type": "integer", "description": "End line number to replace."},
        "new_content": {"type": "string", "description": "New content to insert (with proper indentation)."}
    }
    output_type = "string"

    def __init__(self, working_dir):
        super().__init__()
        # Always store working_dir as absolute path to prevent any path resolution issues
        self.working_dir = os.path.abspath(working_dir)

    def forward(self, filename: str, start_line: int, end_line: int, new_content: str) -> str:
        try:
            filepath = self._safe_path(filename)
        except PermissionError as e:
            return str(e)
        if not os.path.exists(filepath):
            return f"The file {filename} does not exist."
        with open(filepath, "r+") as file:
            lines = file.readlines()
            lines[start_line - 1:end_line] = [new_content + "\n"]
            file.seek(0)
            file.truncate()
            file.write("".join(lines))
        return "Content modified."

    def _safe_path(self, path: str) -> str:
        """Convert path to absolute workspace path with clear error messages for agents."""
        if not self.working_dir:
            return path
            
        abs_working_dir = os.path.abspath(self.working_dir)
        
        # Check if input path is absolute or relative
        if os.path.isabs(path):
            # Absolute path handling
            abs_path = os.path.abspath(path)
            
            # Check if within workspace
            if abs_path.startswith(abs_working_dir):
                return abs_path
            else:
                # Provide actionable error for agent
                raise PermissionError(
                    f"Access denied: The absolute path '{path}' is outside the workspace. "
                    f"Please use a relative path or an absolute path within '{abs_working_dir}'. "
                    f"Example: Use 'subdirectory/filename.txt' instead of the full path."
                )
        else:
            # Relative path - join with workspace
            abs_path = os.path.abspath(os.path.join(abs_working_dir, path))
            
            # Return path (existence checks handled by individual tools as needed)
            return abs_path

class CreateFileWithContent(Tool):
    name = "create_file_with_content"
    description = (
        "Create a new plain text file (e.g., .txt, .py, .md) and write content into it. "
        "This tool does not support creating binary files such as .pdf, .docx, or images."
    )
    inputs = {
        "filename": {"type": "string", "description": "Name of the file to create."},
        "content": {"type": "string", "description": "Content to write into the file."}
    }
    output_type = "string"

    def __init__(self, working_dir):
        super().__init__()
        # Always store working_dir as absolute path to prevent any path resolution issues
        self.working_dir = os.path.abspath(working_dir)

    def forward(self, filename: str, content: str) -> str:
        try:
            filepath = self._safe_path(filename)
        except PermissionError as e:
            return str(e)
        with open(filepath, "w") as file:
            file.write(content)
        return "File created successfully."

    def _safe_path(self, path: str) -> str:
        """Convert path to absolute workspace path with clear error messages for agents."""
        if not self.working_dir:
            return path
            
        abs_working_dir = os.path.abspath(self.working_dir)
        
        # Check if input path is absolute or relative
        if os.path.isabs(path):
            # Absolute path handling
            abs_path = os.path.abspath(path)
            
            # Check if within workspace
            if abs_path.startswith(abs_working_dir):
                return abs_path
            else:
                # Provide actionable error for agent
                raise PermissionError(
                    f"Access denied: The absolute path '{path}' is outside the workspace. "
                    f"Please use a relative path or an absolute path within '{abs_working_dir}'. "
                    f"Example: Use 'subdirectory/filename.txt' instead of the full path."
                )
        else:
            # Relative path - join with workspace
            abs_path = os.path.abspath(os.path.join(abs_working_dir, path))
            
            # Return path (existence checks handled by individual tools as needed)
            return abs_path

class SearchKeyword(Tool):
    name = "search_keyword"
    description = (
        "Search for a keyword in a plain text file or recursively in all plain text files within a folder. "
        "Returns matching lines with file names, line numbers and context lines before and after each match. "
        "Only supports plain text files (e.g., .txt, .py, .md). Not suitable for binary formats like .pdf, .docx, .xlsx."
    )
    inputs = {
        "path": {"type": "string", "description": "Path to the file or folder to search in."},
        "keyword": {"type": "string", "description": "Keyword to search for."},
        "context_lines": {
            "type": "integer",
            "description": "Number of lines to include before and after each match."
        }
    }
    output_type = "string"

    def __init__(self, working_dir):
        super().__init__()
        # Always store working_dir as absolute path to prevent any path resolution issues
        self.working_dir = os.path.abspath(working_dir)

    def forward(self, path: str, keyword: str, context_lines: int) -> str:
        try:
            target_path = self._safe_path(path)
        except PermissionError as e:
            return str(e)
        if not os.path.exists(target_path):
            return f"The path '{path}' does not exist."

        if os.path.isfile(target_path):
            return self._search_in_file(target_path, keyword, context_lines, display_path=path)
        elif os.path.isdir(target_path):
            results = []
            for root, _, files in os.walk(target_path):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    rel_path = os.path.relpath(fpath, self.working_dir)
                    try:
                        result = self._search_in_file(fpath, keyword, context_lines, display_path=rel_path)
                        if "No matches found" not in result:
                            results.append(result)
                    except Exception as e:
                        results.append(f"[{rel_path}]: Error reading file ({e})")
            return "\n\n".join(results) if results else f"No matches found for '{keyword}' in folder '{path}'."
        else:
            return f"The path '{path}' is neither a file nor a directory."

    def _search_in_file(self, filepath: str, keyword: str, context_lines: int, display_path: str) -> str:
        try:
            with open(filepath, "r", encoding="utf-8") as file:
                lines = file.readlines()
        except UnicodeDecodeError:
            return f"[{display_path}]: Cannot read binary or non-text file."

        num_lines = len(lines)
        match_indices = [i for i, line in enumerate(lines) if keyword in line]

        if not match_indices:
            return f"[{display_path}]: No matches found for '{keyword}'."

        output_lines = set()
        for idx in match_indices:
            start = max(0, idx - context_lines)
            end = min(num_lines, idx + context_lines + 1)
            output_lines.update(range(start, end))

        sorted_output = sorted(output_lines)
        formatted_output = [f"{i+1}: {lines[i].rstrip()}" for i in sorted_output]

        return f"--- Matches in [{display_path}] ---\n" + "\n".join(formatted_output)

    def _safe_path(self, path: str) -> str:
        """Convert path to absolute workspace path with clear error messages for agents."""
        if not self.working_dir:
            return path
            
        abs_working_dir = os.path.abspath(self.working_dir)
        
        # Check if input path is absolute or relative
        if os.path.isabs(path):
            # Absolute path handling
            abs_path = os.path.abspath(path)
            
            # Check if within workspace
            if abs_path.startswith(abs_working_dir):
                return abs_path
            else:
                # Provide actionable error for agent
                raise PermissionError(
                    f"Access denied: The absolute path '{path}' is outside the workspace. "
                    f"Please use a relative path or an absolute path within '{abs_working_dir}'. "
                    f"Example: Use 'subdirectory/filename.txt' instead of the full path."
                )
        else:
            # Relative path - join with workspace
            abs_path = os.path.abspath(os.path.join(abs_working_dir, path))
            
            # Return path (existence checks handled by individual tools as needed)
            return abs_path

class DeleteFileOrFolder(Tool):
    name = "delete_file_or_folder"
    description = (
        "Delete a specified file or folder. This action is irreversible."
        "If no filename is provided, the tool will delete everything in the working directory."
        "Only files under the allowed working directory are accessible."
    )
    inputs = {"filename": {"type": "string", "description": "Name of the file or folder to delete."}}
    output_type = "string"

    def __init__(self, working_dir):
        super().__init__()
        # Always store working_dir as absolute path to prevent any path resolution issues
        self.working_dir = os.path.abspath(working_dir)
    
    def forward(self, filename: str) -> str:
        if filename == "":
            abs_working_dir = os.path.abspath(self.working_dir)
            # Only delete inside the working directory
            for root, dirs, files in os.walk(abs_working_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            return "All files and folders in the working directory have been deleted."
        else:
            try:
                filepath = self._safe_path(filename)
            except PermissionError as e:
                return str(e)
            if os.path.exists(filepath):
                if os.path.isfile(filepath):
                    os.remove(filepath)
                    return f"The file {filename} has been deleted."
                elif os.path.isdir(filepath):
                    os.rmdir(filepath)
                    return f"The folder {filename} has been deleted."
                else:
                    return f"The path {filename} is neither a file nor a folder."
            else:
                return f"The file or folder {filename} does not exist."

    def _safe_path(self, path: str) -> str:
        """Convert path to absolute workspace path with clear error messages for agents."""
        if not self.working_dir:
            return path
            
        abs_working_dir = os.path.abspath(self.working_dir)
        
        # Check if input path is absolute or relative
        if os.path.isabs(path):
            # Absolute path handling
            abs_path = os.path.abspath(path)
            
            # Check if within workspace
            if abs_path.startswith(abs_working_dir):
                return abs_path
            else:
                # Provide actionable error for agent
                raise PermissionError(
                    f"Access denied: The absolute path '{path}' is outside the workspace. "
                    f"Please use a relative path or an absolute path within '{abs_working_dir}'. "
                    f"Example: Use 'subdirectory/filename.txt' instead of the full path."
                )
        else:
            # Relative path - join with workspace
            abs_path = os.path.abspath(os.path.join(abs_working_dir, path))
            
            # Return path (existence checks handled by individual tools as needed)
            return abs_path

# class LoadObjectFromPythonFile(Tool):
#     name = "load_object_from_python_file"
#     description = "Load a class or method from a Python file so it can be used by the agent."
#     inputs = {
#         "filename": {"type": "string", "description": "The Python file to load from."},
#         "object_name": {"type": "string", "description": "The name of the class or method to load."}
#     }
#     output_type = "object"  # We return an actual callable Python object

#     def __init__(self, working_dir: str):
#         super().__init__()
#         self.working_dir = working_dir

#     def forward(self, filename: str, object_name: str) -> Any:
#         try:
#             file_path = self._safe_path(filename)
#         except PermissionError as e:
#             raise FileNotFoundError(str(e))
#         if not os.path.exists(file_path):
#             raise FileNotFoundError(f"The file {filename} does not exist.")

#         # Create a module spec
#         module_name = os.path.splitext(os.path.basename(file_path))[0]
#         spec = importlib.util.spec_from_file_location(module_name, file_path)

#         if spec is None or spec.loader is None:
#             raise ImportError(f"Could not load spec for file {filename}")

#         module = importlib.util.module_from_spec(spec)
#         spec.loader.exec_module(module)

#         if not hasattr(module, object_name):
#             raise AttributeError(f"The object {object_name} was not found in {filename}")

#         return getattr(module, object_name)

#     def _safe_path(self, path: str) -> str:
#         # Prevent absolute paths and directory traversal
#         abs_working_dir = os.path.abspath(self.working_dir)
#         abs_path = os.path.abspath(os.path.join(self.working_dir, path))
#         if not abs_path.startswith(abs_working_dir):
#             raise PermissionError("Access outside the working directory is not allowed.")
#         return abs_path