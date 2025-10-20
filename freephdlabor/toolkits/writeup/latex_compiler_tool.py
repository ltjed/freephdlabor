"""
LaTeX Compiler Tool - Self-contained LaTeX to PDF compilation with error handling.

This tool provides complete LaTeX compilation without calling other tools.
"""

import json
import os
import subprocess
import tempfile
import shutil
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from smolagents import Tool


class LaTeXCompilerTool(Tool):
    name = "latex_compiler_tool"
    description = """
    Compile LaTeX documents to PDF with comprehensive error handling.
    
    This tool compiles LaTeX source files to PDF format with automatic error fixing.
    It handles common LaTeX compilation issues including:
    - Missing packages (automatically uses standard packages)
    - Syntax errors and formatting issues
    - Bibliography compilation
    - Multi-pass compilation for references
    - Comprehensive error reporting
    
    The tool is self-contained and does not rely on other validation tools.
    
    Input: Path to main .tex file
    Output: JSON with compilation results and PDF path
    """
    
    inputs = {
        "latex_file_path": {
            "type": "string",
            "description": "Path to the main LaTeX file to compile"
        },
        "research_context": {
            "type": "string", 
            "description": "Research context for understanding content (optional)",
            "nullable": True
        },
        "max_fix_iterations": {
            "type": "integer",
            "description": "Maximum number of auto-fix attempts (default: 3)",
            "nullable": True
        },
        "force_compile": {
            "type": "boolean",
            "description": "Force compilation even with warnings (default: False)",
            "nullable": True
        }
    }
    
    output_type = "string"
    
    def __init__(self, working_dir: Optional[str] = None, model=None):
        """Initialize LaTeX Compiler Tool."""
        super().__init__()
        # Convert to absolute path to prevent nested directory issues
        self.working_dir = os.path.abspath(working_dir) if working_dir else None
        from ..model_utils import get_raw_model
        self.model = get_raw_model(model)

        # Initialize citation search tool for automated resolution
        try:
            from .citation_search_tool import CitationSearchTool
            self.citation_search_tool = CitationSearchTool()
        except ImportError:
            self.citation_search_tool = None
        
    def forward(self, latex_file_path: str, research_context: str = "", 
                max_fix_iterations: int = 3, force_compile: bool = False) -> str:
        """
        Compile LaTeX document to PDF.
        
        Args:
            latex_file_path: Path to .tex file
            research_context: Research context (optional)
            max_fix_iterations: Max auto-fix attempts
            force_compile: Force compilation despite warnings
            
        Returns:
            JSON string with compilation results
        """
        try:
            # Resolve file path with duplication protection
            if self.working_dir and not os.path.isabs(latex_file_path):
                latex_file_path = self._safe_path(latex_file_path)
            
            if not os.path.exists(latex_file_path):
                return json.dumps({
                    "success": False,
                    "error": f"LaTeX file not found: {latex_file_path}",
                    "pdf_path": None
                })
            
            compilation_log = []
            fixes_applied = []
            
            # Clean common artifacts first
            cleaned, clean_log = self._clean_latex_artifacts(latex_file_path)
            if cleaned:
                fixes_applied.extend(clean_log)
                compilation_log.append(f"Applied cleaning fixes: {', '.join(clean_log)}")

            # Automated citation resolution before compilation
            citation_resolved, citation_log = self._resolve_citations_automatically(latex_file_path)
            if citation_resolved:
                fixes_applied.extend(citation_log)
                compilation_log.append(f"Applied citation resolution: {', '.join(citation_log)}")

            # Attempt compilation with auto-fixing
            for iteration in range(max_fix_iterations + 1):
                compilation_log.append(f"\n--- Compilation Attempt {iteration + 1} ---")
                
                # Compile to PDF
                compile_result = self._compile_latex_to_pdf(latex_file_path)
                compilation_log.extend(compile_result['log'])
                
                if compile_result['success']:
                    # Success!
                    return json.dumps({
                        "success": True,
                        "pdf_path": compile_result['pdf_path'],
                        "fixes_applied": fixes_applied,
                        "compilation_log": compilation_log,
                        "iterations_used": iteration + 1
                    })
                
                # If failed and we have more iterations, try to auto-fix
                if iteration < max_fix_iterations:
                    compilation_log.append("Attempting auto-fix...")
                    fixed = self._auto_fix_common_errors(latex_file_path, compile_result['errors'])
                    if fixed:
                        fixes_applied.append(f"iteration_{iteration + 1}_auto_fix")
                        compilation_log.append("Auto-fix applied, retrying...")
                        continue
                    else:
                        compilation_log.append("Auto-fix failed, stopping iterations")
                        break
            
            # All attempts failed
            return json.dumps({
                "success": False,
                "error": "LaTeX compilation failed after all attempts",
                "pdf_path": None,
                "compilation_errors": compile_result.get('errors', []),
                "raw_latex_log": compile_result.get('raw_latex_log', ''),
                "fixes_applied": fixes_applied,
                "compilation_log": compilation_log
            })
        
        except FileNotFoundError as e:
            # Specific handling for file not found - provide clear feedback to agent
            return json.dumps({
                "success": False,
                "error": str(e),
                "error_type": "FileNotFoundError",
                "pdf_path": None,
                "suggestion": "Please check the LaTeX file path and ensure the file exists in the workspace."
            })
            
        except PermissionError as e:
            # Specific handling for permission errors - guide agent to use correct paths
            return json.dumps({
                "success": False,
                "error": str(e),
                "error_type": "PermissionError",
                "pdf_path": None,
                "suggestion": "Use relative paths within the workspace instead of absolute paths."
            })
            
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": f"LaTeX compilation failed: {str(e)}",
                "pdf_path": None
            })

    def _resolve_citations_automatically(self, latex_file_path: str) -> tuple[bool, List[str]]:
        """
        Automatically resolve [CITE:...] tokens in LaTeX file and all included files.

        This method:
        1. Processes the main LaTeX file and all \input{} files recursively
        2. Searches for all [CITE:description] patterns in each file
        3. For each pattern, checks if relevant citation exists in references.bib
        4. If not, uses citation search tool to find and add citations
        5. Replaces [CITE:description] with proper \cite{key} format
        6. Repeats until no [CITE:...] patterns remain

        Returns:
            (bool, List[str]): (whether changes were made, list of fixes applied)
        """
        try:
            # Find references.bib path
            references_bib_path = os.path.join(os.path.dirname(latex_file_path), 'references.bib')

            # Load existing citations from references.bib
            existing_citations = self._load_citation_keys(references_bib_path)

            # Get all LaTeX files to process (main file + all \input{} files)
            files_to_process = self._find_all_latex_files(latex_file_path)

            all_fixes_applied = []
            global_changes_made = False

            # Process each file for citation resolution
            for file_path in files_to_process:
                if not os.path.exists(file_path):
                    all_fixes_applied.append(f"skipped missing file: {file_path}")
                    continue

                file_changed, file_fixes = self._resolve_citations_in_file(
                    file_path, existing_citations, references_bib_path
                )

                if file_changed:
                    global_changes_made = True

                # Add file context to fixes
                rel_path = os.path.relpath(file_path, os.path.dirname(latex_file_path))
                for fix in file_fixes:
                    all_fixes_applied.append(f"{rel_path}: {fix}")

            return global_changes_made, all_fixes_applied

        except Exception as e:
            return False, [f"citation resolution error: {str(e)}"]

    def _find_all_latex_files(self, main_file_path: str) -> List[str]:
        """Find all LaTeX files that need citation processing (main + \input{} files)."""
        files_to_process = [main_file_path]
        main_dir = os.path.dirname(main_file_path)

        try:
            with open(main_file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Find all \input{filename} commands
            input_pattern = r'\\input\{([^}]+)\}'
            input_matches = re.findall(input_pattern, content)

            for input_file in input_matches:
                # Handle relative paths and add .tex extension if missing
                if not input_file.endswith('.tex'):
                    input_file += '.tex'

                if not os.path.isabs(input_file):
                    input_file_path = os.path.join(main_dir, input_file)
                else:
                    input_file_path = input_file

                if input_file_path not in files_to_process:
                    files_to_process.append(input_file_path)

        except Exception as e:
            print(f"Warning: Error scanning for \\input{{}} files: {e}")

        return files_to_process

    def _search_citation_with_retry(self, description: str, max_attempts: int = 5) -> Optional[Dict[str, Any]]:
        """
        Search for citation with retry logic for transient failures.

        Semantic Scholar API often returns 500 Internal Server Errors which are
        transient. This method retries with exponential backoff to handle these
        temporary failures, significantly improving citation resolution success rate.

        Args:
            description: Citation description to search for
            max_attempts: Maximum number of retry attempts (default: 5)

        Returns:
            Citation dict if found, None if all attempts fail
        """
        import time
        import json

        if not self.citation_search_tool:
            return None

        for attempt in range(max_attempts):
            try:
                search_result = self.citation_search_tool.forward(
                    search_query=description,
                    max_results=1,
                    search_source="both"
                )

                # Parse search result
                if search_result and '"success": true' in search_result.lower():
                    result_data = json.loads(search_result)
                    if result_data.get('citations'):
                        return result_data['citations'][0]

                # No results found - not a transient error, return None
                return None

            except Exception as e:
                error_str = str(e).lower()
                is_transient = (
                    '500' in error_str or
                    'internal server error' in error_str or
                    'timeout' in error_str or
                    'connection' in error_str or
                    '503' in error_str
                )

                if is_transient and attempt < max_attempts - 1:
                    # Exponential backoff: 2, 4, 8, 16, 32 seconds
                    wait_time = 2 ** (attempt + 1)
                    print(f"Warning: Citation search for '{description}' failed (attempt {attempt + 1}/{max_attempts}): {e}")
                    print(f"         Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    # Non-transient error or final attempt failed
                    if attempt == max_attempts - 1:
                        print(f"Warning: Citation search for '{description}' failed after {max_attempts} attempts: {e}")
                    return None

        return None

    def _resolve_citations_in_file(self, file_path: str, existing_citations: dict, references_bib_path: str) -> tuple[bool, List[str]]:
        """Resolve [CITE:...] tokens in a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content
            fixes_applied = []

            # Pattern to find [CITE:description] or [cite:description] tokens (case-insensitive)
            cite_pattern = r'\[(?:CITE|cite):([^\]]+)\]'
            max_iterations = 10  # Prevent infinite loops
            iteration = 0

            while iteration < max_iterations:
                matches = list(re.finditer(cite_pattern, content))
                if not matches:
                    break  # No more [CITE:...] tokens found

                iteration += 1
                changes_made = False

                # Process each match
                for match in reversed(matches):  # Process in reverse to maintain positions
                    description = match.group(1).strip()
                    full_token = match.group(0)

                    # Try to find existing citation
                    existing_key = self._find_best_citation_match(description, existing_citations)

                    if existing_key:
                        # Use existing citation
                        replacement = f"\\cite{{{existing_key}}}"
                        content = content[:match.start()] + replacement + content[match.end():]
                        fixes_applied.append(f"resolved '{description}' to existing citation '{existing_key}'")
                        changes_made = True
                    else:
                        # Search for new citation with retry logic for transient failures
                        citation = self._search_citation_with_retry(description, max_attempts=5)

                        if citation:
                            # Successfully found citation
                            # Generate unique citation key
                            new_key = self._generate_citation_key(citation, existing_citations)

                            # Add to references.bib
                            if self._add_citation_to_bib(references_bib_path, new_key, citation):
                                existing_citations[new_key] = citation
                                replacement = f"\\cite{{{new_key}}}"
                                content = content[:match.start()] + replacement + content[match.end():]
                                fixes_applied.append(f"found and added new citation '{new_key}' for '{description}'")
                                changes_made = True
                            else:
                                # Failed to add citation to bib file, remove token
                                content = content[:match.start()] + content[match.end():]
                                fixes_applied.append(f"removed citation token '{description}' (failed to add to bib)")
                                changes_made = True
                        else:
                            # No citation found after all retries, remove token
                            content = content[:match.start()] + content[match.end():]
                            fixes_applied.append(f"removed citation token '{description}' (not found after retries)")
                            changes_made = True

                if not changes_made:
                    break  # No changes in this iteration, stop

            # Write back if changes were made
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True, fixes_applied

            return False, []

        except Exception as e:
            return False, [f"citation resolution error: {str(e)}"]

    def _load_citation_keys(self, bib_path: str) -> Dict[str, Dict[str, str]]:
        """Load citation keys and metadata from references.bib file."""
        if not os.path.exists(bib_path):
            return {}

        citations = {}
        try:
            with open(bib_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Split by '@' to get individual entries
            bib_entries = content.split('@')[1:]  # Skip empty first element

            for entry in bib_entries:
                if not entry.strip():
                    continue

                # Extract citation key (first identifier after the type)
                key_match = re.search(r'^\w+\{([^,\s]+)', entry)
                if key_match:
                    key = key_match.group(1)

                    # Extract title for description matching
                    title_match = re.search(r'title\s*=\s*\{([^}]*)\}', entry, re.DOTALL)
                    title = title_match.group(1).strip() if title_match else ""

                    # Clean title (remove newlines, extra spaces)
                    title = ' '.join(title.split())

                    citations[key] = {
                        'title': title,
                        'entry': entry.strip()
                    }

        except Exception as e:
            print(f"Warning: Error loading citations from {bib_path}: {e}")

        return citations

    def _find_best_citation_match(self, description: str, citations: Dict[str, Dict[str, str]]) -> Optional[str]:
        """
        Find exact citation key match only.

        No fuzzy matching - descriptive queries like 'dropout_paper' should be
        resolved via Semantic Scholar search, not keyword matching which produces
        incorrect citations (e.g., matching 'deep_learning_review' to a wildfire paper).

        Only returns a match if the description exactly matches an existing citation key.
        """
        # Exact key match only (e.g., [CITE:sanchez2011] when sanchez2011 exists)
        if description in citations:
            return description

        # No fuzzy matching - let Semantic Scholar API handle descriptive queries
        return None

    def _generate_citation_key(self, citation: Dict[str, Any], existing_citations: Dict[str, Dict[str, str]]) -> str:
        """Generate a unique citation key for a new citation."""
        # Extract author and year for key generation
        title = citation.get('title', 'Unknown')
        authors = citation.get('authors', [])
        year = citation.get('year', 'unknown')

        # Use first author's last name if available
        if authors and isinstance(authors, list) and authors:
            first_author = authors[0]
            if isinstance(first_author, str):
                author_name = first_author.split()[-1].lower()  # Get last name
            else:
                author_name = str(first_author).split()[-1].lower()
        else:
            # Use first word of title
            author_name = title.split()[0].lower() if title.split() else 'unknown'

        # Clean author name (remove non-alphanumeric)
        author_name = re.sub(r'[^a-z0-9]', '', author_name)

        # Create base key
        base_key = f"{author_name}{year}"

        # Ensure uniqueness
        key = base_key
        counter = 1
        while key in existing_citations:
            key = f"{base_key}_{counter}"
            counter += 1

        return key

    def _add_citation_to_bib(self, bib_path: str, key: str, citation: Dict[str, Any]) -> bool:
        """Add a new citation to references.bib file."""
        try:
            # Format citation as BibTeX entry
            entry_type = citation.get('entry_type', 'article')
            title = citation.get('title', 'Unknown Title')
            authors = citation.get('authors', [])
            year = citation.get('year', 'unknown')
            venue = citation.get('venue', '')
            url = citation.get('url', '')

            # Format authors
            if isinstance(authors, list):
                author_str = ' and '.join(str(author) for author in authors)
            else:
                author_str = str(authors) if authors else 'Unknown'

            # Create BibTeX entry
            bib_entry = f"""
@{entry_type}{{{key},
    title = {{{title}}},
    author = {{{author_str}}},
    year = {{{year}}}"""

            if venue:
                if entry_type.lower() == 'article':
                    bib_entry += f",\n    journal = {{{venue}}}"
                else:
                    bib_entry += f",\n    booktitle = {{{venue}}}"

            if url:
                bib_entry += f",\n    url = {{{url}}}"

            bib_entry += "\n}\n"

            # Append to references.bib
            with open(bib_path, 'a', encoding='utf-8') as f:
                f.write(bib_entry)

            return True

        except Exception as e:
            print(f"Error adding citation to bib file: {e}")
            return False

    def _find_pdflatex_path(self) -> Optional[str]:
        """Adaptively find pdflatex executable path."""
        # Method 1: Use which command
        try:
            result = subprocess.run(['which', 'pdflatex'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                path = result.stdout.strip()
                if os.path.isfile(path) and os.access(path, os.X_OK):
                    return path
        except Exception:
            pass

        # Method 2: Check common installation paths
        common_paths = [
            '/usr/bin/pdflatex',
            '/usr/local/bin/pdflatex',
            '/home/tl784/texlive/2025/bin/x86_64-linux/pdflatex',
            '/home/tl784/texlive/latest/bin/x86_64-linux/pdflatex',
            '/gpfs/radev/home/tl784/texlive/latest/bin/x86_64-linux/pdflatex',
            '/gpfs/radev/home/tl784/texlive/2025/bin/x86_64-linux/pdflatex',
            '/opt/texlive/2025/bin/x86_64-linux/pdflatex',
            '/usr/local/texlive/2025/bin/x86_64-linux/pdflatex'
        ]

        for path in common_paths:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path

        # Method 3: Search PATH environment
        try:
            for path_dir in os.environ.get('PATH', '').split(os.pathsep):
                pdflatex_path = os.path.join(path_dir, 'pdflatex')
                if os.path.isfile(pdflatex_path) and os.access(pdflatex_path, os.X_OK):
                    return pdflatex_path
        except Exception:
            pass

        return None

    def _find_bibtex_path(self) -> Optional[str]:
        """Adaptively find bibtex executable path."""
        # Method 1: Use which command
        try:
            result = subprocess.run(['which', 'bibtex'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                path = result.stdout.strip()
                if os.path.isfile(path) and os.access(path, os.X_OK):
                    return path
        except Exception:
            pass

        # Method 2: Check common installation paths
        common_paths = [
            '/usr/bin/bibtex',
            '/usr/local/bin/bibtex',
            '/home/tl784/texlive/2025/bin/x86_64-linux/bibtex',
            '/home/tl784/texlive/latest/bin/x86_64-linux/bibtex',
            '/gpfs/radev/home/tl784/texlive/latest/bin/x86_64-linux/bibtex',
            '/gpfs/radev/home/tl784/texlive/2025/bin/x86_64-linux/bibtex',
            '/opt/texlive/2025/bin/x86_64-linux/bibtex',
            '/usr/local/texlive/2025/bin/x86_64-linux/bibtex'
        ]

        for path in common_paths:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path

        # Method 3: Search PATH environment
        try:
            for path_dir in os.environ.get('PATH', '').split(os.pathsep):
                bibtex_path = os.path.join(path_dir, 'bibtex')
                if os.path.isfile(bibtex_path) and os.access(bibtex_path, os.X_OK):
                    return bibtex_path
        except Exception:
            pass

        return None

    def _document_uses_bibliography(self, latex_file_path: str) -> bool:
        """Check if the LaTeX document uses bibliography commands."""
        try:
            with open(latex_file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check for bibliography-related commands
            bib_patterns = [
                r'\\bibliography\s*\{',
                r'\\bibliographystyle\s*\{',
                r'\\cite\s*\{',
                r'\\citep\s*\{',
                r'\\citet\s*\{',
                r'\\nocite\s*\{'
            ]

            import re
            for pattern in bib_patterns:
                if re.search(pattern, content):
                    return True

            return False
        except Exception:
            return False

    def _compile_latex_to_pdf(self, latex_file_path: str) -> Dict[str, Any]:
        """Compile LaTeX file to PDF using proper pdflatex → bibtex → pdflatex → pdflatex sequence."""

        tex_dir = os.path.dirname(os.path.abspath(latex_file_path))
        tex_filename = os.path.basename(latex_file_path)
        tex_basename = tex_filename.replace('.tex', '')
        pdf_filename = tex_filename.replace('.tex', '.pdf')
        pdf_path = os.path.join(tex_dir, pdf_filename)

        log = []
        errors = []
        raw_latex_output = []  # Collect full pdflatex output for debugging

        try:
            # Change to directory containing tex file
            original_dir = os.getcwd()
            os.chdir(tex_dir)

            # Find executables
            pdflatex_path = self._find_pdflatex_path()
            bibtex_path = self._find_bibtex_path()

            if not pdflatex_path:
                log.append("ERROR: pdflatex not found in system")
                return {
                    "success": False,
                    "pdf_path": None,
                    "errors": ["pdflatex executable not found on system"],
                    "log": log,
                    "raw_latex_log": ""
                }

            # Set environment to include PATH and other needed variables
            env = os.environ.copy()
            pdflatex_dir = os.path.dirname(pdflatex_path)
            env['PATH'] = f"{pdflatex_dir}:{env.get('PATH', '')}"

            # Check if document uses bibliography
            needs_bibtex = self._document_uses_bibliography(latex_file_path)

            if needs_bibtex and not bibtex_path:
                log.append("WARNING: Document uses bibliography but bibtex not found")
                needs_bibtex = False  # Continue without bibtex if not available

            # STEP 1: First pdflatex pass (creates .aux file)
            log.append(f"Step 1: Running first pdflatex pass on {tex_filename}")
            result1 = subprocess.run(
                [pdflatex_path, '-interaction=nonstopmode', tex_filename],
                capture_output=True,
                text=True,
                timeout=120,
                env=env
            )

            if result1.returncode != 0:
                log.append(f"First pdflatex pass failed with return code {result1.returncode}")
                errors.extend(self._parse_latex_errors(result1.stdout))
                errors.extend(self._parse_latex_errors(result1.stderr))
                # Capture full output for debugging
                raw_latex_output.append(f"=== First pdflatex pass stdout ===\n{result1.stdout}")
                raw_latex_output.append(f"=== First pdflatex pass stderr ===\n{result1.stderr}")
                return {
                    "success": False,
                    "pdf_path": None,
                    "errors": errors,
                    "log": log,
                    "raw_latex_log": "\n\n".join(raw_latex_output)
                }

            # STEP 2: Run bibtex if bibliography is used
            if needs_bibtex:
                log.append(f"Step 2: Running bibtex on {tex_basename}")
                result_bibtex = subprocess.run(
                    [bibtex_path, tex_basename],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    env=env
                )

                if result_bibtex.returncode != 0:
                    log.append(f"bibtex completed with warnings (return code {result_bibtex.returncode})")
                    if result_bibtex.stdout:
                        log.append(f"bibtex output: {result_bibtex.stdout}")
                    if result_bibtex.stderr:
                        log.append(f"bibtex errors: {result_bibtex.stderr}")
                else:
                    log.append("bibtex completed successfully")

                # STEP 3: Second pdflatex pass (incorporates bibliography)
                log.append(f"Step 3: Running second pdflatex pass to incorporate bibliography")
                result2 = subprocess.run(
                    [pdflatex_path, '-interaction=nonstopmode', tex_filename],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    env=env
                )

                if result2.returncode != 0:
                    log.append(f"Second pdflatex pass failed with return code {result2.returncode}")
                    errors.extend(self._parse_latex_errors(result2.stdout))
                    errors.extend(self._parse_latex_errors(result2.stderr))
                    # Capture full output for debugging
                    raw_latex_output.append(f"=== Second pdflatex pass stdout ===\n{result2.stdout}")
                    raw_latex_output.append(f"=== Second pdflatex pass stderr ===\n{result2.stderr}")
                    return {
                        "success": False,
                        "pdf_path": None,
                        "errors": errors,
                        "log": log,
                        "raw_latex_log": "\n\n".join(raw_latex_output)
                    }

                # STEP 4: Third pdflatex pass (resolves cross-references)
                log.append(f"Step 4: Running third pdflatex pass to resolve cross-references")
                result3 = subprocess.run(
                    [pdflatex_path, '-interaction=nonstopmode', tex_filename],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    env=env
                )

                if result3.returncode != 0:
                    log.append(f"Third pdflatex pass failed with return code {result3.returncode}")
                    errors.extend(self._parse_latex_errors(result3.stdout))
                    errors.extend(self._parse_latex_errors(result3.stderr))
                    # Capture full output for debugging
                    raw_latex_output.append(f"=== Third pdflatex pass stdout ===\n{result3.stdout}")
                    raw_latex_output.append(f"=== Third pdflatex pass stderr ===\n{result3.stderr}")
                    return {
                        "success": False,
                        "pdf_path": None,
                        "errors": errors,
                        "log": log,
                        "raw_latex_log": "\n\n".join(raw_latex_output)
                    }
            else:
                log.append("No bibliography detected, skipping bibtex and additional passes")

            # Check if PDF was created
            if os.path.exists(pdf_path):
                log.append(f"PDF successfully created: {pdf_path}")
                return {
                    "success": True,
                    "pdf_path": pdf_path,
                    "errors": [],
                    "log": log
                }
            else:
                log.append("PDF was not created despite successful compilation")
                return {
                    "success": False,
                    "pdf_path": None,
                    "errors": ["PDF file not generated"],
                    "log": log,
                    "raw_latex_log": "\n\n".join(raw_latex_output)
                }

        except subprocess.TimeoutExpired:
            log.append("LaTeX compilation timed out")
            return {
                "success": False,
                "pdf_path": None,
                "errors": ["Compilation timeout"],
                "log": log,
                "raw_latex_log": "\n\n".join(raw_latex_output)
            }
        except Exception as e:
            log.append(f"Compilation error: {str(e)}")
            return {
                "success": False,
                "pdf_path": None,
                "errors": [str(e)],
                "log": log,
                "raw_latex_log": "\n\n".join(raw_latex_output)
            }
        finally:
            os.chdir(original_dir)
    
    def _parse_latex_errors(self, output: str) -> List[str]:
        """Parse LaTeX compiler output for errors."""
        errors = []
        if not output:
            return errors
            
        lines = output.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('!'):
                # This is an error line
                error_msg = line.strip()
                # Try to get the next line for more context
                if i + 1 < len(lines):
                    error_msg += " " + lines[i + 1].strip()
                errors.append(error_msg)
        
        return errors
    
    def _clean_latex_artifacts(self, latex_file_path: str) -> tuple[bool, List[str]]:
        """Clean common LaTeX artifacts and formatting issues."""
        
        try:
            with open(latex_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            fixes = []
            
            # Fix 1: Remove line numbers (e.g., "1→\section{Introduction}")
            import re
            if re.search(r'^\s*\d+→', content, re.MULTILINE):
                content = re.sub(r'^\s*\d+→', '', content, flags=re.MULTILINE)
                fixes.append("removed line numbers")
            
            # Fix 2: Fix JSON content in tex files
            if content.strip().startswith('{') and '"latex_content"' in content:
                try:
                    import json
                    json_data = json.loads(content)
                    if 'latex_content' in json_data:
                        content = json_data['latex_content']
                        fixes.append("extracted LaTeX from JSON")
                except:
                    pass
            
            # Fix 3: Common package issues - use standard packages
            if '\\usepackage[nonatbib]{neurips_2024}' in content:
                content = content.replace(
                    '\\usepackage[nonatbib]{neurips_2024}',
                    '\\usepackage[margin=1in]{geometry}\\n\\usepackage{natbib}'
                )
                fixes.append("replaced neurips_2024 with standard packages")
            
            # Fix 4: Handle ICML style package
            if '\\usepackage{icml2024}' in content or '\\usepackage[accepted]{icml2024}' in content:
                # Ensure ICML style files are in the compilation directory
                icml_files = ['icml2024.sty', 'icml2024.bst', 'algorithm.sty', 'algorithmic.sty', 'fancyhdr.sty']
                source_dir = os.path.dirname(os.path.abspath(__file__))
                compile_dir = os.path.dirname(latex_file_path)
                
                for icml_file in icml_files:
                    source_path = os.path.join(source_dir, icml_file)
                    dest_path = os.path.join(compile_dir, icml_file)
                    if os.path.exists(source_path) and not os.path.exists(dest_path):
                        import shutil
                        shutil.copy2(source_path, dest_path)
                        fixes.append(f"copied {icml_file} to compilation directory")
            
            # Write back if changes were made
            if content != original_content:
                with open(latex_file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True, fixes
            
            return False, []
            
        except Exception as e:
            return False, [f"cleaning error: {str(e)}"]
    
    def _auto_fix_common_errors(self, latex_file_path: str, errors: List[str]) -> bool:
        """Auto-fix common LaTeX compilation errors."""
        
        if not errors:
            return False
            
        try:
            with open(latex_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Fix common error patterns
            for error in errors:
                error_lower = error.lower()
                
                # Missing $ in math mode
                if 'missing $' in error_lower:
                    # This is complex to fix automatically, skip for now
                    continue
                
                # Undefined control sequence
                if 'undefined control sequence' in error_lower:
                    # Add common packages that might be missing
                    if '\\documentclass' in content and '\\usepackage{amsmath}' not in content:
                        content = content.replace(
                            '\\documentclass',
                            '\\documentclass'
                        ).replace(
                            '\\begin{document}',
                            '\\usepackage{amsmath}\\n\\usepackage{amssymb}\\n\\begin{document}'
                        )
                
                # File not found errors
                if 'file not found' in error_lower or 'cannot find' in error_lower:
                    # Remove problematic includes
                    if '\\input{' in content:
                        # For now, just log this - more complex fix needed
                        pass
            
            # Write back if changes made
            if content != original_content:
                with open(latex_file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True
            
            return False
            
        except Exception:
            return False
    
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
                    f"Example: Use 'paper_workspace/final_paper.tex' instead of the full path."
                )
        else:
            # Relative path - join with workspace
            abs_path = os.path.abspath(os.path.join(abs_working_dir, path))
            
            # For compiler, we need to check if file exists for reading
            # Don't check for write operations (PDF output)
            if path.endswith('.tex') and not os.path.exists(abs_path):
                # Provide helpful error for agent
                parent_dir = os.path.dirname(abs_path)
                if os.path.exists(parent_dir):
                    raise FileNotFoundError(
                        f"LaTeX file not found: '{path}' does not exist in the workspace. "
                        f"The parent directory exists. Please check the filename."
                    )
                else:
                    raise FileNotFoundError(
                        f"LaTeX file not found: '{path}' does not exist in the workspace. "
                        f"The directory '{os.path.dirname(path)}' was not found."
                    )
            
            return abs_path