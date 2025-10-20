"""
LaTeXGeneratorTool - Generate academic LaTeX content from structured inputs.

This tool is the core content generation engine of the WriteupAgent.
It receives clean, structured inputs from the WriteupAgent and generates
publication-quality LaTeX sections, writing them directly to files.

Key changes:
- Tool now writes LaTeX content directly to files
- Returns file path and status instead of JSON with content
- Eliminates JSON parsing issues in WriteupAgent

The WriteupAgent is responsible for:
- Reading experiment results and data
- Running supporting tools (citations, figures, etc.)
- Structuring inputs for this tool
- Orchestrating file generation

This tool focuses on high-quality academic writing generation and file management.
"""

import json
import os
from typing import Dict, Any, List, Optional
from smolagents import Tool, ChatMessage

# No need to import LLM functions - model is passed to constructor


class LaTeXGeneratorTool(Tool):
    name = "latex_generator_tool"
    description = """
    Generate high-quality LaTeX content for academic papers and writes files directly to workspace.
    
    This is the core content generation engine that transforms structured research data
    into publication-ready LaTeX sections. This tool writes LaTeX content directly to files,
    eliminating JSON parsing issues by handling all file operations internally.
    
    Key capabilities:
    - Generate paper sections: Abstract, Introduction, Methods, Results, Discussion
    - Transform experimental data into clear scientific narrative
    - Integrate citations and figure references seamlessly
    - Follow academic writing conventions and journal standards
    - Writes files directly to workspace (e.g., abstract.tex, introduction.tex)
    
    Output behavior:
    - Individual sections: Writes content directly to {section_type}.tex in paper_workspace/ subdirectory
    - Main document: Writes final_paper.tex with \\input{} commands to include individual sections
    - Returns JSON with file path and generation status (no content parsing needed)

    ICML template structure (used by default if icml2024.sty exists):
    ```latex
    \\documentclass{article}
    \\usepackage{icml2024}
    \\usepackage{graphicx}
    \\usepackage{amsmath}

    \\icmltitlerunning{Short Title}
    \\begin{document}
    \\twocolumn[
    \\icmltitle{Full Paper Title}
    \\begin{icmlauthorlist}
    \\icmlauthor{AI Scientist}{institution}
    \\end{icmlauthorlist}
    \\icmlaffiliation{institution}{Institution Name}
    \\icmlcorrespondingauthor{AI Scientist}{email@example.com}
    \\icmlkeywords{keywords, separated, by, commas}
    \\vskip 0.3in
    ]
    \\input{abstract}
    \\input{introduction}
    ...
    \\bibliography{references}
    \\bibliographystyle{icml2024}
    \\end{document}
    ```

    The WriteupAgent should prepare all inputs - this tool handles generation and file writing.
    """
    
    inputs = {
        "section_type": {
            "type": "string",
            "description": "LaTeX section type to generate:\n" +
                          "• 'abstract': Concise summary of research problem, approach, and key findings\n" +
                          "• 'introduction': Problem motivation, background, and contributions\n" +
                          "• 'methods': Technical approach, experimental setup, and implementation details\n" +
                          "• 'results': Experimental findings, data analysis, and quantitative results\n" +
                          "• 'discussion': Interpretation of results, implications, and limitations\n" +
                          "• 'conclusion': Summary of contributions and future work\n" +
                          "• 'main_document': Complete LaTeX document with \\input{} commands for all sections"
        },
        "content_description": {
            "type": "string", 
            "description": "Natural language description of the content to generate. Include: research problem/idea, experimental setup, key findings/results, methodology details, figure descriptions, baseline comparisons, statistical significance, limitations, and any other relevant information. Be specific about numerical results, experimental conditions, and concrete findings."
        },
        "output_dir": {
            "type": "string",
            "description": "Output directory for the generated .tex file (default: 'paper_workspace')",
            "nullable": True
        },
        "writing_style": {
            "type": "string",
            "description": "Academic writing style (default: 'technical'):\n" +
                          "• 'concise': Brief, dense writing for space-constrained venues (workshop papers, letters)\n" +
                          "• 'detailed': Comprehensive explanations with thorough background (journal papers)\n" +
                          "• 'technical': Standard academic style with precise technical language (conference papers)\n" +
                          "• 'accessible': Clear explanations suitable for broader scientific audience",
            "nullable": True
        },
        "target_venue": {
            "type": "string", 
            "description": "Target publication venue (e.g., 'NeurIPS', 'ICML', 'Nature', 'Science') to match style conventions",
            "nullable": True
        }
    }
    
    output_type = "string"

    def __init__(self, model=None, working_dir=None):
        """
        Initialize LaTeXGeneratorTool.
        
        Args:
            model: The LLM model instance to use for generation
            working_dir: Base working directory for file operations
        """
        super().__init__()
        from ..model_utils import get_raw_model
        self.model = get_raw_model(model)
        # Convert to absolute path to prevent nested directory issues
        self.working_dir = os.path.abspath(working_dir or os.getcwd())
        # Load available citations from references.bib
        self.available_citations = self._load_citations()
    
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
                    f"Example: Use 'paper_workspace/output.tex' instead of the full path."
                )
        else:
            # Relative path - join with workspace and create directories if needed
            abs_path = os.path.abspath(os.path.join(abs_working_dir, path))
            
            # Create parent directory if it doesn't exist (LaTeXGenerator creates files)
            parent_dir = os.path.dirname(abs_path)
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)
            
            return abs_path
    
    def forward(self, section_type: str, content_description: str, output_dir: str = "paper_workspace", 
                writing_style: str = "technical", target_venue: str = None) -> str:
        """
        Generate LaTeX content for academic papers and write to file.
        
        Args:
            section_type: Type of section to generate
            content_description: Natural language description of research content
            output_dir: Directory to write the output file (relative to working_dir)
            writing_style: Academic writing style preference  
            target_venue: Target publication venue for style matching
            
        Returns:
            JSON string with file path and generation status
        """
        try:
            # Use content description directly
            content_info = content_description.strip()
            
            # Check if model is available
            if not self.model:
                return json.dumps({
                    "status": "error",
                    "error": "No LLM model provided to LaTeXGeneratorTool",
                    "file_path": None
                })
            
            # Determine output file path
            output_dir_path = os.path.join(self.working_dir, output_dir)
            os.makedirs(output_dir_path, exist_ok=True)
            
            if section_type == "main_document":
                filename = "final_paper.tex"
            else:
                filename = f"{section_type}.tex"
            
            file_path = os.path.join(output_dir_path, filename)
            
            # Generate section-specific content
            if section_type == "main_document":
                latex_content = self._generate_main_document(self.model, content_info, writing_style, target_venue)
            else:
                latex_content = self._generate_section(self.model, section_type, content_info, writing_style, target_venue)
            
            # Write to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(latex_content)
            
            # Return success status with file path
            result = {
                "status": "success",
                "section_type": section_type,
                "file_path": os.path.relpath(file_path, self.working_dir),
                "absolute_path": file_path,
                "writing_style": writing_style,
                "target_venue": target_venue,
                "content_length": len(latex_content),
                "message": f"LaTeX content successfully written to {filename}"
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            error_result = {
                "status": "error",
                "error": f"LaTeX generation failed: {str(e)}",
                "section_type": section_type,
                "file_path": None
            }
            return json.dumps(error_result, indent=2)
    
    def _generate_section(self, model, section_type: str, content_description: str, 
                         writing_style: str, target_venue: Optional[str]) -> str:
        """Generate a specific section of the paper."""
        
        # Build section-specific prompt
        system_prompt = self._get_system_prompt(section_type, writing_style, target_venue)
        user_prompt = self._build_section_prompt(section_type, content_description)
        
        # Generate content using LLM with proper ChatMessage format
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        messages = [ChatMessage(role="user", content=full_prompt)]
        response = model.generate(messages).content
        
        # Clean and format the response
        return self._clean_latex_output(response)
    
    def _generate_main_document(self, model, content_description: str, 
                              writing_style: str, target_venue: Optional[str]) -> str:
        """Generate a main document skeleton with \\input{} commands to include individual sections."""
        
        # Extract title from content_description if provided
        paper_title = None
        import re
        title_match = re.search(r"[Tt]itle\s+(?:of\s+the\s+paper\s+)?is\s+['\"]([^'\"]+)['\"]|[Tt]itle:\s*['\"]([^'\"]+)['\"]|[Pp]aper\s+[Tt]itle:\s*['\"]([^'\"]+)['\"]", content_description)
        if title_match:
            paper_title = title_match.group(1) or title_match.group(2) or title_match.group(3)
        
        # Generate document preamble
        preamble = self._generate_document_preamble(target_venue, paper_title)
        
        # Define standard academic paper sections
        sections = ["abstract", "introduction", "methods", "results", "discussion", "conclusion"]
        
        # Build the main document with \input{} commands
        main_content = [preamble]
        main_content.append("")  # Empty line after preamble
        
        # Add input commands for each section (flat structure within paper_workspace/)
        for section in sections:
            if section == "abstract":
                # Abstract is typically included directly, not as a separate input
                main_content.append("\\input{abstract}")
            else:
                main_content.append(f"\\input{{{section}}}")
        
        # Add bibliography and document ending
        main_content.append("")
        main_content.append("\\bibliography{references}")
        main_content.append("\\bibliographystyle{plain}")
        main_content.append("")
        main_content.append("\\end{document}")
        
        return "\n".join(main_content)
    
    def _get_system_prompt(self, section_type: str, writing_style: str, target_venue: Optional[str]) -> str:
        """Build system prompt for the specific section type."""
        
        # Scan for available figures and their descriptions
        figures_info = self._scan_figures_directory()
        figures_context = self._format_figures_context(figures_info)

        # Get available citations context
        citations_context = self._format_citations_context()

        base_system = f"""You are an expert academic writer specializing in AI/ML research papers.
        Generate high-quality LaTeX content with proper scientific style and formatting.

        Writing style: {writing_style}
        Target venue: {target_venue or 'General academic conference'}

        {figures_context}

        CRITICAL CITATION INSTRUCTIONS:
        {citations_context}

        - NEVER use [? ] or [?] placeholders for citations
        - ALWAYS use proper \\cite{{key}} format with actual citation keys from the list above
        - If you need a citation but don't find a perfect match, use the closest relevant one
        - If no relevant citations exist, use [CITE:brief_description] format for the WriteupAgent to resolve later

        CRITICAL FORMATTING RULES:
        - NEVER use \\subref{{}} command - use \\ref{{}} instead for subfigure references
        - NEVER mix math and text modes incorrectly (avoid "Missing $ inserted" errors)
        - ALWAYS properly close environments (equations, itemize, enumerate)
        - AVOID orphaned \\item commands outside list environments
        - Use \\eqref{{}} for equation references, \\ref{{}} for figures/tables/sections
        - Ensure balanced braces {{}} in all LaTeX commands

        🚨 CRITICAL DATA USAGE RULES - ABSOLUTE REQUIREMENTS 🚨:

        **YOU HAVE NO ACCESS TO EXPERIMENTAL DATA FILES**:
        - You CANNOT read baseline_summary.json, research_summary.json, or ablation_summary.json
        - You CANNOT access experimental result files or logs
        - You CANNOT infer or assume any numerical values
        - You MUST ONLY use data explicitly provided in the content_description parameter

        **MANDATORY DATA FIDELITY**:
        - ✅ USE EXACT numerical values from content_description (e.g., "F1 = 0.637")
        - ✅ USE EXACT hyperparameters from content_description (e.g., "learning rate = 1e-4")
        - ✅ USE EXACT dataset names from content_description (e.g., "IMDB", "SST-2")
        - ✅ USE EXACT model specifications from content_description (e.g., "4 hidden states")
        - ❌ NEVER fabricate, estimate, or hallucinate numerical values
        - ❌ NEVER use placeholder values like "X%", "N epochs", "good performance"
        - ❌ NEVER make up experimental results not provided

        **IF DATA IS MISSING FROM content_description**:
        - DO NOT fabricate the missing data
        - DO NOT use generic placeholders
        - Instead, write: "\\textbf{{[DATA REQUIRED: specific_metric_name]}}"
        - Example: If F1 score not provided, write "\\textbf{{[DATA REQUIRED: F1 score for IMDB dataset]}}"

        **VERIFICATION CHECKLIST** (apply to every sentence):
        1. Does this numerical value appear in content_description? → YES = use it, NO = [DATA REQUIRED]
        2. Does this metric appear in content_description? → YES = use it, NO = [DATA REQUIRED]
        3. Does this experimental detail appear in content_description? → YES = use it, NO = [DATA REQUIRED]

        **EXAMPLES**:
        - ❌ WRONG: "Our method achieves good performance" (vague, no data)
        - ❌ WRONG: "F1 score improved to approximately 0.6" (approximation, not provided)
        - ❌ WRONG: "We trained for 100 epochs" (if not in content_description)
        - ✅ CORRECT: "Horizon-500 Transition F1 = 0.637 on IMDB dataset" (exact from content_description)
        - ✅ CORRECT: "BIC selected 4 hidden states" (exact from content_description)
        - ✅ CORRECT: "Performance improved from 0.0 to 0.637" (both values from content_description)

        Requirements:
        - Use proper LaTeX formatting and commands
        - Follow academic writing conventions
        - **MANDATORY**: Integrate citations using [cite: key] placeholder format (NOT \\cite{{}} directly)
          Example: [cite: rabiner1989tutorial] or [cite: Hidden Markov Models tutorial]
          LaTeXCompilerTool will auto-resolve these placeholders
        - Reference figures using \\ref{{fig:label}} format and include them with \\includegraphics{{figures/filename}}
        - Use ONLY the figures listed above with their exact filenames (including .png extension)
        - Figure paths are relative to the LaTeX document: figures/filename.png
        - Place figures in appropriate locations within the text flow using figure environments
        - Use clear, precise scientific language
        - Maintain logical flow and coherence
        - Include appropriate technical depth for the venue
        - **CRITICALLY: Use ONLY data explicitly provided in content_description - NO fabrication**
        """
        
        section_specific = {
            "abstract": """
            Generate a compelling abstract (150-250 words) that:
            - Clearly states the problem and motivation
            - Summarizes the key methodology and contributions
            - Highlights main results and their significance
            - Uses concise, impactful language
            """,
            
            "introduction": """
            Generate a comprehensive introduction that:
            - Motivates the research problem with clear context
            - Reviews relevant literature and identifies gaps
            - Clearly states contributions and novelty
            - Outlines the paper structure
            - Establishes the significance of the work
            """,
            
            "methods": """
            Generate a detailed methods section that:
            - Describes the methodology clearly and precisely
            - Provides sufficient detail for reproducibility
            - Explains algorithmic approaches and architectures
            - Justifies design choices and hyperparameters
            - Includes experimental setup and evaluation metrics
            """,
            
            "results": """
            Generate a results section that:
            - Presents experimental findings clearly and objectively
            - Includes quantitative results with statistical significance
            - References figures and tables appropriately
            - Compares with baselines and related work
            - Highlights key insights and patterns
            """,
            
            "discussion": """
            Generate a discussion section that:
            - Interprets results in the context of the research questions
            - Analyzes strengths and limitations honestly
            - Compares with related work and explains differences
            - Discusses broader implications and significance
            - Suggests future research directions
            """,
            
            "conclusion": """
            Generate a conclusion that:
            - Summarizes key contributions and findings
            - Reinforces the significance of the work
            - Acknowledges limitations appropriately
            - Suggests concrete future work directions
            - Ends with impact statement
            """
        }
        
        return base_system + "\n" + section_specific.get(section_type, "Generate appropriate academic content.")
    
    def _build_section_prompt(self, section_type: str, content_description: str) -> str:
        """Build the user prompt with natural language content description."""
        
        prompt = f"""Generate a high-quality {section_type} section for an academic paper based on the following research content:

{content_description}

Requirements:
- Generate publication-quality LaTeX content
- Use proper LaTeX formatting and commands
- Include appropriate citations using \\cite{{key}} format when relevant
- Reference figures using \\ref{{fig:label}} format when mentioned
- Base ALL content on the provided research information
- Do NOT fabricate or hallucinate information not provided
- Use specific numerical results and experimental details from the description
- Maintain academic writing standards for the {section_type} section

Generate only the LaTeX content for this section."""
        
        return prompt
    
    def _generate_document_preamble(self, target_venue: Optional[str], paper_title: Optional[str] = None) -> str:
        """Generate document preamble based on target venue."""

        # Use provided title or default
        title = paper_title if paper_title else "Research Paper Title"

        # Use ICML template by default if available (ResourcePreparationAgent copies it automatically)
        icml_style_path = os.path.join(self.working_dir, "paper_workspace", "icml2024.sty") if self.working_dir else None
        use_icml = icml_style_path and os.path.exists(icml_style_path)

        if use_icml:
            # ICML conference style with proper template
            return f"""\\documentclass{{article}}

% ICML 2024 packages and style
\\usepackage{{microtype}}
\\usepackage{{graphicx}}
\\usepackage{{subfigure}}
\\usepackage{{booktabs}}
\\usepackage{{hyperref}}

% Use the ICML 2024 style (comment out for blind review)
\\usepackage[accepted]{{icml2024}}

% Math packages
\\usepackage{{amsmath}}
\\usepackage{{amssymb}}
\\usepackage{{mathtools}}
\\usepackage{{amsthm}}

% Algorithm packages
\\usepackage{{algorithm}}
\\usepackage{{algorithmic}}

% Make hyperref and algorithmic work together
\\newcommand{{\\theHalgorithm}}{{\\arabic{{algorithm}}}}

% Running title
\\icmltitlerunning{{{title}}}

\\begin{{document}}

\\twocolumn[
\\icmltitle{{{title}}}

\\begin{{icmlauthorlist}}
\\icmlauthor{{Author Names}}{{inst1}}
\\end{{icmlauthorlist}}

\\icmlaffiliation{{inst1}}{{Institution/Company, Location}}

\\icmlcorrespondingauthor{{Author Name}}{{email@example.com}}

\\icmlkeywords{{Machine Learning, ICML}}

\\vskip 0.3in
]

\\printAffiliationsAndNotice{{}}"""
        elif target_venue and target_venue.lower() in ["neurips", "iclr"]:
            # Other ML conference styles
            return f"""\\documentclass{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage[T1]{{fontenc}}
\\usepackage{{amsmath,amsfonts,amssymb}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{algorithm}}
\\usepackage{{algpseudocode}}
\\usepackage{{hyperref}}
\\usepackage{{natbib}}

\\title{{{title}}}
\\author{{Author Names}}
\\date{{}}

\\begin{{document}}
\\maketitle"""
        else:
            # General academic style
            return f"""\\documentclass{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{amsmath,amsfonts,amssymb}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{hyperref}}
\\usepackage{{natbib}}

\\title{{{title}}}
\\author{{Author Names}}
\\date{{}}

\\begin{{document}}
\\maketitle"""
    
    def _clean_latex_output(self, raw_output: str) -> str:
        """Clean and format the LaTeX output."""
        
        # Remove any markdown formatting that might have been added
        content = raw_output.strip()
        content = content.replace("```latex", "").replace("```", "").strip()
        
        # Ensure proper spacing around section headers
        content = content.replace("\\section{", "\n\\section{")
        content = content.replace("\\subsection{", "\n\\subsection{")
        content = content.replace("\\subsubsection{", "\n\\subsubsection{")
        
        # Clean up excessive whitespace
        lines = content.split('\n')
        cleaned_lines = []
        for line in lines:
            cleaned_line = line.strip()
            if cleaned_line:
                cleaned_lines.append(cleaned_line)
            elif cleaned_lines and cleaned_lines[-1]:  # Add single blank line
                cleaned_lines.append("")
        
        return '\n'.join(cleaned_lines)
    
    def _scan_figures_directory(self) -> Dict[str, str]:
        """
        Scan the figures directory for .png files and their corresponding .txt descriptions.
        
        Returns:
            Dictionary mapping figure filename to description text
        """
        figures_info = {}
        
        if not self.working_dir:
            return figures_info
            
        figures_dir = os.path.join(self.working_dir, "paper_workspace", "figures")
        if not os.path.exists(figures_dir):
            return figures_info
            
        try:
            # Find all .png files
            for filename in os.listdir(figures_dir):
                if filename.endswith('.png'):
                    png_path = os.path.join(figures_dir, filename)
                    
                    # Look for corresponding .txt description file
                    txt_filename = filename.replace('.png', '.txt')
                    txt_path = os.path.join(figures_dir, txt_filename)
                    
                    if os.path.exists(txt_path):
                        try:
                            with open(txt_path, 'r', encoding='utf-8') as f:
                                description = f.read().strip()
                            figures_info[filename] = description
                        except Exception as e:
                            print(f"Warning: Could not read description file {txt_path}: {e}")
                            figures_info[filename] = "Figure description not available"
                    else:
                        figures_info[filename] = "Figure description not available"
                        
        except Exception as e:
            print(f"Warning: Could not scan figures directory {figures_dir}: {e}")
            
        return figures_info
    
    def _format_figures_context(self, figures_info: Dict[str, str]) -> str:
        """
        Format the figures information for inclusion in the system prompt.
        
        Args:
            figures_info: Dictionary mapping figure filename to description
            
        Returns:
            Formatted string for the system prompt
        """
        if not figures_info:
            return "AVAILABLE FIGURES: None - no figures found in figures/ directory."
            
        context_lines = ["AVAILABLE FIGURES:"]
        for filename, description in figures_info.items():
            context_lines.append(f"- {filename}: {description}")
            
        context_lines.append("")
        context_lines.append("IMPORTANT: Use these exact filenames when creating \\includegraphics{{figures/filename}} commands.")
        
        return "\n".join(context_lines)

    def _load_citations(self) -> Dict[str, str]:
        """Load and parse available citations from references.bib file."""
        citations = {}

        if not self.working_dir:
            return citations

        # Look for references.bib in paper_workspace directory
        bib_paths = [
            os.path.join(self.working_dir, "paper_workspace", "references.bib"),
            os.path.join(self.working_dir, "references.bib")
        ]

        for bib_path in bib_paths:
            if os.path.exists(bib_path):
                try:
                    with open(bib_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Parse BibTeX entries by splitting on @ symbols
                    import re
                    bib_entries = content.split('@')[1:]  # Skip empty first element

                    for entry in bib_entries:
                        # Extract citation key
                        key_match = re.search(r'^(\w+)\{([^,]+),', entry)
                        if not key_match:
                            continue

                        entry_type, cite_key = key_match.groups()
                        cite_key = cite_key.strip()

                        # Extract title
                        title_match = re.search(r'title\s*=\s*\{([^}]*)\}', entry, re.DOTALL)
                        if title_match:
                            title = title_match.group(1).strip()
                            # Clean up multi-line formatting
                            title = ' '.join(title.split())
                        else:
                            title = "Unknown title"

                        citations[cite_key] = title

                    break  # Use first found file

                except Exception as e:
                    print(f"Warning: Could not parse {bib_path}: {e}")
                    continue

        return citations

    def _format_citations_context(self) -> str:
        """Format available citations for inclusion in LLM prompts."""
        if not self.available_citations:
            return "No citations available in references.bib"

        context = "Available citations (use \\cite{key} format):\n"
        for cite_key, title in self.available_citations.items():
            context += f"- {cite_key}: {title}\n"

        return context

