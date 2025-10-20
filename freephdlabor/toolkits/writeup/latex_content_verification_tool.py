"""
LaTeXContentVerificationTool - Verify LaTeX content meets success criteria.

This tool provides comprehensive content verification including:
- Section character count analysis
- Content quality assessment  
- Success criteria validation
- Structural completeness check

Essential for ensuring papers meet submission requirements.
"""

import json
import os
import re
from typing import Dict, Any, Optional, List
from smolagents import Tool


class LaTeXContentVerificationTool(Tool):
    name = "latex_content_verification_tool"
    description = """
    Verify LaTeX content meets all success criteria and quality standards.
    
    This tool is essential for:
    - Validating content length requirements (>15K characters)
    - Checking section completeness and quality
    - Ensuring substantial content in each major section
    - Verifying file deliverables exist
    - Assessing overall paper quality
    
    Use this tool:
    - Before calling final_answer() to verify all criteria are met
    - To check if content is substantial enough for submission
    - To identify which sections need more content
    - To ensure all required files exist
    
    The tool performs comprehensive analysis to ensure papers meet
    academic publication standards and success criteria.
    
    Input: LaTeX file path
    Output: Detailed verification results with pass/fail status
    """
    
    inputs = {
        "latex_file_path": {
            "type": "string", 
            "description": "Path to the LaTeX file to verify (e.g., 'final_paper.tex')"
        }
    }
    
    outputs = {
        "verification_result": {
            "type": "string",
            "description": "Comprehensive verification results with success criteria status"
        }
    }
    
    output_type = "string"

    def __init__(self, working_dir: Optional[str] = None):
        """Initialize LaTeXContentVerificationTool.
        
        Args:
            working_dir: Working directory for workspace-aware file access
        """
        super().__init__()
        # Convert to absolute path to prevent nested directory issues
        self.working_dir = os.path.abspath(working_dir) if working_dir else None
        
    def forward(self, latex_file_path: str) -> str:
        """
        Verify LaTeX content meets all success criteria.
        
        Args:
            latex_file_path: Path to the LaTeX file to verify
            
        Returns:
            JSON string containing comprehensive verification results
        """
        try:
            # Resolve file path (workspace-aware)
            file_path = self._safe_path(latex_file_path.strip()) if self.working_dir else latex_file_path.strip()
            
            # Note: _safe_path now checks existence, so this is redundant but kept for safety
            if not os.path.exists(file_path):
                return json.dumps({
                    "success": False,
                    "error": f"LaTeX file not found: {file_path}",
                    "criteria_met": False
                }, indent=2)
            
            # Read LaTeX content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Perform verification checks
            file_checks = self._check_required_files()
            section_analysis = self._analyze_sections(content)
            content_quality = self._assess_content_quality(content, section_analysis)
            overall_assessment = self._overall_assessment(file_checks, section_analysis, content_quality)
            
            result = {
                "file_path": file_path,
                "verification_timestamp": "completed",
                "success": overall_assessment["all_criteria_met"],
                "overall_assessment": overall_assessment,
                "file_checks": file_checks,
                "section_analysis": section_analysis,
                "content_quality": content_quality,
                "recommendations": self._generate_recommendations(file_checks, section_analysis, content_quality)
            }
            
            return json.dumps(result, indent=2)
        
        except FileNotFoundError as e:
            # Specific handling for file not found - provide clear feedback to agent
            return json.dumps({
                "success": False,
                "error": str(e),
                "error_type": "FileNotFoundError",
                "criteria_met": False,
                "suggestion": "Please check the file path and ensure the file exists in the workspace."
            }, indent=2)
            
        except PermissionError as e:
            # Specific handling for permission errors - guide agent to use correct paths
            return json.dumps({
                "success": False,
                "error": str(e),
                "error_type": "PermissionError",
                "criteria_met": False,
                "suggestion": "Use relative paths within the workspace instead of absolute paths."
            }, indent=2)
            
        except Exception as e:
            error_result = {
                "success": False,
                "error": f"Verification failed: {str(e)}",
                "criteria_met": False
            }
            return json.dumps(error_result, indent=2)
    
    def _check_required_files(self) -> Dict[str, Any]:
        """Check if all required deliverable files exist."""
        checks = {}
        base_dir = self.working_dir if self.working_dir else "."
        
        # Check final_paper.tex
        tex_path = os.path.join(base_dir, "final_paper.tex")
        checks["tex_exists"] = os.path.exists(tex_path)
        checks["tex_size"] = os.path.getsize(tex_path) if checks["tex_exists"] else 0
        
        # Check final_paper.pdf  
        pdf_path = os.path.join(base_dir, "final_paper.pdf")
        checks["pdf_exists"] = os.path.exists(pdf_path)
        checks["pdf_size"] = os.path.getsize(pdf_path) if checks["pdf_exists"] else 0
        
        # Check references.bib
        bib_path = os.path.join(base_dir, "references.bib")
        checks["bib_exists"] = os.path.exists(bib_path)
        checks["bib_size"] = os.path.getsize(bib_path) if checks["bib_exists"] else 0
        
        return checks
    
    def _analyze_sections(self, content: str) -> Dict[str, Any]:
        """Analyze section content and character counts."""
        sections = {}
        
        # Define section patterns
        section_patterns = {
            "abstract": (r'\\begin{abstract}(.*?)\\end{abstract}', re.DOTALL),
            "introduction": (r'\\section\*?{Introduction}(.*?)(?=\\section|\Z)', re.DOTALL | re.IGNORECASE),
            "methodology": (r'\\section\*?{(?:Method[s|ology]|Approach)}(.*?)(?=\\section|\Z)', re.DOTALL | re.IGNORECASE),
            "results": (r'\\section\*?{Results}(.*?)(?=\\section|\Z)', re.DOTALL | re.IGNORECASE),
            "discussion": (r'\\section\*?{(?:Discussion|Conclusion)}(.*?)(?=\\section|\Z)', re.DOTALL | re.IGNORECASE),
            "related_work": (r'\\section\*?{Related Work}(.*?)(?=\\section|\Z)', re.DOTALL | re.IGNORECASE)
        }
        
        total_content_chars = 0
        
        for section_name, (pattern, flags) in section_patterns.items():
            match = re.search(pattern, content, flags)
            if match:
                section_content = match.group(1).strip()
                # Remove LaTeX commands for more accurate character count
                clean_content = self._clean_latex_content(section_content)
                char_count = len(clean_content)
                
                sections[section_name] = {
                    "found": True,
                    "raw_chars": len(section_content),
                    "content_chars": char_count,
                    "has_substantial_content": char_count > 200,  # Minimum threshold per section
                    "quality_score": self._assess_section_quality(clean_content)
                }
                total_content_chars += char_count
            else:
                sections[section_name] = {
                    "found": False,
                    "raw_chars": 0,
                    "content_chars": 0,
                    "has_substantial_content": False,
                    "quality_score": 0
                }
        
        sections["total_content_chars"] = total_content_chars
        sections["meets_length_requirement"] = total_content_chars > 15000
        
        return sections
    
    def _clean_latex_content(self, content: str) -> str:
        """Remove LaTeX commands and markup to get clean text."""
        # Remove comments
        content = re.sub(r'%.*$', '', content, flags=re.MULTILINE)
        
        # Remove common LaTeX commands but keep their content
        content = re.sub(r'\\(?:textbf|textit|emph|texttt|underline)\{([^}]*)\}', r'\1', content)
        content = re.sub(r'\\(?:section|subsection|subsubsection)\*?\{([^}]*)\}', r'\1', content)
        
        # Remove citations, references, labels
        content = re.sub(r'\\(?:cite|ref|label)\{[^}]*\}', '', content)
        
        # Remove figure/table environments but keep captions
        content = re.sub(r'\\begin\{(?:figure|table)\}.*?\\end\{(?:figure|table)\}', '', content, flags=re.DOTALL)
        
        # Remove other commands
        content = re.sub(r'\\[a-zA-Z]+(?:\[[^\]]*\])?(?:\{[^}]*\})*', '', content)
        
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content).strip()
        
        return content
    
    def _assess_section_quality(self, clean_content: str) -> float:
        """Assess the quality of section content (0-1 score)."""
        if not clean_content:
            return 0.0
        
        score = 0.0
        
        # Length factor (more content = higher score, up to a point)
        length_score = min(len(clean_content) / 1000, 1.0)  # 1000 chars = full score
        score += length_score * 0.4
        
        # Sentence structure (basic check for complete sentences)
        sentences = [s.strip() for s in clean_content.split('.') if s.strip()]
        if len(sentences) >= 3:
            score += 0.3
        
        # Vocabulary diversity (unique words vs total words)
        words = clean_content.lower().split()
        if words:
            unique_ratio = len(set(words)) / len(words)
            score += unique_ratio * 0.3
        
        return min(score, 1.0)
    
    def _assess_content_quality(self, content: str, section_analysis: Dict) -> Dict[str, Any]:
        """Assess overall content quality."""
        quality = {}
        
        # Check for placeholder title
        title_match = re.search(r'\\title\{([^}]+)\}', content)
        if title_match:
            title = title_match.group(1).strip()
            quality["has_title"] = True
            quality["title"] = title
            quality["has_placeholder_title"] = title == "Research Paper Title"
        else:
            quality["has_title"] = False
            quality["title"] = None
            quality["has_placeholder_title"] = False
        
        # Check for figures and validate they exist
        figure_matches = re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', content)
        quality["figure_count"] = len(figure_matches)
        quality["has_figures"] = len(figure_matches) > 0
        
        # Validate figure files exist
        missing_figures = []
        existing_figures = []
        base_dir = self.working_dir if self.working_dir else "."
        
        for fig_path in figure_matches:
            # Handle both relative and absolute paths
            if not fig_path.startswith('/'):
                full_path = os.path.join(base_dir, fig_path)
            else:
                full_path = fig_path
                
            if os.path.exists(full_path):
                existing_figures.append(fig_path)
            else:
                missing_figures.append(fig_path)
        
        quality["existing_figures"] = existing_figures
        quality["missing_figures"] = missing_figures
        quality["all_figures_exist"] = len(missing_figures) == 0
        
        # Check for tables
        table_count = len(re.findall(r'\\begin\{table\}', content))
        quality["has_tables"] = table_count > 0
        quality["table_count"] = table_count
        
        # Check for mathematical content
        math_content = len(re.findall(r'\$[^$]+\$|\\begin\{equation\}|\\begin\{align\}', content))
        quality["has_math"] = math_content > 0
        quality["math_expressions"] = math_content
        
        # Check for citations and validate bibliography
        # For documents with \input{} commands, also check included files
        all_content = content
        input_matches = re.findall(r'\\input\{([^}]+)\}', content)
        
        if input_matches:
            # Load content from included files
            base_dir = self.working_dir if self.working_dir else "."
            for input_file in input_matches:
                input_path = os.path.join(base_dir, input_file + ".tex" if not input_file.endswith(".tex") else input_file)
                if os.path.exists(input_path):
                    try:
                        with open(input_path, 'r', encoding='utf-8') as f:
                            included_content = f.read()
                        all_content += "\n" + included_content
                    except Exception as e:
                        print(f"Warning: Could not read included file {input_path}: {e}")
        
        citation_matches = re.findall(r'\\cite\{([^}]+)\}', all_content)
        cited_keys = set()
        for match in citation_matches:
            # Handle multiple keys in a single cite command (e.g., \cite{key1, key2})
            keys = [k.strip() for k in match.split(',')]
            cited_keys.update(keys)
        
        quality["has_citations"] = len(cited_keys) > 0
        quality["citation_count"] = len(citation_matches)
        quality["unique_citation_keys"] = list(cited_keys)
        
        # Validate bibliography file if it exists - CRITICAL for citation coordination
        bib_validation = self._validate_bibliography(cited_keys)
        quality.update(bib_validation)
        
        # NEW: Citation coordination validation
        quality["citation_coordination_valid"] = (
            len(bib_validation["missing_citations"]) == 0 and 
            len(bib_validation["placeholder_entries"]) == 0
        )
        
        # Overall quality score with stricter figure requirements AND citation coordination
        quality_factors = [
            section_analysis["meets_length_requirement"],
            quality["has_figures"],
            quality["all_figures_exist"],  # All referenced figures must exist
            quality["figure_count"] >= 3,  # NEW: Require minimum 3 figures for academic papers
            quality["has_citations"],
            quality.get("bib_file_valid", False),  # Bibliography must be valid
            quality["citation_coordination_valid"],  # NEW: No missing/placeholder citations
            sum(1 for s in section_analysis.values() 
                if isinstance(s, dict) and s.get("has_substantial_content", False)) >= 4
        ]
        quality["overall_quality_score"] = sum(quality_factors) / len(quality_factors)
        
        return quality
    
    def _validate_bibliography(self, cited_keys: set) -> Dict[str, Any]:
        """Validate that .bib file contains entries for cited keys."""
        bib_validation = {
            "bib_file_valid": False,
            "missing_citations": [],
            "placeholder_entries": [],
            "valid_citations": []
        }
        
        if not cited_keys:
            bib_validation["bib_file_valid"] = True  # No citations to validate
            return bib_validation
        
        base_dir = self.working_dir if self.working_dir else "."
        bib_path = os.path.join(base_dir, "references.bib")
        
        if not os.path.exists(bib_path):
            bib_validation["missing_citations"] = list(cited_keys)
            return bib_validation
        
        try:
            with open(bib_path, 'r', encoding='utf-8') as f:
                bib_content = f.read()
            
            # Find all @entry{key, patterns in bib file
            bib_entries = re.findall(r'@\w+\{([^,]+),', bib_content)
            available_keys = set(entry.strip() for entry in bib_entries)
            
            # Check for placeholder entries - more specific patterns
            placeholder_patterns = [
                r'author\s*=\s*["\'](?:Author|Author Names?)\s*["\']',
                r'title\s*=\s*["\'](?:Title|Research Paper Title|Paper Title)\s*["\']',
                r'year\s*=\s*["\'](?:Year)\s*["\']'  # Only \"Year\", not specific years like 2023
            ]
            
            for key in available_keys:
                # Extract the entry for this key
                key_pattern = rf'@\w+\{{{re.escape(key)}\s*,(.*?)^\}}'
                key_match = re.search(key_pattern, bib_content, re.MULTILINE | re.DOTALL)
                if key_match:
                    entry_content = key_match.group(1)
                    # Check if this entry has placeholder content
                    is_placeholder = any(re.search(pattern, entry_content, re.IGNORECASE) 
                                       for pattern in placeholder_patterns)
                    if is_placeholder:
                        bib_validation["placeholder_entries"].append(key)
                    else:
                        bib_validation["valid_citations"].append(key)
            
            # Find missing citations
            bib_validation["missing_citations"] = list(cited_keys - available_keys)
            
            # Bibliography is valid if all cited keys exist and none are placeholders
            bib_validation["bib_file_valid"] = (
                len(bib_validation["missing_citations"]) == 0 and
                len(bib_validation["placeholder_entries"]) == 0 and
                len(cited_keys) > 0
            )
            
        except Exception as e:
            bib_validation["error"] = f"Failed to parse bibliography: {str(e)}"
        
        return bib_validation
    
    def _overall_assessment(self, file_checks: Dict, section_analysis: Dict, content_quality: Dict) -> Dict[str, Any]:
        """Provide overall assessment of success criteria."""
        criteria = {}
        
        # File existence criteria
        criteria["files_exist"] = (
            file_checks["tex_exists"] and 
            file_checks["pdf_exists"] and 
            file_checks["bib_exists"]
        )
        
        # Content length criteria
        criteria["sufficient_content"] = section_analysis["meets_length_requirement"]
        
        # Section completeness
        required_sections = ["abstract", "introduction", "methodology", "results"]
        criteria["required_sections_complete"] = all(
            section_analysis.get(section, {}).get("found", False) 
            for section in required_sections
        )
        
        # Content quality
        criteria["adequate_quality"] = content_quality["overall_quality_score"] >= 0.6
        
        # Bibliography validation
        criteria["valid_bibliography"] = content_quality.get("bib_file_valid", False)
        
        # Figure quantity and quality requirements
        criteria["sufficient_figures"] = content_quality.get("figure_count", 0) >= 3
        
        # Check for placeholder title
        criteria["has_proper_title"] = (
            content_quality.get("has_title", False) and 
            not content_quality.get("has_placeholder_title", False)
        )
        
        # NEW: Citation coordination validation - CRITICAL for preventing [?] citations
        criteria["citation_coordination_valid"] = content_quality.get("citation_coordination_valid", False)
        
        # Overall assessment
        all_criteria_met = all(criteria.values())
        
        return {
            "all_criteria_met": all_criteria_met,
            "criteria_breakdown": criteria,
            "success_percentage": sum(criteria.values()) / len(criteria) * 100
        }
    
    def _generate_recommendations(self, file_checks: Dict, section_analysis: Dict, content_quality: Dict) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # File recommendations
        if not file_checks["tex_exists"]:
            recommendations.append("Create final_paper.tex file")
        if not file_checks["pdf_exists"]:
            recommendations.append("Compile LaTeX to generate final_paper.pdf")
        if not file_checks["bib_exists"]:
            recommendations.append("Create references.bib file with citations")
        
        # Content recommendations
        if not section_analysis["meets_length_requirement"]:
            recommendations.append(f"Expand content - current: {section_analysis['total_content_chars']} chars, need: >15,000 chars")
        
        # Section-specific recommendations
        for section, data in section_analysis.items():
            if isinstance(data, dict) and not data.get("found", True):
                recommendations.append(f"Add missing {section} section")
            elif isinstance(data, dict) and not data.get("has_substantial_content", True):
                recommendations.append(f"Expand {section} section - needs more content")
        
        # Title recommendations
        if content_quality.get("has_placeholder_title", False):
            recommendations.append("Replace placeholder title with actual paper title")
        elif not content_quality.get("has_title", False):
            recommendations.append("Add paper title")
            
        # Bibliography recommendations - ENHANCED for coordination issue
        missing_cites = content_quality.get("missing_citations", [])
        placeholder_cites = content_quality.get("placeholder_entries", [])
        if missing_cites:
            recommendations.append(f"CRITICAL: Add missing bibliography entries for: {', '.join(missing_cites[:5])} - Use CitationSearchTool to find and add entries to references.bib")
        if placeholder_cites:
            recommendations.append(f"CRITICAL: Replace placeholder bibliography entries: {', '.join(placeholder_cites[:5])} - Use CitationSearchTool to find proper literature")
        if not content_quality.get("citation_coordination_valid", False):
            recommendations.append("CITATION COORDINATION FAILURE: All \\cite{} commands must have corresponding entries in references.bib - This causes [?] citations in PDF")
            
        # Figure quantity recommendations
        figure_count = content_quality.get("figure_count", 0)
        if figure_count == 0:
            recommendations.append("Add figures to illustrate results - papers require visual evidence")
        elif figure_count < 3:
            recommendations.append(f"Add more figures - found {figure_count}, academic papers typically need 3-5 figures")
        
        # Quality recommendations
        if not content_quality["has_citations"]:
            recommendations.append("Add citations to support claims")
        
        return recommendations
    
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
            
            # For read operations, verify file exists
            if not os.path.exists(abs_path):
                # Provide helpful error for agent
                parent_dir = os.path.dirname(abs_path)
                if os.path.exists(parent_dir):
                    raise FileNotFoundError(
                        f"File not found: '{path}' does not exist in the workspace. "
                        f"The parent directory exists. Please check the filename."
                    )
                else:
                    raise FileNotFoundError(
                        f"File not found: '{path}' does not exist in the workspace. "
                        f"The directory '{os.path.dirname(path)}' was not found."
                    )
            
            return abs_path