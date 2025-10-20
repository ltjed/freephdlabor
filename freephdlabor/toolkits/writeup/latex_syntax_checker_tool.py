"""
LaTeXSyntaxCheckerTool - Check LaTeX syntax and identify potential issues.

This tool provides LaTeX syntax validation including:
- Syntax error detection without full compilation
- Missing package warnings
- Unmatched braces/environments detection
- Citation and reference validation
- Common LaTeX pitfall identification
- Style and formatting recommendations

Useful for quick syntax validation before compilation.
"""

import json
import os
import re
import tempfile
from typing import List, Dict, Any, Optional, Tuple
from smolagents import Tool


class LaTeXSyntaxCheckerTool(Tool):
    name = "latex_syntax_checker_tool"
    description = """
    Check LaTeX syntax and identify potential issues without full compilation.
    
    This tool is essential for:
    - Quick syntax validation before compilation
    - Identifying common LaTeX errors and warnings
    - Checking for missing packages or commands
    - Validating citations and references
    - Detecting unmatched braces and environments
    
    Use this tool when:
    - You want to check LaTeX syntax quickly
    - You're debugging LaTeX compilation issues
    - You want to catch errors before running full compilation
    - You're reviewing LaTeX code for common mistakes
    - You need style and formatting recommendations
    
    The tool performs static analysis of LaTeX code to catch common issues
    that would cause compilation failures or poor formatting.
    
    Input: LaTeX file path or content
    Output: Detailed syntax check results with errors, warnings, and recommendations
    """
    
    inputs = {
        "latex_input": {
            "type": "string", 
            "description": "Path to .tex file or raw LaTeX content to check"
        },
        "check_level": {
            "type": "string",
            "description": "Check level: 'basic', 'thorough', or 'strict' (default: 'thorough')",
            "nullable": True
        }
    }
    
    outputs = {
        "syntax_check_result": {
            "type": "string",
            "description": "Detailed syntax check results with errors, warnings, and recommendations"
        }
    }
    
    output_type = "string"

    def __init__(self, working_dir: Optional[str] = None):
        """Initialize LaTeXSyntaxCheckerTool.
        
        Args:
            working_dir: Working directory for workspace-aware file access
        """
        super().__init__()
        # Convert to absolute path to prevent nested directory issues
        self.working_dir = os.path.abspath(working_dir) if working_dir else None
        
    def forward(self, latex_input: str, check_level: str = "thorough") -> str:
        """
        Check LaTeX syntax and identify issues.
        
        Args:
            latex_input: Path to .tex file or raw LaTeX content
            check_level: Level of checking ('basic', 'thorough', 'strict')
            
        Returns:
            JSON string containing syntax check results
        """
        try:
            # Get LaTeX content (workspace-aware)
            latex_input_stripped = latex_input.strip()
            resolved_input = self._safe_path(latex_input_stripped) if self.working_dir else latex_input_stripped
            
            if os.path.exists(resolved_input):
                with open(resolved_input, 'r', encoding='utf-8') as f:
                    latex_content = f.read()
                file_path = resolved_input
            else:
                latex_content = latex_input
                file_path = None
            
            # Perform syntax checks
            errors = []
            warnings = []
            recommendations = []
            
            # Basic checks (always performed)
            errors.extend(self._check_basic_syntax(latex_content))
            warnings.extend(self._check_basic_warnings(latex_content))
            
            # Thorough checks
            if check_level in ['thorough', 'strict']:
                errors.extend(self._check_environments(latex_content))
                errors.extend(self._check_math_mode(latex_content))
                warnings.extend(self._check_citations_references(latex_content))
                warnings.extend(self._check_packages(latex_content))
                recommendations.extend(self._check_style_recommendations(latex_content))
            
            # Strict checks
            if check_level == 'strict':
                warnings.extend(self._check_strict_formatting(latex_content))
                recommendations.extend(self._check_advanced_recommendations(latex_content))
            
            # Summary
            total_issues = len(errors) + len(warnings)
            severity = self._assess_severity(errors, warnings)
            
            result = {
                "file_path": file_path,
                "check_level": check_level,
                "summary": {
                    "total_issues": total_issues,
                    "errors": len(errors),
                    "warnings": len(warnings),
                    "recommendations": len(recommendations),
                    "severity": severity
                },
                "errors": errors,
                "warnings": warnings,
                "recommendations": recommendations,
                "analysis": {
                    "document_class": self._extract_document_class(latex_content),
                    "packages_used": self._extract_packages(latex_content),
                    "sections_found": self._count_sections(latex_content),
                    "word_count_estimate": self._estimate_word_count(latex_content)
                }
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            error_result = {
                "file_path": latex_input if os.path.exists(latex_input.strip()) else None,
                "check_level": check_level,
                "error": f"Syntax check failed: {str(e)}",
                "summary": {"total_issues": 0, "errors": 0, "warnings": 0, "recommendations": 0, "severity": "unknown"},
                "errors": [],
                "warnings": [],
                "recommendations": []
            }
            return json.dumps(error_result, indent=2)
    
    def _check_basic_syntax(self, content: str) -> List[Dict[str, Any]]:
        """Check basic LaTeX syntax errors."""
        errors = []
        lines = content.split('\n')
        
        brace_stack = []
        
        for line_num, line in enumerate(lines, 1):
            # Check for unmatched braces
            for char_pos, char in enumerate(line):
                if char == '{':
                    brace_stack.append((line_num, char_pos))
                elif char == '}':
                    if not brace_stack:
                        errors.append({
                            "type": "syntax_error",
                            "line": line_num,
                            "column": char_pos,
                            "message": "Unmatched closing brace",
                            "code": line.strip()
                        })
                    else:
                        brace_stack.pop()
            
            # Skip the basic environment check here - it's handled better in _check_environments
        
        # Check for remaining unmatched opening braces
        for line_num, char_pos in brace_stack:
            errors.append({
                "type": "syntax_error",
                "line": line_num,
                "column": char_pos,
                "message": "Unmatched opening brace",
                "code": lines[line_num-1].strip() if line_num <= len(lines) else ""
            })
        
        return errors
    
    def _check_basic_warnings(self, content: str) -> List[Dict[str, Any]]:
        """Check for basic warnings."""
        warnings = []
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            # Check for potentially problematic constructs
            if '$$' in line:
                warnings.append({
                    "type": "deprecated_syntax",
                    "line": line_num,
                    "message": "Use \\[ \\] instead of $$ for display math",
                    "code": line.strip()
                })
            
            if re.search(r'\\\\\\s*$', line):
                warnings.append({
                    "type": "formatting_warning",
                    "line": line_num,
                    "message": "Avoid \\\\ at end of paragraphs, use blank line instead",
                    "code": line.strip()
                })
            
            # Check for common typos
            if re.search(r'\\emph\s*\\emph', line):
                warnings.append({
                    "type": "style_warning",
                    "line": line_num,
                    "message": "Nested \\emph commands, consider using \\textbf",
                    "code": line.strip()
                })
        
        return warnings
    
    def _check_environments(self, content: str) -> List[Dict[str, Any]]:
        """Check environment matching."""
        errors = []
        
        # Find all \begin and \end commands
        begin_pattern = re.compile(r'\\begin\{([^}]+)\}')
        end_pattern = re.compile(r'\\end\{([^}]+)\}')
        
        begins = [(m.group(1), m.start()) for m in begin_pattern.finditer(content)]
        ends = [(m.group(1), m.start()) for m in end_pattern.finditer(content)]
        
        # Simple matching check
        begin_stack = []
        
        all_commands = sorted(
            [(pos, 'begin', name) for name, pos in begins] + 
            [(pos, 'end', name) for name, pos in ends]
        )
        
        for pos, cmd_type, env_name in all_commands:
            if cmd_type == 'begin':
                begin_stack.append((env_name, pos))
            elif cmd_type == 'end':
                if not begin_stack:
                    errors.append({
                        "type": "environment_error",
                        "position": pos,
                        "message": f"\\end{{{env_name}}} without matching \\begin{{{env_name}}}",
                        "environment": env_name
                    })
                else:
                    last_begin_env, last_begin_pos = begin_stack.pop()
                    if last_begin_env != env_name:
                        errors.append({
                            "type": "environment_mismatch",
                            "position": pos,
                            "message": f"\\end{{{env_name}}} doesn't match \\begin{{{last_begin_env}}}",
                            "expected": last_begin_env,
                            "found": env_name
                        })
        
        # Check for unmatched begins
        for env_name, pos in begin_stack:
            errors.append({
                "type": "environment_error",
                "position": pos,
                "message": f"\\begin{{{env_name}}} without matching \\end{{{env_name}}}",
                "environment": env_name
            })
        
        return errors
    
    def _check_math_mode(self, content: str) -> List[Dict[str, Any]]:
        """Check math mode syntax."""
        errors = []
        
        # Check for unmatched $ signs
        single_dollar_count = content.count('$') - content.count('\\$')  # Exclude escaped $
        if single_dollar_count % 2 != 0:
            errors.append({
                "type": "math_error",
                "message": "Unmatched $ for inline math mode",
                "count": single_dollar_count
            })
        
        # Check for common math errors
        if re.search(r'\$\s*\$', content):
            errors.append({
                "type": "math_error", 
                "message": "Empty math mode $$ found",
                "suggestion": "Remove empty math delimiters"
            })
        
        return errors
    
    def _check_citations_references(self, content: str) -> List[Dict[str, Any]]:
        """Check citations and references."""
        warnings = []
        
        # Find \cite commands
        cite_pattern = re.compile(r'\\cite\{([^}]+)\}')
        cites = [m.group(1) for m in cite_pattern.finditer(content)]
        
        # Find \ref commands  
        ref_pattern = re.compile(r'\\ref\{([^}]+)\}')
        refs = [m.group(1) for m in ref_pattern.finditer(content)]
        
        # Find \label commands
        label_pattern = re.compile(r'\\label\{([^}]+)\}')
        labels = [m.group(1) for m in label_pattern.finditer(content)]
        
        # Check for undefined references
        for ref in refs:
            if ref not in labels:
                warnings.append({
                    "type": "reference_warning",
                    "message": f"Reference '{ref}' may be undefined (no matching \\label found)",
                    "reference": ref
                })
        
        # Check for unused labels
        for label in labels:
            if label not in refs:
                warnings.append({
                    "type": "label_warning",
                    "message": f"Label '{label}' defined but never referenced",
                    "label": label
                })
        
        return warnings
    
    def _check_packages(self, content: str) -> List[Dict[str, Any]]:
        """Check package usage."""
        warnings = []
        
        # Common packages and their typical commands
        package_commands = {
            'amsmath': [r'\\align', r'\\equation', r'\\gather'],
            'graphicx': [r'\\includegraphics'],
            'hyperref': [r'\\href', r'\\url'],
            'biblatex': [r'\\printbibliography'],
            'geometry': [r'\\newgeometry'],
            'xcolor': [r'\\textcolor', r'\\colorbox']
        }
        
        # Find used packages
        package_pattern = re.compile(r'\\usepackage(?:\[[^\]]*\])?\{([^}]+)\}')
        used_packages = set()
        for m in package_pattern.finditer(content):
            packages = [p.strip() for p in m.group(1).split(',')]
            used_packages.update(packages)
        
        # Check for missing packages
        for package, commands in package_commands.items():
            if package not in used_packages:
                for cmd_pattern in commands:
                    if re.search(cmd_pattern, content):
                        warnings.append({
                            "type": "package_warning",
                            "message": f"Using commands that require '{package}' package but package not loaded",
                            "package": package,
                            "command_pattern": cmd_pattern
                        })
                        break
        
        return warnings
    
    def _check_style_recommendations(self, content: str) -> List[Dict[str, Any]]:
        """Check style and provide recommendations."""
        recommendations = []
        
        # Check for consistent quotation marks
        if '"' in content:
            recommendations.append({
                "type": "style_recommendation",
                "message": "Consider using `` and '' for quotation marks instead of \"",
                "suggestion": "Replace \" with `` for opening and '' for closing quotes"
            })
        
        # Check for hyphenation
        if '--' not in content and '–' not in content:
            if re.search(r'\b\w+-\w+\b', content):
                recommendations.append({
                    "type": "style_recommendation", 
                    "message": "Consider using -- for en-dashes in ranges and compound words",
                    "suggestion": "Use -- for number ranges (e.g., pages 10--20)"
                })
        
        return recommendations
    
    def _check_strict_formatting(self, content: str) -> List[Dict[str, Any]]:
        """Strict formatting checks."""
        warnings = []
        
        # Check line length (common in academic writing)
        lines = content.split('\n')
        for line_num, line in enumerate(lines, 1):
            if len(line) > 80 and not line.strip().startswith('%'):
                warnings.append({
                    "type": "formatting_warning",
                    "line": line_num,
                    "message": f"Line length ({len(line)}) exceeds 80 characters",
                    "suggestion": "Consider breaking long lines for better readability"
                })
        
        return warnings
    
    def _check_advanced_recommendations(self, content: str) -> List[Dict[str, Any]]:
        """Advanced style recommendations."""
        recommendations = []
        
        # Check for consistent spacing
        if re.search(r'[.!?]\s{2,}', content):
            recommendations.append({
                "type": "style_recommendation",
                "message": "Multiple spaces after sentence endings detected",
                "suggestion": "Use single space after periods for consistency"
            })
        
        return recommendations
    
    def _assess_severity(self, errors: List, warnings: List) -> str:
        """Assess overall severity of issues."""
        if len(errors) > 5:
            return "high"
        elif len(errors) > 0:
            return "medium"
        elif len(warnings) > 10:
            return "medium"
        elif len(warnings) > 0:
            return "low"
        else:
            return "none"
    
    def _extract_document_class(self, content: str) -> Optional[str]:
        """Extract document class."""
        match = re.search(r'\\documentclass(?:\[[^\]]*\])?\{([^}]+)\}', content)
        return match.group(1) if match else None
    
    def _extract_packages(self, content: str) -> List[str]:
        """Extract used packages."""
        packages = set()
        for match in re.finditer(r'\\usepackage(?:\[[^\]]*\])?\{([^}]+)\}', content):
            packages.update(p.strip() for p in match.group(1).split(','))
        return sorted(list(packages))
    
    def _count_sections(self, content: str) -> Dict[str, int]:
        """Count sections."""
        return {
            "sections": len(re.findall(r'\\section\{', content)),
            "subsections": len(re.findall(r'\\subsection\{', content)),
            "subsubsections": len(re.findall(r'\\subsubsection\{', content))
        }
    
    def _estimate_word_count(self, content: str) -> int:
        """Estimate word count (rough)."""
        # Remove LaTeX commands and count words
        text = re.sub(r'\\[a-zA-Z]+(?:\[[^\]]*\])?(?:\{[^}]*\})*', '', content)
        text = re.sub(r'[{}%]', '', text)
        words = text.split()
        return len([w for w in words if w.strip()])
    
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
            
            # For syntax checker, check if LaTeX file exists for reading
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