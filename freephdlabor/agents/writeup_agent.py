"""
WriteupAgent implementation using smolagents framework.
Minimal implementation focused on paper writing via writeup tools.
Designed to be managed by ManagerAgent for delegation-based workflow.
"""

import os
import json
import time
from typing import Optional
from .base_research_agent import BaseResearchAgent
from smolagents.memory import ActionStep, MemoryStep

# SPECIALIZED WRITEUP AGENT - LaTeX-focused tools with citation support
# Tools MIGRATED to ResourcePreparationAgent:
# - ExperimentDataOrganizerTool ❌ (prep agent handles all organization)
# - TrainingAnalysisPlotTool ❌ (prep agent handles all plotting)
# - ComparisonPlotTool ❌ (prep agent handles all plotting)
# - StatisticalAnalysisPlotTool ❌ (prep agent handles all plotting)
# - MultiPanelCompositionTool ❌ (prep agent handles all plotting)
# - PlotEnhancementTool ❌ (prep agent handles plot enhancement)

# CORE LaTeX WORKFLOW TOOLS:
from ..toolkits.writeup.latex_generator_tool import LaTeXGeneratorTool
from ..toolkits.writeup.latex_reflection_tool import LaTeXReflectionTool
from ..toolkits.writeup.latex_compiler_tool import LaTeXCompilerTool
from ..toolkits.writeup.latex_content_verification_tool import LaTeXContentVerificationTool
from ..toolkits.writeup.latex_syntax_checker_tool import LaTeXSyntaxCheckerTool
from ..toolkits.writeup.vlm_document_analysis_tool import VLMDocumentAnalysisTool  # PDF validation only
from ..toolkits.general_tools.file_editing.file_editing_tools import (
    SeeFile, CreateFileWithContent, ModifyFile, ListDir, SearchKeyword, DeleteFileOrFolder
)
from ..prompts.writeup_instructions import get_writeup_system_prompt



class WriteupAgent(BaseResearchAgent):
    """
    SPECIALIZED WriteupAgent - Focused on LaTeX writing and compilation with citation support.

    SPECIALIZATION CHANGES:
    - STREAMLINED tools (reduced decision paralysis)
    - EXPECTS pre-organized resources from ResourcePreparationAgent
    - FOCUSES on LaTeX content creation, compilation, and quality validation
    - Citations automatically handled by LaTeXCompilerTool (no manual management needed)

    Design Philosophy:
    - Streamlined LaTeX-focused workflow with citation support
    - Uses pre-validated resources from paper_workspace/
    - Can search for and validate citations dynamically
    - Designed to work AFTER ResourcePreparationAgent has prepared everything
    - Managed by ManagerAgent for delegation-based workflow

    Specialized Agent Process:
    1. READ resource_inventory.md to understand available pre-organized resources
    2. GENERATE LaTeX content using organized figures and data (use [cite: description] placeholders)
    3. ITERATIVELY improve content quality using reflection tools
    4. COMPILE to PDF (LaTeXCompilerTool automatically resolves citations)
    5. VALIDATE success criteria and report completion
    """
    
    def __init__(self, model, workspace_dir: Optional[str] = None, **kwargs):
        """
        Initialize the WriteupAgent.
        
        Args:
            model: The LLM model to use for the agent
            workspace_dir: Directory for workspace operations
            **kwargs: Additional arguments passed to BaseResearchAgent
        """
        # Convert workspace_dir to absolute path immediately to prevent nested directory issues
        if workspace_dir:
            workspace_dir = os.path.abspath(workspace_dir)
            
        # Legacy compatibility: set agent_folder for any code that might reference it
        if workspace_dir:
            self.agent_folder = os.path.join(workspace_dir, "writeup_agent")
        
        # Initialize tools - comprehensive set for academic writing (workspace-aware)
        # NOTE: Tools get raw model for efficiency, agents use LoggingLiteLLMModel for decision tracking
        from ..toolkits.model_utils import get_raw_model
        raw_model = get_raw_model(model)
        
        # STREAMLINED TOOLS - LaTeX workflow with citation support
        # Focus on LaTeX writing, compilation, and citation management
        tools = [
            # CORE LaTeX WORKFLOW (ESSENTIAL - 6 tools)
            LaTeXGeneratorTool(model=raw_model, working_dir=workspace_dir),      # THE CONTENT CREATION BRAIN
            LaTeXReflectionTool(model=raw_model, working_dir=workspace_dir),     # THE QUALITY GUARDIAN
            LaTeXCompilerTool(model=raw_model, working_dir=workspace_dir),       # PDF compilation with BibTeX support
            LaTeXContentVerificationTool(working_dir=workspace_dir),             # Success criteria verification
            LaTeXSyntaxCheckerTool(working_dir=workspace_dir),                   # Document structure validation

            # PDF VALIDATION (1 tool)
            VLMDocumentAnalysisTool(model=raw_model, working_dir=workspace_dir), # PDF validation and analysis

            # NOTE: Citation resolution handled automatically by LaTeXCompilerTool during compilation
            # NOTE: Plotting tools handled by ResourcePreparationAgent (see prep agent for all plotting)
            # NOTE: Data organization handled by ResourcePreparationAgent (see resource_inventory.md)
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
            tools.extend(file_editing_tools)  # Adds 6 file management tools

        # FINAL TOOL COUNT: 7 core tools + 1 citation tool + 6 file tools = 14 total tools
        # Focused on LaTeX workflow with proper citation support
        
        # Generate complete system prompt using template
        system_prompt = get_writeup_system_prompt(
            tools=tools,
            managed_agents=None  # WriteupAgent typically doesn't manage other agents
        )
        
        # Context management is now automatically handled by BaseResearchAgent
        # Model-specific thresholds are calculated automatically
        # Can still override with max_context_tokens for backward compatibility
        max_context_tokens = kwargs.pop('max_context_tokens', None)
        if max_context_tokens:
            kwargs['token_threshold'] = max_context_tokens
        
        # Combine additional_authorized_imports
        default_imports = ['json', 'os', 'subprocess', 'tempfile', 'shutil', 'pathlib', 'glob', 'numpy', 'numpy.random', 'matplotlib', 'matplotlib.pyplot', 'pandas', 'seaborn', 'scipy', 'scipy.stats', 'sklearn']
        passed_imports = kwargs.pop('additional_authorized_imports', [])
        combined_imports = list(set(default_imports + passed_imports))
        
        # Create success criteria validation function for final_answer_checks
        def validate_success_criteria(final_answer, memory):
            """Validate success criteria before allowing termination."""
            return self._validate_writeup_success_criteria(final_answer, memory)
        
        # Initialize BaseResearchAgent with specialized tools and system prompt
        # Context management now automatically integrated via BaseResearchAgent
        super().__init__(
            model=model,  # Pass original model, BaseResearchAgent will handle logging
            agent_name="writeup_agent",
            workspace_dir=workspace_dir,
            tools=tools,
            additional_authorized_imports=combined_imports,
            max_steps=150,  # Increased from default 20 to allow comprehensive paper writing
            final_answer_checks=[validate_success_criteria],  # CRITICAL: Enable quality gate validation
            **kwargs
        )

        # Set system prompt after initialization (correct smolagents pattern)
        self.prompt_templates["system_prompt"] = system_prompt

        # Resume memory if possible
        self.resume_memory()

    def _validate_writeup_success_criteria(self, final_answer, memory):
        """
        Validate success criteria before allowing termination.
        
        Args:
            final_answer: The proposed final answer from the agent
            memory: The agent's memory containing conversation history
            
        Returns:
            bool: True if success criteria are met, False otherwise
            
        This method blocks termination if success criteria aren't met by raising
        an exception with detailed feedback to help the agent continue working.
        """
        print("\n🔍 MANDATORY SUCCESS CRITERIA VERIFICATION")
        print("=" * 60)
        
        # Import here to avoid circular imports
        from ..toolkits.writeup.latex_content_verification_tool import LaTeXContentVerificationTool
        
        # Check if final_paper.tex exists first
        tex_path = os.path.join(self.workspace_dir or ".", "final_paper.tex")
        pdf_path = os.path.join(self.workspace_dir or ".", "final_paper.pdf")
        
        if not os.path.exists(tex_path):
            print("❌ TERMINATION BLOCKED: final_paper.tex does not exist")
            print("\n📋 REQUIRED ACTIONS:")
            print("1. Create individual sections using LaTeXGeneratorTool")
            print("2. Apply iterative reflection to improve each section") 
            print("3. Create final_paper.tex using LaTeXGeneratorTool with section_type='main_document'")
            print("4. Ensure final_paper.tex uses \\input{} for all sections")
            print("5. Compile final_paper.tex to PDF")
            raise ValueError("TERMINATION_BLOCKED: Missing final_paper.tex. Please create the complete LaTeX document first.")
        
        # Check if final_paper.pdf exists and is valid
        if not os.path.exists(pdf_path):
            print("❌ TERMINATION BLOCKED: final_paper.pdf does not exist")
            print("\n📋 CRITICAL FAILURE: LaTeX compilation failed or was not attempted")
            print("1. You MUST successfully compile final_paper.tex to PDF using LaTeXCompilerTool")
            print("2. If compilation fails with errors, you MUST fix the LaTeX errors")
            print("3. If compilation times out, check for infinite loops or simplify the document")
            print("4. NEVER declare task complete without a valid final_paper.pdf")
            print("\n⚠️  Task Status: FAILED - No PDF produced")
            raise ValueError("TERMINATION_BLOCKED: Missing final_paper.pdf. LaTeX compilation failed. The task cannot be completed without a valid PDF.")
        
        # Verify PDF is valid using VLMDocumentAnalysisTool
        print("🔍 Validating PDF integrity...")
        from ..toolkits.writeup.vlm_document_analysis_tool import VLMDocumentAnalysisTool
        import json
        
        try:
            # Use VLM tool to validate PDF (it will check if PDF can be opened and read)
            vlm_tool = VLMDocumentAnalysisTool(model=self.model, working_dir=self.workspace_dir)
            validation_result_str = vlm_tool.forward(
                file_paths="final_paper.pdf",
                analysis_focus="pdf_validation"
            )
            
            # Parse validation result
            validation_result = json.loads(validation_result_str)
            
            # Check for errors in PDF processing
            if validation_result.get("error"):
                print(f"❌ TERMINATION BLOCKED: PDF validation failed - {validation_result.get('error')}")
                print("\n📋 CRITICAL FAILURE: PDF is corrupted or unreadable")
                print("1. The PDF file exists but cannot be properly read")
                print("2. Check LaTeX compilation logs for errors")
                print("3. Ensure LaTeX compilation completed successfully")
                print("4. Try recompiling with simplified content if timeout occurs")
                print("\n⚠️  Task Status: FAILED - Corrupted PDF")
                raise ValueError(f"TERMINATION_BLOCKED: PDF validation failed - {validation_result.get('error')}. Fix compilation and regenerate.")
            
            # Check if PDF is publication ready
            overall_assessment = validation_result.get("overall_assessment", {})
            if not overall_assessment.get("publication_ready", False):
                # Extract all issue categories from top-level validation result
                # VLMDocumentAnalysisTool returns issues in separate top-level lists
                layout_issues = validation_result.get("layout_issues", [])
                missing_citations = validation_result.get("missing_citations", [])
                missing_figures = validation_result.get("missing_figures", [])
                structural_problems = validation_result.get("structural_problems", [])

                # Combine all issues for overall count
                all_issues = layout_issues + missing_citations + missing_figures + structural_problems

                print("❌ TERMINATION BLOCKED: PDF has critical issues")
                print("\n📋 CRITICAL ISSUES FOUND IN PDF:")

                # Display issues by category for clarity
                if layout_issues:
                    print("\n🔧 Layout Issues:")
                    for issue in layout_issues:
                        print(f"  • {issue}")

                if missing_citations:
                    print("\n📚 Missing Citations:")
                    for issue in missing_citations:
                        print(f"  • {issue}")

                if missing_figures:
                    print("\n🖼️  Missing Figures:")
                    for issue in missing_figures:
                        print(f"  • {issue}")

                if structural_problems:
                    print("\n📐 Structural Problems:")
                    for issue in structural_problems:
                        print(f"  • {issue}")

                if not all_issues:
                    print("  • No specific issues identified, but PDF marked as not publication-ready")
                    print("  • This may indicate VLM validation issues or generic quality concerns")

                print("\n⚠️  Task Status: FAILED - PDF not publication ready")

                # Create detailed error message with actual issues
                issue_summary = ", ".join(all_issues[:3]) if all_issues else "PDF validation failed without specific details"
                raise ValueError(f"TERMINATION_BLOCKED: PDF has critical issues: {issue_summary}. Fix these issues and regenerate.")
                
        except Exception as e:
            print(f"❌ TERMINATION BLOCKED: PDF validation error - {str(e)}")
            print("\n📋 CRITICAL FAILURE: Unable to validate PDF")
            print("1. PDF may be corrupted or incomplete")
            print("2. Ensure LaTeX compilation completed without errors")
            print("\n⚠️  Task Status: FAILED - PDF validation error")
            raise ValueError(f"TERMINATION_BLOCKED: PDF validation failed with error: {str(e)}. Regenerate the PDF.")
        
        # Run mandatory verification
        print("🔧 Running LaTeXContentVerificationTool...")
        verification_tool = LaTeXContentVerificationTool(working_dir=self.workspace_dir)
        verification_result_str = verification_tool.forward("final_paper.tex")
        
        try:
            import json
            verification_result = json.loads(verification_result_str)
        except:
            print("⚠️  Warning: Could not parse verification result")
            verification_result = {"overall_assessment": {"all_criteria_met": False}}
        
        all_criteria_met = verification_result.get("overall_assessment", {}).get("all_criteria_met", False)
        
        if all_criteria_met:
            print("✅ SUCCESS CRITERIA VERIFIED - Termination allowed")
            print("=" * 60)
            # Return True to allow termination
            return True
        else:
            print("❌ TERMINATION BLOCKED: Success criteria not met")
            print("=" * 60)
            
            # Extract specific failure details for actionable feedback
            criteria_breakdown = verification_result.get("overall_assessment", {}).get("criteria_breakdown", {})
            recommendations = verification_result.get("recommendations", [])
            section_analysis = verification_result.get("section_analysis", {})
            content_quality = verification_result.get("content_quality", {})
            
            # Provide detailed feedback
            print("\n📋 CRITICAL ISSUES TO FIX:")
            
            # File existence issues
            file_checks = verification_result.get("file_checks", {})
            if not file_checks.get("pdf_exists", True):
                print("• Missing final_paper.pdf - Compile LaTeX to PDF")
            if not file_checks.get("bib_exists", True):
                print("• Missing references.bib - Create bibliography file")
            
            # Section completeness issues
            print("\n📝 MISSING OR INADEQUATE SECTIONS:")
            for section, analysis in section_analysis.items():
                if isinstance(analysis, dict) and not analysis.get("found", True):
                    print(f"• Missing {section} section")
                elif isinstance(analysis, dict) and not analysis.get("has_substantial_content", True):
                    print(f"• {section} section needs more content ({analysis.get('content_chars', 0)} chars)")
            
            # Content quality issues
            if not content_quality.get("has_figures", True):
                print(f"• Missing figures (found: {content_quality.get('figure_count', 0)}, need: 3-5)")
            if not content_quality.get("has_citations", True):
                print(f"• Missing citations (found: {content_quality.get('citation_count', 0)})")
            
            # Length requirement
            total_chars = section_analysis.get("total_content_chars", 0)
            if total_chars < 15000:
                print(f"• Content too short: {total_chars} chars (need: >15,000 chars)")
            
            # Specific recommendations
            if recommendations:
                print("\n🔧 RECOMMENDED ACTIONS:")
                for i, rec in enumerate(recommendations[:10], 1):  # Limit to top 10
                    print(f"{i}. {rec}")
            
            print("\n⚠️  You must fix these issues before the task can be completed.")
            print("Use your available tools to address each issue, then try to complete again.")
            print("=" * 60)
            
            # Raise an exception to block termination with detailed feedback
            feedback_message = f"""TERMINATION BLOCKED: Success criteria verification failed.

CRITICAL ISSUES FOUND:
- Files missing: {not file_checks.get('pdf_exists', True) or not file_checks.get('bib_exists', True)}
- Sections incomplete: {sum(1 for s, a in section_analysis.items() if isinstance(a, dict) and (not a.get('found', True) or not a.get('has_substantial_content', True)))} sections need work
- Content length: {total_chars}/15000+ characters required
- Figures missing: {content_quality.get('figure_count', 0)}/3-5 required
- Citations missing: {content_quality.get('citation_count', 0)} found

Please fix these issues using your available tools and try again."""
            
            # Raise exception to prevent termination and force agent to continue
            raise ValueError(feedback_message)
