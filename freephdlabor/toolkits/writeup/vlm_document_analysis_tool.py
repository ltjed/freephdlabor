"""
VLMDocumentAnalysisTool - Analyze scientific figures and PDF documents using Vision-Language Models.

This tool provides comprehensive visual analysis including:
- Scientific figure analysis (plots, charts, visualizations)
- PDF document visual validation (layout, citations, figures)
- Content description and quality assessment
- Technical element evaluation (axes, legends, labels)
- Publication quality evaluation and error detection
- Layout problem identification (formula overflow, spacing issues)
- Missing element detection (figures, citations, references)

Uses the VLM functionality from freephdlabor.llm for image and document analysis.
"""

import json
import os
import re
import tempfile
from typing import List, Dict, Any, Optional, Tuple
from smolagents import Tool

from ...llm import get_response_from_vlm, create_vlm_client

# Try to import PyMuPDF for PDF processing
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


class VLMDocumentAnalysisTool(Tool):
    name = "vlm_document_analysis_tool"
    description = """
    Analyze scientific figures and PDF documents using Vision-Language Models for comprehensive visual validation.
    
    This tool is essential for:
    - Understanding experimental results presented in figures
    - Creating accurate figure captions and descriptions
    - Identifying key trends and findings in visualizations
    - Assessing figure quality for publication standards
    - **PDF VISUAL VALIDATION**: Detecting layout problems, missing figures, missing citations
    - **QUALITY ASSURANCE**: Identifying severe layout issues (formulas exceeding page boundaries)
    - **CITATION VERIFICATION**: Detecting "?" marks indicating missing bibliography entries
    - **FIGURE VALIDATION**: Identifying empty spaces where figures should appear
    
    Use this tool when:
    - You need to describe figures in your LaTeX writeup
    - You want to understand what insights figures convey
    - You need to assess if figures are publication-ready
    - **MANDATORY**: Validating final PDF for visual correctness before task completion
    - **CRITICAL**: Checking compiled PDF for layout problems and missing elements
    - You're writing results/discussion sections referencing figures
    
    The tool analyzes: line plots, bar charts, heatmaps, scatter plots, subplots, 
    PDF pages, LaTeX-compiled documents, and other scientific visualizations.
    
    Input: Path to image file(s), PDF file(s), or list of paths
    Output: Detailed structured analysis with scientific insights and visual validation results
    """
    
    inputs = {
        "file_paths": {
            "type": "string", 
            "description": "Single file path or JSON list of file paths to analyze (supports images: PNG, JPG, PDF pages)"
        },
        "analysis_focus": {
            "type": "string",
            "description": "Analysis mode (default: 'comprehensive'):\n" +
                          "**IMAGE-ONLY MODES (PNG, JPG, etc.):**\n" +
                          "• 'image_content': Extract scientific insights and research conclusions from figures/plots\n" +
                          "• 'image_quality': Assess figure publication readiness, visual clarity, professional presentation\n" +
                          "• 'image_trends': Focus on data patterns, experimental trends, quantitative results in visualizations\n" +
                          "• 'image_technical': Analyze technical details like axes labels, legends, statistical significance\n" +
                          "• 'comprehensive': Complete image analysis combining content, quality, and technical assessment\n" +
                          "**PDF-ONLY MODES:**\n" +
                          "• 'pdf_reading': Read and analyze PDF research papers - extract text, analyze figures, understand research content\n" +
                          "• 'pdf_validation': Check LaTeX compilation quality - missing citations (?), broken figures, layout errors",
            "nullable": True
        }
    }
    
    outputs = {
        "analysis": {
            "type": "string",
            "description": "Detailed structured analysis of the figure(s) with scientific insights"
        }
    }
    
    output_type = "string"

    def __init__(self, model=None, working_dir: Optional[str] = None):
        """
        Initialize VLMDocumentAnalysisTool.
        
        Args:
            model: LLM model object (not used directly, kept for consistency)
            working_dir: Working directory for workspace-aware file access
        """
        super().__init__()
        # Use GPT-4o for proven VLM performance on research tasks
        # GPT-4o: Mature vision model with established performance
        # GPT-5 shows poor object detection (mAP50:95 1.5 vs competitors' 13.3)
        self.vlm_model = "gpt-4o-2024-05-13"  # Proven GPT-4o for reliable scientific analysis
        # Convert to absolute path to prevent nested directory issues
        self.working_dir = os.path.abspath(working_dir) if working_dir else None
        
    def forward(self, file_paths, analysis_focus: str = "comprehensive") -> str:
        """
        Analyze scientific figures or PDF documents using VLM and return detailed insights.
        
        Args:
            file_paths: Single path (str), list of paths, or JSON string of paths (images or PDFs)
            analysis_focus: Focus area for analysis (including 'pdf_validation' for document verification)
            
        Returns:
            JSON string containing detailed analysis and validation results
        """
        try:
            # Parse file paths - handle string, list, or JSON input
            if isinstance(file_paths, list):
                # Already a list
                paths = file_paths
            elif isinstance(file_paths, str):
                if file_paths.startswith('[') and file_paths.endswith(']'):
                    # JSON list format
                    paths = json.loads(file_paths)
                else:
                    # Single path
                    paths = [file_paths.strip()]
            else:
                # Convert to string and try again
                paths = [str(file_paths)]
            
            # Validate paths exist (using workspace-aware path resolution)
            valid_paths = []
            for path in paths:
                try:
                    resolved_path = self._safe_path(path) if self.working_dir else path
                    print(f"Resolved Path {resolved_path}")
                    if os.path.exists(resolved_path):
                        valid_paths.append(resolved_path)
                    else:
                        print(f"Warning: Image path does not exist: {path}")
                except PermissionError as e:
                    print(f"Warning: Access denied to path {path}: {e}")
            
            if not valid_paths:
                return json.dumps({
                    "error": "No valid file paths found",
                    "analysis": None
                })
            
            # Check for PDF files and handle them specially
            pdf_files = [p for p in valid_paths if p.lower().endswith('.pdf')]
            image_files = [p for p in valid_paths if not p.lower().endswith('.pdf')]
            
            # Handle PDF files based on analysis focus
            if pdf_files:
                if analysis_focus == "pdf_validation":
                    # Use PDF validation pipeline for LaTeX compilation checking
                    return self._analyze_pdf_comprehensively(pdf_files[0])
                elif analysis_focus == "pdf_reading":
                    # Use PDF reading pipeline for research paper content analysis
                    return self._analyze_pdf_for_research(pdf_files[0], analysis_focus)
                else:
                    # For other modes, default to PDF reading behavior
                    return self._analyze_pdf_for_research(pdf_files[0], analysis_focus)
            
            # Handle image files only
            if not image_files:
                return json.dumps({
                    "error": "No valid image or PDF files found",
                    "provided_files": valid_paths
                })
            
            # Create VLM client for image analysis
            client, model = create_vlm_client(self.vlm_model)
            
            # Prepare analysis prompt based on focus
            analysis_prompt = self._get_analysis_prompt(analysis_focus, len(image_files))
            
            # System message for scientific figure analysis
            system_message = """You are an expert scientific figure analyst specializing in machine learning and AI research papers. 
            You provide detailed, accurate, and insightful analysis of experimental visualizations. 
            Focus on extracting meaningful scientific insights and assessing publication quality."""
            
            # Perform VLM analysis on images
            response, _ = get_response_from_vlm(
                prompt=analysis_prompt,
                images=image_files,
                client=client,
                model=model,
                system_message=system_message,
                print_debug=False
            )
            
            # Structure the analysis
            structured_analysis = self._structure_analysis(response, valid_paths, analysis_focus)
            
            return json.dumps(structured_analysis, indent=2)
            
        except Exception as e:
            error_result = {
                "error": f"VLM document analysis failed: {str(e)}",
                "analysis": None,
                "file_paths": file_paths,
                "focus": analysis_focus
            }
            return json.dumps(error_result, indent=2)
    
    def _analyze_pdf_comprehensively(self, pdf_path: str) -> str:
        """
        Comprehensive PDF analysis pipeline:
        1. Extract text and images from PDF
        2. Generate context-aware questions for each image
        3. Use VLM to analyze images with specific questions
        4. Reconstruct full document with image analysis integrated
        """
        if not PYMUPDF_AVAILABLE:
            return json.dumps({
                "error": "PyMuPDF not available",
                "message": "Install PyMuPDF with: pip install PyMuPDF",
                "fallback_suggestion": "Use LaTeX compilation analysis instead"
            })
        
        try:
            # Ensure workspace-aware path resolution
            safe_pdf_path = self._safe_path(pdf_path) if self.working_dir else pdf_path
            
            # Step 1: Extract text and images from PDF
            extracted_data = self._extract_pdf_content(safe_pdf_path)
            
            # Step 2: Generate context-aware questions for each image
            image_analyses = []
            for image_info in extracted_data["images"]:
                # Generate questions based on context
                questions = self._generate_context_questions(image_info["context"], image_info["expected_content"])
                
                # Step 3: Analyze image with VLM
                vlm_analysis = self._analyze_image_with_questions(image_info["image_path"], questions)
                
                image_analyses.append({
                    "image_id": image_info["image_id"],
                    "context": image_info["context"],
                    "questions": questions,
                    "vlm_analysis": vlm_analysis,
                    "image_path": image_info["image_path"]
                })
            
            # Step 4: Reconstruct full document
            final_text = self._reconstruct_document_with_analysis(extracted_data["text"], image_analyses)
            
            # Analyze for publication issues
            publication_issues = self._identify_publication_issues(extracted_data, image_analyses)
            
            return json.dumps({
                "status": "success",
                "analysis_type": "comprehensive_pdf_analysis",
                "original_text_length": len(extracted_data["text"]),
                "images_analyzed": len(image_analyses),
                "publication_issues": publication_issues,
                "reconstructed_text": final_text,
                "image_analyses": image_analyses,
                "pdf_path": safe_pdf_path
            }, indent=2)
            
        except Exception as e:
            return json.dumps({
                "error": f"Comprehensive PDF analysis failed: {str(e)}",
                "pdf_path": pdf_path  # Use original path in error case
            })
    
    def _analyze_pdf_for_research(self, pdf_path: str, analysis_focus: str) -> str:
        """
        Research paper analysis pipeline adapted from PDF validation:
        1. Extract text and images from PDF
        2. Generate research-focused questions for each image
        3. Use VLM to analyze images with research context
        4. Reconstruct document with scientific insights integrated
        """
        if not PYMUPDF_AVAILABLE:
            return json.dumps({
                "error": "PyMuPDF not available",
                "message": "Install PyMuPDF with: pip install PyMuPDF",
                "fallback_suggestion": "Use individual image analysis instead"
            })
        
        try:
            # Ensure workspace-aware path resolution
            safe_pdf_path = self._safe_path(pdf_path) if self.working_dir else pdf_path
            
            # Step 1: Extract text and images from PDF
            extracted_data = self._extract_pdf_content(safe_pdf_path)
            
            # Step 2: Analyze images with research-focused questions
            image_analyses = []
            for image_info in extracted_data["images"]:
                # Generate research-focused questions instead of validation questions
                research_questions = self._generate_research_questions(
                    image_info["context"], 
                    image_info["expected_content"],
                    analysis_focus
                )
                
                # Analyze image with research context
                analysis_result = self._analyze_image_with_questions(
                    image_info["image_path"], 
                    research_questions
                )
                
                image_analyses.append({
                    "image_id": image_info["image_id"],
                    "page_number": image_info["page_number"],
                    "context": image_info["context"],
                    "expected_content": image_info["expected_content"],
                    "research_analysis": analysis_result,
                    "image_path": image_info["image_path"]
                })
            
            # Step 3: Extract research insights from text and images
            research_insights = self._extract_research_insights(extracted_data["text"], image_analyses, analysis_focus)
            
            return json.dumps({
                "status": "success",
                "analysis_type": f"research_paper_analysis_{analysis_focus}",
                "document_length": len(extracted_data["text"]),
                "images_analyzed": len(image_analyses),
                "research_insights": research_insights,
                "full_text": extracted_data["text"][:5000] + "..." if len(extracted_data["text"]) > 5000 else extracted_data["text"],
                "image_analyses": image_analyses,
                "pdf_path": safe_pdf_path
            }, indent=2)
            
        except Exception as e:
            return json.dumps({
                "error": f"Research PDF analysis failed: {str(e)}",
                "pdf_path": pdf_path  # Use original path in error case
            })
    
    def _extract_pdf_content(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text and images from PDF using PyMuPDF."""
        doc = fitz.open(pdf_path)
        full_text = ""
        images = []
        image_counter = 0
        
        total_pages = len(doc)
        
        for page_num in range(total_pages):
            page = doc.load_page(page_num)
            
            # Extract text from page
            page_text = page.get_text()
            
            # Find images on page
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                try:
                    # Extract image
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # VALIDATE IMAGE DATA: Check if we have actual image content
                    if not self._is_valid_image_data(image_bytes, image_ext):
                        print(f"Warning: Skipping invalid/missing image data on page {page_num}, image {img_index}")
                        # Add placeholder for missing image instead
                        context = self._extract_image_context(page_text, page_num, img_index)
                        images.append({
                            "image_id": image_counter,
                            "image_path": None,  # No valid image file
                            "page_number": page_num,
                            "context": context,
                            "expected_content": self._infer_expected_content(context),
                            "placeholder": f"[MISSING_IMAGE_{image_counter}_PLACEHOLDER]",
                            "status": "missing_or_invalid"
                        })
                        image_counter += 1
                        continue
                    
                    # Save image to temporary file
                    temp_file = tempfile.NamedTemporaryFile(suffix=f"_image_{image_counter}.{image_ext}", delete=False)
                    temp_file.write(image_bytes)
                    temp_file.close()
                    
                except Exception as e:
                    print(f"Warning: Failed to extract image on page {page_num}, image {img_index}: {e}")
                    # Add placeholder for failed extraction
                    context = self._extract_image_context(page_text, page_num, img_index) 
                    images.append({
                        "image_id": image_counter,
                        "image_path": None,
                        "page_number": page_num,
                        "context": context,
                        "expected_content": self._infer_expected_content(context),
                        "placeholder": f"[FAILED_IMAGE_{image_counter}_PLACEHOLDER]",
                        "status": "extraction_failed"
                    })
                    image_counter += 1
                    continue
                
                # Find context around image in text
                context = self._extract_image_context(page_text, page_num, img_index)
                expected_content = self._infer_expected_content(context)
                
                # Create placeholder in text
                placeholder = f"[IMAGE_{image_counter}_PLACEHOLDER]"
                # Insert placeholder at reasonable location in page text
                if page_text.strip():
                    # Find a good insertion point (after a paragraph or section)
                    lines = page_text.split('\n')
                    for i, line in enumerate(lines):
                        if line.strip() and not line.startswith(' ') and i < len(lines) - 1:
                            lines.insert(i + 1, placeholder)
                            break
                    page_text = '\n'.join(lines)
                else:
                    page_text = placeholder
                
                images.append({
                    "image_id": image_counter,
                    "image_path": temp_file.name,
                    "page_number": page_num,
                    "context": context,
                    "expected_content": expected_content,
                    "placeholder": placeholder
                })
                
                image_counter += 1
            
            full_text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
        
        doc.close()
        
        return {
            "text": full_text,
            "images": images,
            "total_pages": total_pages,
            "total_images": image_counter
        }
    
    def _extract_image_context(self, page_text: str, page_num: int, img_index: int) -> str:
        """Extract contextual text around where an image appears."""
        # Split text into lines and find context
        lines = page_text.split('\n')
        
        # Look for figure references, captions, etc.
        context_lines = []
        
        # Search for figure/table references
        figure_patterns = [
            r'[Ff]igure\s+\d+',
            r'[Tt]able\s+\d+',
            r'[Ff]ig\.\s+\d+',
            r'[Pp]lot\s+\d+',
            r'[Gg]raph\s+\d+',
            r'[Cc]hart\s+\d+'
        ]
        
        for i, line in enumerate(lines):
            for pattern in figure_patterns:
                if re.search(pattern, line):
                    # Include surrounding lines for context
                    start = max(0, i - 2)
                    end = min(len(lines), i + 3)
                    context_lines.extend(lines[start:end])
                    break
        
        # If no specific references found, use general context
        if not context_lines and len(lines) > 5:
            mid = len(lines) // 2
            context_lines = lines[max(0, mid-2):min(len(lines), mid+3)]
        
        return " ".join(context_lines).strip()
    
    def _infer_expected_content(self, context: str) -> str:
        """Infer what type of content the image should contain based on context."""
        context_lower = context.lower()
        
        if any(word in context_lower for word in ['accuracy', 'performance', 'score', 'metric']):
            return "performance_chart"
        elif any(word in context_lower for word in ['loss', 'error', 'training']):
            return "training_curve"
        elif any(word in context_lower for word in ['comparison', 'versus', 'vs', 'compare']):
            return "comparison_plot"
        elif any(word in context_lower for word in ['distribution', 'histogram', 'density']):
            return "distribution_plot"
        elif any(word in context_lower for word in ['table', 'matrix', 'grid']):
            return "table_or_matrix"
        elif any(word in context_lower for word in ['architecture', 'model', 'network']):
            return "architecture_diagram"
        else:
            return "general_scientific_figure"
    
    def _generate_context_questions(self, context: str, expected_content: str) -> List[str]:
        """Generate specific questions about the image based on textual context."""
        base_questions = [
            "What type of visualization is shown in this image?",
            "What are the main data points or values that can be extracted?",
            "Are there any missing elements, broken displays, or quality issues?",
            "Does the image match publication quality standards?"
        ]
        
        # Add content-specific questions
        content_specific = {
            "performance_chart": [
                "What performance metrics are being compared?",
                "What are the specific numerical values shown?",
                "Are error bars or confidence intervals present?",
                "Which method or approach performs best?"
            ],
            "training_curve": [
                "What is being plotted on the x and y axes?",
                "Does the curve show convergence or instability?",
                "Are there multiple curves being compared?",
                "What can be inferred about the training process?"
            ],
            "comparison_plot": [
                "What entities or methods are being compared?",
                "What metric or dimension is used for comparison?",
                "Are the differences statistically significant?",
                "Which approach shows superior performance?"
            ],
            "table_or_matrix": [
                "What data is organized in this table/matrix?",
                "What are the row and column headers?",
                "Are there any notable patterns or trends?",
                "Are all cells properly filled with data?"
            ]
        }
        
        questions = base_questions.copy()
        if expected_content in content_specific:
            questions.extend(content_specific[expected_content])
        
        # Add context-specific questions
        if context:
            questions.append(f"Based on this context: '{context[:200]}...', does the image content match the expectation?")
        
        return questions
    
    def _generate_research_questions(self, context: str, expected_content: str, analysis_focus: str) -> List[str]:
        """Generate research-focused questions about images for paper analysis."""
        base_questions = [
            "What type of scientific visualization or data is presented?",
            "What are the main experimental results or findings shown?",
            "What research claims or hypotheses does this support?",
            "What numerical values, trends, or patterns are visible?"
        ]
        
        # Handle both old and new focus names for backward compatibility
        if analysis_focus in ["content", "image_content"]:
            analysis_focus = "image_content"
        elif analysis_focus in ["trends", "image_trends"]:
            analysis_focus = "image_trends"
        elif analysis_focus in ["technical", "image_technical"]:
            analysis_focus = "image_technical"
        
        # Focus-specific questions
        focus_questions = {
            "image_content": [
                "What scientific insights can be extracted from this figure?",
                "How does this relate to the research problem being solved?",
                "What evidence does this provide for the paper's claims?",
                "What are the key takeaways for understanding the research?"
            ],
            "image_trends": [
                "What data patterns or trends are evident?",
                "How do different conditions or methods compare?",
                "Are there any surprising or counterintuitive results?",
                "What does the progression or relationship show?"
            ],
            "image_technical": [
                "What methodology or experimental setup is illustrated?",
                "What technical details about the approach are revealed?",
                "Are there statistical significance indicators?",
                "What parameters or hyperparameters are being varied?"
            ],
            "pdf_reading": [
                "What is the main contribution illustrated by this figure?",
                "How does this figure support the paper's thesis?",
                "What experimental evidence is provided?",
                "What can be learned about the proposed method's performance?"
            ]
        }
        
        questions = base_questions.copy()
        if analysis_focus in focus_questions:
            questions.extend(focus_questions[analysis_focus])
        else:
            questions.extend(focus_questions["pdf_reading"])  # Default to PDF reading questions
        
        # Add context-specific questions
        if context:
            questions.append(f"Given this context from the paper: '{context[:300]}...', what specific insights does this figure provide?")
        
        return questions
    
    def _extract_research_insights(self, full_text: str, image_analyses: List[Dict], analysis_focus: str) -> Dict[str, Any]:
        """Extract high-level research insights from text and image analyses."""
        insights = {
            "paper_summary": "",
            "key_findings": [],
            "methodology": "",
            "experimental_results": [],
            "limitations": [],
            "contributions": []
        }
        
        # Basic text analysis to extract insights
        text_lower = full_text.lower()
        
        # Extract key sections (basic heuristic approach)
        if "abstract" in text_lower:
            abstract_start = text_lower.find("abstract")
            abstract_end = min(text_lower.find("introduction", abstract_start), 
                             text_lower.find("1.", abstract_start) if text_lower.find("1.", abstract_start) != -1 else len(full_text))
            if abstract_end > abstract_start:
                insights["paper_summary"] = full_text[abstract_start:abstract_end][:500]
        
        # Extract contributions from conclusion or abstract
        contribution_keywords = ["contribution", "propose", "present", "novel", "new method"]
        for keyword in contribution_keywords:
            if keyword in text_lower:
                # Find sentences containing contribution keywords
                sentences = full_text.split('. ')
                for sentence in sentences:
                    if keyword in sentence.lower() and len(sentence) < 200:
                        insights["contributions"].append(sentence.strip())
        
        # Analyze figures for experimental insights
        for img_analysis in image_analyses:
            if img_analysis.get("research_analysis"):
                analysis_text = str(img_analysis["research_analysis"])
                if "performance" in analysis_text.lower() or "result" in analysis_text.lower():
                    insights["experimental_results"].append({
                        "figure_context": img_analysis["context"][:100],
                        "findings": analysis_text[:300]
                    })
        
        return insights
    
    def _analyze_image_with_questions(self, image_path: str, questions: List[str]) -> Dict[str, Any]:
        """Use VLM to analyze image with specific questions."""
        # Handle missing/invalid images
        if image_path is None:
            return {
                "status": "missing_image",
                "response": "IMAGE NOT FOUND: This appears to be a missing or invalid image placeholder. The PDF likely references an image file that was not available during compilation (e.g., missing image file when running pdflatex). This results in a placeholder or broken image reference in the final PDF.",
                "questions_asked": len(questions),
                "missing_image": True
            }
        
        try:
            client, model = create_vlm_client(self.vlm_model)
            
            # Format questions into a structured prompt
            questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
            
            prompt = f"""Analyze this scientific image and answer the following specific questions:

{questions_text}

Please provide detailed, specific answers to each question. If any issues are detected (missing data, poor quality, broken elements), describe them clearly."""
            
            system_message = """You are an expert scientific figure analyst. Provide precise, detailed answers to specific questions about research figures. Focus on extracting concrete data and identifying any quality or content issues."""
            
            response, _ = get_response_from_vlm(
                prompt=prompt,
                images=[image_path],
                client=client,
                model=model,
                system_message=system_message,
                print_debug=False
            )
            
            return {
                "status": "success",
                "response": response,
                "questions_asked": len(questions)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "response": f"Failed to analyze image: {str(e)}"
            }
    
    def _reconstruct_document_with_analysis(self, original_text: str, image_analyses: List[Dict]) -> str:
        """Reconstruct document by replacing placeholders with VLM analysis."""
        reconstructed_text = original_text
        
        for analysis in image_analyses:
            placeholder = analysis.get("placeholder", f"[IMAGE_{analysis['image_id']}_PLACEHOLDER]")
            
            # Create detailed description from VLM analysis
            vlm_response = analysis["vlm_analysis"].get("response", "Analysis failed")
            context = analysis.get("context", "")
            
            replacement_text = f"""
[IMAGE {analysis['image_id']} ANALYSIS]
Context: {context}
VLM Analysis: {vlm_response}
[END IMAGE {analysis['image_id']} ANALYSIS]
"""
            
            reconstructed_text = reconstructed_text.replace(placeholder, replacement_text)
        
        return reconstructed_text
    
    def _identify_publication_issues(self, extracted_data: Dict, image_analyses: List[Dict]) -> List[str]:
        """Identify publication quality issues from the analysis."""
        issues = []
        
        # Check for missing images
        if extracted_data["total_images"] == 0:
            issues.append("No images found in PDF - figures may not be displaying")
        
        # Check each image analysis for problems
        for analysis in image_analyses:
            vlm_analysis = analysis["vlm_analysis"]
            vlm_response = vlm_analysis.get("response", "").lower()
            
            # Check for missing images first
            if vlm_analysis.get("status") == "missing_image" or vlm_analysis.get("missing_image"):
                issues.append(f"Image {analysis['image_id']}: MISSING IMAGE - PDF contains placeholder for image that was not available during compilation")
            elif any(phrase in vlm_response for phrase in ["missing", "empty", "blank", "not visible", "broken"]):
                issues.append(f"Image {analysis['image_id']}: Potential display or content issues detected")
            
            if any(phrase in vlm_response for phrase in ["low quality", "poor resolution", "unclear", "blurry"]):
                issues.append(f"Image {analysis['image_id']}: Quality issues detected")
        
        # Check text for citation issues
        text = extracted_data["text"]
        question_marks = text.count("?")
        
        if question_marks > 3:  # More than normal questioning sentences
            issues.append(f"Multiple question marks detected ({question_marks}) - likely missing citations")
        
        # Check for specific citation patterns with question marks
        citation_patterns = [r'\[\?\]', r'\(\?\)', r'\\cite\{\?\}', r'\\ref\{\?\}']
        for pattern in citation_patterns:
            if re.search(pattern, text):
                issues.append("Missing citation references detected (citation commands with ? marks)")
        
        return issues
    
    def _get_analysis_prompt(self, focus: str, num_images: int) -> str:
        """Generate appropriate analysis prompt based on focus and number of images."""
        
        base_instruction = f"""You are analyzing {num_images} scientific figure(s) from an AI/ML research paper. """
        
        if num_images > 1:
            base_instruction += "Compare and contrast the figures, noting relationships between them. "
        
        # Handle both old names (for backward compatibility) and new clear names
        if focus in ["content", "image_content"]:
            focus = "image_content"
        elif focus in ["quality", "image_quality"]:
            focus = "image_quality"
        elif focus in ["trends", "image_trends"]:
            focus = "image_trends"
        elif focus in ["technical", "image_technical"]:
            focus = "image_technical"
            
        focus_instructions = {
            "image_content": """
            Focus primarily on:
            1. **Content Description**: What data, results, or concepts are presented?
            2. **Key Findings**: What are the main insights, trends, or conclusions?
            3. **Data Interpretation**: What do the values, patterns, and relationships indicate?
            """,
            
            "image_quality": """
            Focus primarily on:
            1. **Visual Quality**: Clarity, readability, and aesthetic appeal
            2. **Technical Standards**: Proper axes, labels, legends, and annotations
            3. **Publication Readiness**: Meets scientific publication standards?
            4. **Improvement Suggestions**: How could the figure be enhanced?
            """,
            
            "image_trends": """
            Focus primarily on:
            1. **Pattern Identification**: What trends, patterns, or relationships are visible?
            2. **Comparative Analysis**: How do different conditions/methods compare?
            3. **Statistical Insights**: What do the distributions, correlations, or progressions show?
            4. **Experimental Outcomes**: What do the results suggest about the research hypothesis?
            """,
            
            "image_technical": """
            Focus primarily on:
            1. **Technical Elements**: Axes scales, units, labels, legends, annotations
            2. **Methodology Indicators**: What experimental setup or analysis method is shown?
            3. **Data Presentation**: How is the data organized, scaled, and presented?
            4. **Figure Construction**: Layout, subplots, color schemes, line styles
            """,
            
            "pdf_validation": """
            Focus on PDF document visual validation:
            1. **Layout Problems**: Are formulas, text, or figures cut off or extending beyond page boundaries?
            2. **Missing Citations**: Are there "?" marks where citations should appear (indicating missing bibliography)?
            3. **Missing Figures**: Are there empty spaces, broken image placeholders, or missing figure content?
            4. **Structural Issues**: Are there duplicate sections, malformed layouts, or spacing problems?
            5. **Text Quality**: Is text properly formatted with correct line spacing and margins?
            6. **Overall Presentation**: Does the document meet publication quality standards?
            
            For each issue found, provide specific details about location and severity.
            """,
            
            "comprehensive": """
            Provide a comprehensive analysis covering:
            1. **Content Type**: Figure/document type and what data/results are shown
            2. **Key Findings**: Main insights, trends, and scientific conclusions (for figures)
            3. **Technical Assessment**: Quality of axes, labels, legends, and overall clarity
            4. **Visual Validation**: Layout problems, missing elements, formatting issues (for PDFs)
            5. **Scientific Value**: What this contributes to the research narrative
            6. **Publication Quality**: Readiness for publication and potential improvements
            """
        }
        
        # Special handling for PDF validation
        if focus == "pdf_validation":
            return base_instruction + focus_instructions["pdf_validation"] + """
            
            **CRITICAL PDF VALIDATION INSTRUCTIONS**:
            - Examine EVERY page of the PDF carefully
            - Look for "?" symbols where citations should appear (indicates missing bibliography)
            - Check for empty spaces or broken image placeholders where figures should be
            - Identify any text, formulas, or figures that extend beyond page margins
            - Note any duplicate sections or repeated content blocks
            - Assess overall layout quality and formatting consistency
            - Report ALL issues found with specific page numbers and locations
            
            Provide a PASS/FAIL assessment for publication readiness.
            """
        
        return base_instruction + focus_instructions.get(focus, focus_instructions["comprehensive"]) + """
        
        Provide your analysis in a clear, structured format with specific observations and actionable insights.
        Be precise about what you observe and avoid speculation beyond what's directly visible.
        """
    
    def _structure_analysis(self, raw_response: str, image_paths: List[str], focus: str) -> Dict[str, Any]:
        """Structure the VLM response into a standardized format."""
        
        result = {
            "analysis_type": "vlm_document_analysis",
            "focus": focus,
            "file_count": len(image_paths),
            "file_paths": image_paths,
            "detailed_analysis": raw_response,
            "metadata": {
                "model_used": self.vlm_model,
                "analysis_timestamp": None,  # Could add timestamp if needed
                "character_count": len(raw_response)
            }
        }
        
        # Special processing for PDF validation
        if focus == "pdf_validation":
            validation_results = self._extract_pdf_validation_results(raw_response)
            result["pdf_validation"] = validation_results
        
        # Try to extract key sections from the response for structured access
        try:
            sections = self._extract_analysis_sections(raw_response)
            if sections:
                result["structured_sections"] = sections
        except Exception:
            # If structuring fails, keep the raw response
            pass
        
        return result
    
    def _extract_analysis_sections(self, response: str) -> Optional[Dict[str, str]]:
        """Attempt to extract structured sections from the VLM response."""
        sections = {}
        
        # Look for common section headers
        import re
        
        section_patterns = [
            (r"(?:^|\n)\*\*Figure Type[:\s]*\*\*([^\n]*(?:\n(?!\*\*)[^\n]*)*)", "figure_type"),
            (r"(?:^|\n)\*\*Content[:\s]*\*\*([^\n]*(?:\n(?!\*\*)[^\n]*)*)", "content"),
            (r"(?:^|\n)\*\*Key Findings[:\s]*\*\*([^\n]*(?:\n(?!\*\*)[^\n]*)*)", "key_findings"),
            (r"(?:^|\n)\*\*Technical[:\s]*\*\*([^\n]*(?:\n(?!\*\*)[^\n]*)*)", "technical"),
            (r"(?:^|\n)\*\*Quality[:\s]*\*\*([^\n]*(?:\n(?!\*\*)[^\n]*)*)", "quality"),
            (r"(?:^|\n)#{1,3}\s*([^#\n]+)", "headings")
        ]
        
        for pattern, key in section_patterns[:-1]:  # Exclude headings for now
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                sections[key] = match.group(1).strip()
        
        return sections if sections else None
    
    def _extract_pdf_validation_results(self, response: str) -> Dict[str, Any]:
        """Extract PDF validation results from VLM response."""
        validation = {
            "overall_assessment": "UNKNOWN",
            "layout_issues": [],
            "missing_citations": [],
            "missing_figures": [],
            "structural_problems": [],
            "publication_ready": False,
            "critical_issues_found": False
        }
        
        response_lower = response.lower()
        
        # Determine overall assessment
        if "pass" in response_lower and "fail" not in response_lower:
            validation["overall_assessment"] = "PASS"
            validation["publication_ready"] = True
        elif "fail" in response_lower:
            validation["overall_assessment"] = "FAIL"
            validation["publication_ready"] = False
        
        # Extract specific issues using pattern matching
        import re
        
        # Look for missing citations ("?" marks)
        if "?" in response or "missing citation" in response_lower or "undefined citation" in response_lower:
            citation_matches = re.findall(r'citation[^.]*?\?[^.]*', response, re.IGNORECASE)
            validation["missing_citations"].extend(citation_matches)
        
        # Look for missing figures
        figure_patterns = [
            r'missing figure[^.]*',
            r'empty.*?figure[^.]*',
            r'figure.*?not.*?found[^.]*',
            r'broken.*?image[^.]*'
        ]
        for pattern in figure_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            validation["missing_figures"].extend(matches)
        
        # Look for layout issues
        layout_patterns = [
            r'formula.*?extend[^.]*',
            r'text.*?overflow[^.]*',
            r'margin[^.]*problem[^.]*',
            r'page.*?boundary[^.]*'
        ]
        for pattern in layout_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            validation["layout_issues"].extend(matches)
        
        # Look for structural problems
        structure_patterns = [
            r'duplicate.*?section[^.]*',
            r'repeated.*?content[^.]*',
            r'malformed[^.]*',
            r'structural.*?issue[^.]*'
        ]
        for pattern in structure_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            validation["structural_problems"].extend(matches)
        
        # Determine if critical issues found
        total_issues = (
            len(validation["layout_issues"]) + 
            len(validation["missing_citations"]) + 
            len(validation["missing_figures"]) + 
            len(validation["structural_problems"])
        )
        validation["critical_issues_found"] = total_issues > 0
        
        return validation
    
    def _is_valid_image_data(self, image_bytes: bytes, image_ext: str) -> bool:
        """
        Validate that extracted image data represents a real image.
        
        Args:
            image_bytes: Raw image data extracted from PDF
            image_ext: File extension (e.g., 'png', 'jpg')
            
        Returns:
            True if the image data appears to be valid, False otherwise
        """
        if not image_bytes or len(image_bytes) < 100:
            # Very small files are likely placeholders or broken
            return False
            
        # Check for common image file signatures
        image_signatures = {
            'png': [b'\x89PNG\r\n\x1a\n'],
            'jpg': [b'\xff\xd8\xff', b'\xff\xd8'],
            'jpeg': [b'\xff\xd8\xff', b'\xff\xd8'], 
            'gif': [b'GIF87a', b'GIF89a'],
            'bmp': [b'BM'],
            'tiff': [b'II*\x00', b'MM\x00*'],
            'webp': [b'RIFF']
        }
        
        ext_lower = image_ext.lower()
        if ext_lower in image_signatures:
            signatures = image_signatures[ext_lower]
            for sig in signatures:
                if image_bytes.startswith(sig):
                    return True
            return False
        
        # For unknown extensions, just check if we have reasonable data
        return len(image_bytes) > 1000  # Assume files > 1KB are likely real images
    
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
                    f"Example: Use 'paper_workspace/final_paper.pdf' instead of the full path."
                )
        else:
            # Relative path - join with workspace
            abs_path = os.path.abspath(os.path.join(abs_working_dir, path))
            
            # For VLM analysis, files may not exist yet (being analyzed for creation)
            # Only check if file should exist (for read operations)
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
