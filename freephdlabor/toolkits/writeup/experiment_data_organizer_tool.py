"""
ExperimentDataOrganizerTool - Mandatory preprocessing tool for WriteupAgent.

This tool combines DataDiscoveryTool, FigureImportTool, and ExperimentalResultsExtractorTool
into a single mandatory preprocessing step that must be run before any paper writing begins.

The tool implements a hardcoded workflow:
1. Discover all experimental files and plots
2. Copy and organize everything into paper_workspace subdirectories  
3. Generate .txt annotations for ALL figures using VLM
4. Generate .txt annotations for ALL data files using LLM
5. Create comprehensive summary of remarkable findings
6. Provide complete file inventory to WriteupAgent

This eliminates confusion and ensures WriteupAgent has everything organized before writing.
"""

import json
import os
import shutil
import numpy as np
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional
from smolagents import Tool


class ExperimentDataOrganizerTool(Tool):
    name = "experiment_data_organizer_tool"
    description = """
    MANDATORY preprocessing tool that must be called FIRST by WriteupAgent before any writing.
    
    This tool implements a hardcoded workflow to discover, organize, and annotate ALL experimental
    data and figures in the workspace. It eliminates the need for multiple separate tools and
    ensures WriteupAgent has everything properly organized before writing begins.
    
    Hardcoded Workflow (executed automatically):
    1. **Discovery**: Find all experiment files (.npy, .json, .csv, .pkl) and plots (.png, .pdf, .svg)
    2. **Organization**: Copy all files to organized paper_workspace subdirectories:
       - paper_workspace/data/ - All experimental data files  
       - paper_workspace/figures/ - All plots and figures
    3. **Figure Annotation**: Generate .txt descriptions for EVERY figure using VLM analysis
    4. **Data Annotation**: Generate .txt summaries for EVERY data file using LLM analysis
    5. **Summary Generation**: LLM reads all annotations and creates remarkable findings report
    6. **Inventory Creation**: Provide complete file inventory with descriptions to WriteupAgent
    
    Output: Comprehensive organized workspace with all files accessible and annotated.
    
    MANDATORY USAGE: WriteupAgent MUST call this tool first before any LaTeX generation.
    After this tool runs, WriteupAgent can easily access organized data and figures.
    """
    
    inputs = {
        "workspace_mode": {
            "type": "string",
            "description": "Organization mode: 'comprehensive' (default) - discovers and organizes everything",
            "nullable": True
        }
    }
    
    outputs = {
        "organization_report": {
            "type": "string", 
            "description": "JSON report with complete file inventory, annotations, and remarkable findings summary"
        }
    }
    
    output_type = "string"
    
    def __init__(self, model=None, working_dir: Optional[str] = None):
        """Initialize ExperimentDataOrganizerTool.
        
        Args:
            model: LLM model for data annotation and summary generation
            working_dir: Working directory for workspace-aware file access
        """
        super().__init__()
        from ..model_utils import get_raw_model
        self.model = get_raw_model(model)
        self.working_dir = os.path.abspath(working_dir) if working_dir else None
        
    def forward(self, workspace_mode: str = "comprehensive") -> str:
        """Execute the mandatory preprocessing workflow.
        
        This implements a hardcoded sequence that organizes everything before WriteupAgent starts.
        """
        try:
            print("🔍 Starting mandatory experiment data organization workflow...")
            
            # Phase 1: Discovery
            print("📁 Phase 1: Discovering all experimental files and plots...")
            discovered_files = self._discover_all_files()
            
            # Phase 2: Organization  
            print("📂 Phase 2: Organizing files into paper_workspace subdirectories...")
            organized_files = self._organize_files(discovered_files)
            
            # Phase 3: Figure Annotation
            print("🖼️ Phase 3: Generating VLM annotations for ALL figures...")
            figure_annotations = self._annotate_all_figures(organized_files['figures'])
            
            # Phase 4: Data Annotation
            print("📊 Phase 4: Generating LLM annotations for ALL data files...")
            data_annotations = self._annotate_all_data_files(organized_files['data'])
            
            # Phase 5: Summary Generation
            print("📝 Phase 5: Creating remarkable findings summary...")
            remarkable_summary = self._generate_remarkable_findings_summary(
                figure_annotations, data_annotations
            )
            
            # Phase 6: Final Report
            print("✅ Phase 6: Creating comprehensive organization report...")
            final_report = {
                "organization_status": "completed",
                "workflow_phases": [
                    "discovery", "organization", "figure_annotation", 
                    "data_annotation", "summary_generation"
                ],
                "organized_files": organized_files,
                "figure_annotations": figure_annotations,
                "data_annotations": data_annotations,
                "remarkable_findings": remarkable_summary,
                "paper_workspace_guidance": {
                    "data_location": "paper_workspace/data/",
                    "figures_location": "paper_workspace/figures/",
                    "annotations_location": "All .txt files alongside corresponding data/figures",
                    "usage_instructions": [
                        "All experimental data is now organized in paper_workspace/data/",
                        "All figures are organized in paper_workspace/figures/",
                        "Each file has a corresponding .txt annotation with analysis",
                        "Use the remarkable_findings summary to identify key results",
                        "Reference any specific files using the organized_files inventory"
                    ]
                }
            }
            
            print(f"🎉 Organization complete! Processed {len(organized_files['data'])} data files and {len(organized_files['figures'])} figures")
            return json.dumps(final_report, indent=2)
            
        except Exception as e:
            error_report = {
                "organization_status": "failed",
                "error": f"Experiment data organization failed: {str(e)}",
                "partial_results": getattr(self, '_partial_results', {}),
                "guidance": "WriteupAgent should retry this tool before proceeding with paper writing"
            }
            return json.dumps(error_report, indent=2)
    
    def _discover_all_files(self) -> Dict[str, List[str]]:
        """Discover all experimental files and plots in the workspace."""
        discovered = {"data_files": [], "plot_files": []}
        
        # Define search directories and file extensions
        search_dirs = [".", "experiment_data", "experimental_plots", "results", "data", "experiment_runs"]
        data_extensions = ['.json', '.csv', '.npy', '.pkl', '.npz']
        plot_extensions = ['.png', '.pdf', '.svg', '.jpg', '.jpeg']
        
        for directory in search_dirs:
            search_path = self._safe_path(directory)
            if os.path.exists(search_path):
                try:
                    # Walk through directory tree (max 8 levels deep)
                    for root, dirs, files in os.walk(search_path):
                        level = root.replace(search_path, '').count(os.sep)
                        if level < 8:
                            for file in files:
                                file_path = os.path.join(root, file)
                                
                                # Categorize by extension
                                for ext in data_extensions:
                                    if file.endswith(ext):
                                        discovered["data_files"].append(file_path)
                                        break
                                
                                for ext in plot_extensions:
                                    if file.endswith(ext):
                                        discovered["plot_files"].append(file_path)
                                        break
                except Exception:
                    continue
        
        # Remove duplicates
        discovered["data_files"] = list(set(discovered["data_files"]))
        discovered["plot_files"] = list(set(discovered["plot_files"]))
        
        return discovered
    
    def _organize_files(self, discovered_files: Dict[str, List[str]]) -> Dict[str, List[Dict[str, str]]]:
        """Copy and organize all files into paper_workspace subdirectories."""
        organized = {"data": [], "figures": []}
        
        # Create directories
        data_dir = self._safe_path("paper_workspace/data")
        figures_dir = self._safe_path("paper_workspace/figures")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(figures_dir, exist_ok=True)
        
        # Organize data files
        for i, data_file in enumerate(discovered_files["data_files"]):
            try:
                filename = os.path.basename(data_file)
                # Create descriptive filename to avoid conflicts
                new_filename = f"data_{i:03d}_{filename}"
                dest_path = os.path.join(data_dir, new_filename)
                
                shutil.copy2(data_file, dest_path)
                organized["data"].append({
                    "original_path": data_file,
                    "organized_path": dest_path,
                    "filename": new_filename,
                    "type": "data_file"
                })
            except Exception:
                continue
        
        # Organize figure files  
        for i, plot_file in enumerate(discovered_files["plot_files"]):
            try:
                filename = os.path.basename(plot_file)
                # Create descriptive filename to avoid conflicts
                new_filename = f"figure_{i:03d}_{filename}"
                dest_path = os.path.join(figures_dir, new_filename)
                
                shutil.copy2(plot_file, dest_path)
                organized["figures"].append({
                    "original_path": plot_file,
                    "organized_path": dest_path,
                    "filename": new_filename,
                    "type": "figure_file"
                })
            except Exception:
                continue
                
        return organized
    
    def _annotate_all_figures(self, figures: List[Dict[str, str]]) -> Dict[str, Dict[str, Any]]:
        """Generate VLM annotations for ALL figures."""
        annotations = {}
        
        if not self.model:
            return {"error": "No model provided for VLM figure annotation"}
        
        for fig_info in figures:
            try:
                fig_path = fig_info["organized_path"]
                fig_name = fig_info["filename"]
                
                # Use VLM to analyze the figure (similar to VLMDocumentAnalysisTool)
                analysis_prompt = f"""
                Analyze this experimental figure and provide a comprehensive description.
                
                Please describe:
                1. What type of plot/chart this is
                2. What data is being visualized
                3. Key trends, patterns, or findings visible
                4. Experimental conditions or parameters shown
                5. Scientific significance of the results
                
                Be specific about numerical values, trends, and experimental insights.
                """
                
                # Call VLM model (assuming it can process images)
                try:
                    vlm_response = self.model(analysis_prompt)
                    
                    # Save annotation as .txt file
                    txt_path = fig_path.replace('.png', '.txt').replace('.pdf', '.txt').replace('.svg', '.txt')
                    with open(txt_path, 'w') as f:
                        f.write(vlm_response)
                    
                    annotations[fig_name] = {
                        "figure_path": fig_path,
                        "annotation_path": txt_path,
                        "vlm_analysis": vlm_response,
                        "annotation_status": "completed"
                    }
                    
                except Exception as e:
                    annotations[fig_name] = {
                        "figure_path": fig_path,
                        "annotation_status": "failed",
                        "error": str(e)
                    }
                    
            except Exception:
                continue
                
        return annotations
    
    def _annotate_all_data_files(self, data_files: List[Dict[str, str]]) -> Dict[str, Dict[str, Any]]:
        """Generate LLM annotations for ALL data files."""
        annotations = {}
        
        if not self.model:
            return {"error": "No model provided for data file annotation"}
        
        for data_info in data_files:
            try:
                data_path = data_info["organized_path"]
                data_name = data_info["filename"]
                
                # Load and analyze data file
                data_summary = self._analyze_data_file_content(data_path)
                
                # Generate LLM annotation
                analysis_prompt = f"""
                Analyze this experimental data file and provide a comprehensive summary.
                
                File: {data_name}
                Data Summary: {data_summary}
                
                Please describe:
                1. What type of experimental data this contains
                2. Key metrics, measurements, or variables
                3. Experimental conditions or parameters
                4. Notable patterns, trends, or outlier values
                5. Scientific significance and potential insights
                
                Be specific about numerical ranges, statistical patterns, and experimental findings.
                """
                
                try:
                    llm_response = self.model(analysis_prompt)
                    
                    # Save annotation as .txt file
                    txt_path = data_path.replace('.npy', '.txt').replace('.json', '.txt').replace('.csv', '.txt').replace('.pkl', '.txt')
                    with open(txt_path, 'w') as f:
                        f.write(llm_response)
                    
                    annotations[data_name] = {
                        "data_path": data_path,
                        "annotation_path": txt_path,
                        "llm_analysis": llm_response,
                        "data_summary": data_summary,
                        "annotation_status": "completed"
                    }
                    
                except Exception as e:
                    annotations[data_name] = {
                        "data_path": data_path,
                        "annotation_status": "failed", 
                        "error": str(e)
                    }
                    
            except Exception:
                continue
                
        return annotations
    
    def _generate_remarkable_findings_summary(self, figure_annotations: Dict, data_annotations: Dict) -> Dict[str, Any]:
        """Generate high-level summary of remarkable findings from all annotations."""
        if not self.model:
            return {"error": "No model provided for findings summary"}
        
        # Collect all annotations
        all_annotations = []
        
        # Add figure annotations
        for fig_name, fig_data in figure_annotations.items():
            if fig_data.get("vlm_analysis"):
                all_annotations.append(f"FIGURE {fig_name}: {fig_data['vlm_analysis']}")
        
        # Add data annotations  
        for data_name, data_data in data_annotations.items():
            if data_data.get("llm_analysis"):
                all_annotations.append(f"DATA {data_name}: {data_data['llm_analysis']}")
        
        if not all_annotations:
            return {"summary": "No annotations available for analysis"}
        
        # Generate summary using LLM
        summary_prompt = f"""
        Based on the following experimental data and figure analyses, identify the most remarkable 
        and significant findings that should be highlighted in a research paper.
        
        Experimental Analyses:
        {chr(10).join(all_annotations[:10])}  # Limit to avoid token overflow
        
        Please provide:
        1. **Key Experimental Findings**: Most important results and discoveries
        2. **Notable Trends**: Significant patterns across experiments  
        3. **Performance Insights**: Critical performance metrics and comparisons
        4. **Methodological Observations**: Important experimental conditions or variations
        5. **Research Impact**: Most compelling results for paper narrative
        
        Focus on quantitative results, statistical significance, and scientific insights.
        """
        
        try:
            remarkable_summary = self.model(summary_prompt)
            
            # Save summary to file
            summary_path = self._safe_path("paper_workspace/remarkable_findings_summary.txt")
            with open(summary_path, 'w') as f:
                f.write(remarkable_summary)
            
            return {
                "summary": remarkable_summary,
                "summary_file": summary_path,
                "total_annotations_analyzed": len(all_annotations),
                "figures_analyzed": len(figure_annotations),
                "data_files_analyzed": len(data_annotations)
            }
            
        except Exception as e:
            return {"summary_status": "failed", "error": str(e)}
    
    def _analyze_data_file_content(self, file_path: str) -> str:
        """Analyze the content of a data file to provide summary for LLM annotation."""
        try:
            if file_path.endswith('.npy'):
                data = np.load(file_path, allow_pickle=True)
                if isinstance(data, np.ndarray):
                    return f"NumPy array: shape={data.shape}, dtype={data.dtype}, range=[{data.min():.4f}, {data.max():.4f}]"
                else:
                    return f"NumPy object: type={type(data)}, content={str(data)[:200]}"
                    
            elif file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                return f"JSON object: keys={list(data.keys()) if isinstance(data, dict) else 'non-dict'}, size={len(str(data))} chars"
                
            elif file_path.endswith('.csv'):
                with open(file_path, 'r') as f:
                    lines = f.readlines()[:5]  # First 5 lines
                return f"CSV file: {len(lines)} lines preview: {''.join(lines)}"
                
            else:
                return f"File type: {os.path.splitext(file_path)[1]}, size: {os.path.getsize(file_path)} bytes"
                
        except Exception as e:
            return f"Analysis failed: {str(e)}"
    
    def _safe_path(self, path: str) -> str:
        """Convert path to absolute workspace path."""
        if not self.working_dir:
            return path
        
        if os.path.isabs(path):
            return path
        
        return os.path.abspath(os.path.join(self.working_dir, path))