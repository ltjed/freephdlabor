"""
FigureImportTool - Import existing experimental plots for paper inclusion.

This tool helps WriteupAgent discover and import relevant experimental plots
from AI-Scientist runs into the paper's figures directory.
"""

import json
import os
import shutil
from typing import List, Dict, Any, Optional
from smolagents import Tool


class FigureImportTool(Tool):
    name = "figure_import_tool"
    description = """
    Import existing experimental plots into the paper's figures directory.
    
    This tool works with DataDiscoveryTool to:
    - Select the most relevant experimental plots based on categories
    - Copy them to writeup_agent/figures/ with descriptive names
    - Generate metadata for VLM analysis
    - Provide integration guidance for LaTeX inclusion
    
    Use this tool after DataDiscoveryTool identifies available plots.
    It helps bridge the gap between experimental results and paper figures.
    
    Input: List of plot file paths from DataDiscoveryTool
    Output: Status of imported figures with integration recommendations
    """
    
    inputs = {
        "plot_files": {
            "type": "string",
            "description": "JSON string containing list of plot file paths to import (from DataDiscoveryTool output)"
        },
        "selection_criteria": {
            "type": "string", 
            "description": "Selection criteria: 'training_curves', 'comparisons', 'ablations', 'all', or 'auto' (default: 'auto')",
            "nullable": True
        },
        "max_figures": {
            "type": "integer",
            "description": "Maximum number of figures to import (default: 6)",
            "nullable": True
        }
    }
    
    outputs = {
        "import_result": {
            "type": "string",
            "description": "JSON result with imported figure details and integration guidance"
        }
    }
    
    output_type = "string"
    
    def __init__(self, working_dir: Optional[str] = None):
        """Initialize FigureImportTool.
        
        Args:
            working_dir: Working directory for workspace-aware operations
        """
        super().__init__()
        self.working_dir = os.path.abspath(working_dir) if working_dir else None
        
    def forward(self, plot_files: str, selection_criteria: str = "auto", max_figures: int = 6) -> str:
        """
        Import experimental plots into the paper figures directory.
        
        Args:
            plot_files: JSON string with plot file paths
            selection_criteria: How to select plots
            max_figures: Maximum figures to import
            
        Returns:
            JSON string with import results and recommendations
        """
        try:
            # Parse input plot files
            if isinstance(plot_files, str):
                plot_list = json.loads(plot_files)
            else:
                plot_list = plot_files
                
            # Create figures directory
            figures_dir = os.path.join(self.working_dir or ".", "writeup_agent", "figures")
            os.makedirs(figures_dir, exist_ok=True)
                
            # Select plots based on criteria
            selected_plots = self._select_plots(plot_list, selection_criteria, max_figures)
            
            # Import selected plots
            imported_figures = []
            for i, plot_info in enumerate(selected_plots):
                try:
                    imported_fig = self._import_plot(plot_info, figures_dir, i)
                    if imported_fig:
                        imported_figures.append(imported_fig)
                except Exception as e:
                    print(f"Warning: Failed to import {plot_info.get('file_path', 'unknown')}: {e}")
                    continue
            
            # Generate integration recommendations
            recommendations = self._generate_integration_recommendations(imported_figures)
            
            result = {
                "import_summary": {
                    "total_plots_available": len(plot_list),
                    "plots_selected": len(selected_plots),
                    "successfully_imported": len(imported_figures),
                    "figures_directory": figures_dir
                },
                "imported_figures": imported_figures,
                "integration_recommendations": recommendations,
                "next_steps": [
                    "Use VLMDocumentAnalysisTool to generate descriptions for each imported figure",
                    "Reference figures in LaTeX using \\includegraphics{figures/filename.png}",
                    "Create multi-panel layouts with MultiPanelCompositionTool if needed"
                ]
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            error_result = {
                "error": f"Figure import failed: {str(e)}",
                "imported_figures": [],
                "recommendations": []
            }
            return json.dumps(error_result, indent=2)
    
    def _select_plots(self, plot_list: List[Dict], criteria: str, max_figures: int) -> List[Dict]:
        """Select most relevant plots based on criteria."""
        
        if criteria == "auto":
            # Smart selection: prioritize diverse, high-value plots
            priority_categories = ["training_analysis", "method_comparison", "experimental_results"]
            selected = []
            
            for category in priority_categories:
                category_plots = [p for p in plot_list if p.get("plot_category") == category]
                # Take best representatives from each category
                selected.extend(category_plots[:max_figures//len(priority_categories) + 1])
                
            return selected[:max_figures]
            
        elif criteria == "training_curves":
            return [p for p in plot_list if p.get("plot_category") == "training_analysis"][:max_figures]
            
        elif criteria == "comparisons":
            return [p for p in plot_list if p.get("plot_category") == "method_comparison"][:max_figures]
            
        elif criteria == "ablations":
            return [p for p in plot_list if "ablation" in p.get("file_path", "").lower()][:max_figures]
            
        elif criteria == "all":
            return plot_list[:max_figures]
            
        else:
            return plot_list[:max_figures]
    
    def _import_plot(self, plot_info: Dict, figures_dir: str, index: int) -> Optional[Dict]:
        """Import a single plot file."""
        try:
            source_path = plot_info.get("file_path")
            if not source_path or not os.path.exists(source_path):
                return None
                
            # Generate descriptive filename
            original_name = os.path.basename(source_path)
            category = plot_info.get("plot_category", "figure")
            new_name = f"{category}_{index+1}_{original_name}"
            
            dest_path = os.path.join(figures_dir, new_name)
            
            # Copy file
            shutil.copy2(source_path, dest_path)
            
            return {
                "original_path": source_path,
                "imported_path": dest_path,
                "filename": new_name,
                "category": plot_info.get("plot_category"),
                "experimental_context": plot_info.get("experimental_context", {}),
                "latex_reference": f"figures/{new_name}",
                "suggested_caption_context": self._extract_caption_context(plot_info)
            }
            
        except Exception as e:
            print(f"Failed to import plot: {e}")
            return None
    
    def _extract_caption_context(self, plot_info: Dict) -> str:
        """Extract context information for figure captions."""
        filename = os.path.basename(plot_info.get("file_path", ""))
        category = plot_info.get("plot_category", "")
        
        context_hints = []
        
        if "loss" in filename.lower():
            context_hints.append("training loss curves")
        if "accuracy" in filename.lower():
            context_hints.append("accuracy metrics")
        if "lr_" in filename.lower():
            context_hints.append("learning rate ablation")
        if "batch" in filename.lower():
            context_hints.append("batch size analysis")
        if "comparison" in filename.lower():
            context_hints.append("method comparison")
            
        return f"{category} showing {', '.join(context_hints) if context_hints else 'experimental results'}"
    
    def _generate_integration_recommendations(self, imported_figures: List[Dict]) -> List[str]:
        """Generate recommendations for integrating figures into the paper."""
        recommendations = []
        
        if not imported_figures:
            return ["No figures imported. Consider running DataDiscoveryTool first."]
            
        categories = set(fig.get("category") for fig in imported_figures)
        
        if "training_analysis" in categories:
            recommendations.append("Include training analysis figures in Results section to show learning curves")
            
        if "method_comparison" in categories:
            recommendations.append("Use comparison figures to demonstrate method superiority over baselines")
            
        if len(imported_figures) > 3:
            recommendations.append("Consider using MultiPanelCompositionTool to create organized multi-panel figures")
            
        recommendations.append("Generate detailed captions using VLMDocumentAnalysisTool for each imported figure")
        recommendations.append("Reference figures throughout the text using \\ref{fig:label} commands")
        
        return recommendations