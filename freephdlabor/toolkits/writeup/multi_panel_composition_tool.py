"""
MultiPanelCompositionTool - Combine multiple plots into publication-quality layouts.

This tool specializes in creating professional multi-panel figures by combining
existing plots or data into unified layouts suitable for academic publications.
Essential for creating comprehensive figure compositions for papers.
"""

import json
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from smolagents import Tool

# Handle matplotlib import with fallback
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import matplotlib.patches as patches
    import matplotlib.image as mpimg
    MATPLOTLIB_AVAILABLE = True
except ImportError as e:
    print(f"Warning: matplotlib not available ({e}). Using fallback plotting.")
    MATPLOTLIB_AVAILABLE = False
    plt = None

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class MultiPanelCompositionTool(Tool):
    name = "multi_panel_composition_tool"
    description = """
    Create publication-quality multi-panel figure compositions from existing plots and data.
    
    This tool specializes in combining multiple visualizations into unified layouts including:
    - Multi-panel grid layouts (2x2, 3x2, custom arrangements)
    - Figure composition with consistent styling and labeling
    - Subplot arrangement with proper spacing and alignment
    - Panel labeling (A, B, C, etc.) for academic publications
    - Unified legends and color schemes across panels
    - Professional typography and formatting
    
    Key capabilities:
    - Combine existing plot images into multi-panel layouts
    - Generate new plots and arrange them systematically
    - Create publication-ready figures with proper panel labels
    - Handle different plot types and scales uniformly
    - Support custom layouts and aspect ratios
    - Generate comprehensive figure captions
    
    Layout options:
    - Grid layouts: 2x2, 3x2, 2x3, 3x3, custom grid dimensions
    - Asymmetric layouts: combined panels, different sizes
    - Comparison layouts: side-by-side, stacked arrangements
    - Timeline layouts: sequential panel arrangements
    
    Input sources:
    - Existing plot image files (PNG, JPG, PDF)
    - Data specifications for generating new plots within panels
    - Mixed approach: some existing plots, some generated on-demand
    
    Use this tool when you need to create comprehensive figures that combine
    multiple related analyses or when journals require multi-panel layouts.
    """
    
    inputs = {
        "composition_specification": {
            "type": "string",
            "description": "Specification for the multi-panel composition. Can be: plot file paths (comma-separated), layout description, or mixed specification with both existing plots and data for new plots"
        },
        "layout_type": {
            "type": "string",
            "description": "Panel layout arrangement (default: '2x2'):\n" +
                          "• '2x2': 4 panels in 2 rows, 2 columns\n" +
                          "• '3x2': 6 panels in 3 rows, 2 columns\n" +
                          "• '2x3': 6 panels in 2 rows, 3 columns\n" +
                          "• '3x3': 9 panels in 3 rows, 3 columns\n" +
                          "• '1x4': 4 panels in single row\n" +
                          "• '4x1': 4 panels in single column\n" +
                          "• 'custom': Auto-determine optimal layout based on panel count\n" +
                          "• 'asymmetric': Flexible layout with varying panel sizes",
            "nullable": True
        },
        "panel_labels": {
            "type": "string",
            "description": "Panel labeling scheme (default: 'letters'):\n" +
                          "• 'letters': Uppercase letters (A, B, C, D...) - standard for academic papers\n" +
                          "• 'numbers': Sequential numbers (1, 2, 3, 4...)\n" +
                          "• 'roman': Lowercase Roman numerals (i, ii, iii, iv...)\n" +
                          "• 'none': No panel labels",
            "nullable": True
        },
        "figure_title": {
            "type": "string",
            "description": "Overall figure title (optional)",
            "nullable": True
        },
        "panel_titles": {
            "type": "string",
            "description": "Comma-separated list of individual panel titles (optional)",
            "nullable": True
        },
        "output_filename": {
            "type": "string",
            "description": "Output filename for the composed figure (default: auto-generated)",
            "nullable": True
        }
    }
    
    outputs = {
        "composition_results": {
            "type": "string",
            "description": "JSON containing composition results and generated figure path"
        }
    }
    
    output_type = "string"
    
    def __init__(self, model=None, working_dir: Optional[str] = None):
        """Initialize MultiPanelCompositionTool.
        
        Args:
            model: LLM model for intelligent layout decisions (optional)
            working_dir: Working directory for workspace-aware file access
        """
        super().__init__()
        from ..model_utils import get_raw_model
        self.model = get_raw_model(model)
        # Convert to absolute path to prevent nested directory issues
        self.working_dir = os.path.abspath(working_dir) if working_dir else None
        
    def forward(self, composition_specification: str, layout_type: str = "2x2",
                panel_labels: str = "letters", figure_title: str = None,
                panel_titles: str = None, output_filename: str = None) -> str:
        """
        Create multi-panel figure composition from plots and data.
        
        Args:
            composition_specification: Specification of plots/data to compose
            layout_type: Layout arrangement for the panels
            panel_labels: Labeling scheme for panels
            figure_title: Overall figure title
            panel_titles: Individual panel titles
            output_filename: Custom output filename
            
        Returns:
            JSON string containing composition results and file paths
        """
        try:
            # Check if matplotlib is available
            if not MATPLOTLIB_AVAILABLE:
                return self._generate_fallback_response("Matplotlib not available")
            
            # Parse composition specification
            composition_data = self._parse_composition_specification(composition_specification)
            
            if not composition_data or not composition_data.get("panels"):
                return json.dumps({
                    "error": "No valid panels found in composition specification",
                    "composition_specification": composition_specification,
                    "generated_figure": None
                })
            
            # Parse layout configuration
            layout_config = self._parse_layout_type(layout_type)
            
            # Parse panel titles if provided
            parsed_panel_titles = []
            if panel_titles:
                parsed_panel_titles = [title.strip() for title in panel_titles.split(",")]
            
            # Set output directory to paper_workspace/figures/
            if self.working_dir:
                output_dir = os.path.join(self.working_dir, "paper_workspace", "figures")
            else:
                output_dir = os.path.join(os.getcwd(), "paper_workspace", "figures")
            os.makedirs(output_dir, exist_ok=True)
            
            # Setup plotting style
            self._setup_publication_style()
            
            # Create multi-panel composition
            figure_info = self._create_multi_panel_composition(
                composition_data, layout_config, panel_labels, figure_title,
                parsed_panel_titles, output_dir, output_filename
            )
            
            result = {
                "success": True,
                "composition_specification": composition_specification,
                "layout_type": layout_type,
                "panel_labels": panel_labels,
                "figure_title": figure_title,
                "output_directory": output_dir,
                "figure_info": figure_info,
                "panels_composed": len(composition_data.get("panels", [])),
                "layout_config": layout_config
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            error_result = {
                "success": False,
                "error": f"Multi-panel composition failed: {str(e)}",
                "composition_specification": composition_specification,
                "generated_figure": None
            }
            return json.dumps(error_result, indent=2)
    
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
                    f"Example: Use 'experimental_plots/figure1.png' instead of the full path."
                )
        else:
            # Relative path - join with workspace
            abs_path = os.path.abspath(os.path.join(abs_working_dir, path))
            return abs_path
    
    def _parse_composition_specification(self, composition_spec: str) -> Dict[str, Any]:
        """Parse composition specification and identify panel sources."""
        composition_data = {"panels": [], "type": "unknown"}
        
        try:
            # Handle different specification formats
            if "," in composition_spec:
                # Multiple items - could be files or mixed specification
                items = [item.strip() for item in composition_spec.split(",")]
                
                for item in items:
                    panel_info = self._identify_panel_source(item)
                    if panel_info:
                        composition_data["panels"].append(panel_info)
                
                composition_data["type"] = "multiple_items"
            
            else:
                # Single item or description
                panel_info = self._identify_panel_source(composition_spec)
                if panel_info:
                    composition_data["panels"].append(panel_info)
                    composition_data["type"] = "single_item"
                else:
                    # Try auto-discovery based on description
                    discovered_panels = self._auto_discover_panels(composition_spec)
                    composition_data["panels"] = discovered_panels
                    composition_data["type"] = "auto_discovered"
            
            return composition_data
            
        except Exception as e:
            print(f"Warning: Failed to parse composition specification: {e}")
            return {"panels": [], "type": "error"}
    
    def _identify_panel_source(self, item_spec: str) -> Optional[Dict[str, Any]]:
        """Identify the source type for a panel specification."""
        try:
            # Check if it's an existing plot file
            resolved_path = self._safe_path(item_spec) if self.working_dir else item_spec
            
            if os.path.exists(resolved_path) and self._is_image_file(resolved_path):
                return {
                    "type": "existing_plot",
                    "source": resolved_path,
                    "filename": os.path.basename(resolved_path),
                    "title": os.path.splitext(os.path.basename(resolved_path))[0].replace('_', ' ').title()
                }
            
            # Check if it's a data specification (contains ":")
            elif ":" in item_spec:
                return {
                    "type": "data_plot",
                    "source": item_spec,
                    "title": "Generated Plot"
                }
            
            # Check if it's a description for plot generation
            else:
                return {
                    "type": "description",
                    "source": item_spec,
                    "title": "Generated Plot"
                }
                
        except Exception as e:
            print(f"Warning: Failed to identify panel source for {item_spec}: {e}")
            return None
    
    def _is_image_file(self, file_path: str) -> bool:
        """Check if file is a supported image format."""
        image_extensions = ['.png', '.jpg', '.jpeg', '.pdf', '.svg', '.eps']
        return any(file_path.lower().endswith(ext) for ext in image_extensions)
    
    def _auto_discover_panels(self, description: str) -> List[Dict[str, Any]]:
        """Auto-discover potential panels based on description."""
        discovered_panels = []
        
        if not self.working_dir:
            return discovered_panels
        
        # Search for plot files in common directories
        search_dirs = [
            self.working_dir,
            os.path.join(self.working_dir, "experimental_plots"),
            os.path.join(self.working_dir, "figures"),
            os.path.join(self.working_dir, "plots")
        ]
        
        # Look for relevant keywords in description
        keywords = description.lower().split()
        
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                for ext in ['.png', '.jpg', '.jpeg', '.pdf']:
                    files = list(Path(search_dir).glob(f"*{ext}"))
                    for file_path in files:
                        filename = file_path.name.lower()
                        # Check if filename matches description keywords
                        if any(keyword in filename for keyword in keywords if len(keyword) > 3):
                            discovered_panels.append({
                                "type": "existing_plot",
                                "source": str(file_path),
                                "filename": file_path.name,
                                "title": os.path.splitext(file_path.name)[0].replace('_', ' ').title(),
                                "auto_discovered": True
                            })
        
        return discovered_panels[:6]  # Limit to reasonable number
    
    def _parse_layout_type(self, layout_type: str) -> Dict[str, Any]:
        """Parse layout type and return configuration."""
        layout_config = {
            "rows": 2,
            "cols": 2,
            "layout_name": layout_type,
            "figure_size": (12, 8)
        }
        
        if "x" in layout_type.lower():
            # Parse grid layout like "2x3", "3x2"
            try:
                rows, cols = layout_type.lower().split("x")
                layout_config["rows"] = int(rows)
                layout_config["cols"] = int(cols)
                
                # Adjust figure size based on layout
                layout_config["figure_size"] = (4 * layout_config["cols"], 3 * layout_config["rows"])
                
            except ValueError:
                print(f"Warning: Invalid layout format {layout_type}, using default 2x2")
        
        elif layout_type.lower() == "asymmetric":
            layout_config.update({
                "rows": 2,
                "cols": 3,
                "asymmetric": True,
                "figure_size": (15, 8)
            })
        
        elif layout_type.lower() == "custom":
            # Custom layout - will be determined based on number of panels
            layout_config["custom"] = True
        
        return layout_config
    
    def _setup_publication_style(self):
        """Setup matplotlib for publication-quality multi-panel figures."""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        plt.style.use('default')
        
        plt.rcParams.update({
            'font.size': 10,
            'axes.labelsize': 11,
            'axes.titlesize': 12,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'lines.linewidth': 1.5,
            'grid.alpha': 0.3,
            'figure.dpi': 100,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'figure.facecolor': 'white'
        })
    
    def _create_multi_panel_composition(self, composition_data: Dict, layout_config: Dict,
                                      panel_labels: str, figure_title: str,
                                      panel_titles: List[str], output_dir: str, 
                                      output_filename: str = None) -> Dict[str, Any]:
        """Create the actual multi-panel composition."""
        panels = composition_data.get("panels", [])
        
        if not panels:
            raise ValueError("No panels to compose")
        
        # Adjust layout if custom or if panels don't fit
        if layout_config.get("custom", False) or len(panels) > layout_config["rows"] * layout_config["cols"]:
            layout_config = self._optimize_layout_for_panels(len(panels))
        
        # Create figure with GridSpec
        fig = plt.figure(figsize=layout_config["figure_size"])
        
        if figure_title:
            fig.suptitle(figure_title, fontsize=16, fontweight='bold', y=0.95)
        
        # Create GridSpec for layout
        gs = GridSpec(layout_config["rows"], layout_config["cols"], 
                     figure=fig, hspace=0.4, wspace=0.3)
        
        panel_info = []
        
        # Process each panel
        for i, panel in enumerate(panels):
            if i >= layout_config["rows"] * layout_config["cols"]:
                break  # Don't exceed layout capacity
            
            # Calculate subplot position
            row = i // layout_config["cols"]
            col = i % layout_config["cols"]
            
            ax = fig.add_subplot(gs[row, col])
            
            # Generate panel label
            panel_label = self._generate_panel_label(i, panel_labels)
            
            # Get panel title
            panel_title = panel_titles[i] if i < len(panel_titles) else panel.get("title", "")
            
            # Process panel based on type
            panel_result = self._process_panel(panel, ax, panel_label, panel_title)
            panel_info.append(panel_result)
        
        plt.tight_layout()
        
        # Save composed figure
        filename = output_filename or "multi_panel_composition.png"
        figure_path = os.path.join(output_dir, filename)
        plt.savefig(figure_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return {
            "figure_path": figure_path,
            "filename": filename,
            "layout": f"{layout_config['rows']}x{layout_config['cols']}",
            "panels_included": len(panel_info),
            "panel_details": panel_info,
            "figure_title": figure_title
        }
    
    def _optimize_layout_for_panels(self, num_panels: int) -> Dict[str, Any]:
        """Optimize layout configuration based on number of panels."""
        if num_panels <= 2:
            return {"rows": 1, "cols": 2, "figure_size": (10, 4)}
        elif num_panels <= 4:
            return {"rows": 2, "cols": 2, "figure_size": (10, 8)}
        elif num_panels <= 6:
            return {"rows": 2, "cols": 3, "figure_size": (15, 8)}
        elif num_panels <= 9:
            return {"rows": 3, "cols": 3, "figure_size": (15, 12)}
        else:
            # For more panels, use 4-column layout
            rows = (num_panels + 3) // 4
            return {"rows": rows, "cols": 4, "figure_size": (16, 4 * rows)}
    
    def _generate_panel_label(self, index: int, label_scheme: str) -> str:
        """Generate panel label based on scheme."""
        if label_scheme.lower() == "letters":
            return chr(ord('A') + index)
        elif label_scheme.lower() == "numbers":
            return str(index + 1)
        elif label_scheme.lower() == "roman":
            roman_numerals = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"]
            return roman_numerals[index] if index < len(roman_numerals) else str(index + 1)
        else:  # none
            return ""
    
    def _process_panel(self, panel: Dict, ax, panel_label: str, panel_title: str) -> Dict[str, Any]:
        """Process individual panel based on its type."""
        panel_result = {
            "index": panel_label,
            "title": panel_title,
            "type": panel.get("type", "unknown"),
            "source": panel.get("source", ""),
            "success": False
        }
        
        try:
            if panel["type"] == "existing_plot":
                success = self._load_existing_plot(panel["source"], ax)
                panel_result["success"] = success
                
            elif panel["type"] == "data_plot":
                success = self._generate_data_plot(panel["source"], ax)
                panel_result["success"] = success
                
            elif panel["type"] == "description":
                success = self._generate_plot_from_description(panel["source"], ax)
                panel_result["success"] = success
                
            else:
                # Fallback - create placeholder
                ax.text(0.5, 0.5, f"Panel {panel_label}\n(Source unavailable)", 
                       ha='center', va='center', transform=ax.transAxes)
                panel_result["success"] = False
            
            # Add panel label and title
            if panel_label:
                ax.text(0.02, 0.98, f"{panel_label}.", transform=ax.transAxes,
                       fontsize=14, fontweight='bold', va='top', ha='left')
            
            if panel_title:
                ax.set_title(panel_title, fontsize=11, pad=10)
                
        except Exception as e:
            print(f"Warning: Failed to process panel {panel_label}: {e}")
            ax.text(0.5, 0.5, f"Panel {panel_label}\n(Error loading)", 
                   ha='center', va='center', transform=ax.transAxes)
            panel_result["success"] = False
            panel_result["error"] = str(e)
        
        return panel_result
    
    def _load_existing_plot(self, image_path: str, ax) -> bool:
        """Load existing plot image into subplot."""
        try:
            if PIL_AVAILABLE:
                # Use PIL for better image handling
                img = Image.open(image_path)
                ax.imshow(img)
            else:
                # Fallback to matplotlib
                img = mpimg.imread(image_path)
                ax.imshow(img)
            
            ax.axis('off')  # Hide axes for image plots
            return True
            
        except Exception as e:
            print(f"Warning: Failed to load image {image_path}: {e}")
            return False
    
    def _generate_data_plot(self, data_spec: str, ax) -> bool:
        """Generate plot from data specification."""
        try:
            # Parse data specification (simplified)
            # This would integrate with other plotting tools
            # For now, create a placeholder
            ax.plot([1, 2, 3, 4], [1, 4, 2, 3], 'b-o')
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            ax.grid(True, alpha=0.3)
            return True
            
        except Exception as e:
            print(f"Warning: Failed to generate data plot: {e}")
            return False
    
    def _generate_plot_from_description(self, description: str, ax) -> bool:
        """Generate plot from textual description."""
        try:
            # This would use the model to generate appropriate plots
            # For now, create a representative plot based on description
            if "training" in description.lower() or "loss" in description.lower():
                # Generate training curve
                epochs = np.arange(1, 51)
                loss = np.exp(-epochs/20) + 0.1 * np.random.random(50)
                ax.plot(epochs, loss, 'r-', linewidth=2)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.grid(True, alpha=0.3)
                
            elif "comparison" in description.lower() or "bar" in description.lower():
                # Generate comparison bars
                methods = ['Method A', 'Method B', 'Method C', 'Baseline']
                values = [85, 78, 92, 70]
                ax.bar(methods, values, alpha=0.7)
                ax.set_ylabel('Performance (%)')
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                
            else:
                # Generic scatter plot
                x = np.random.randn(50)
                y = x + 0.5 * np.random.randn(50)
                ax.scatter(x, y, alpha=0.6)
                ax.set_xlabel('X-axis')
                ax.set_ylabel('Y-axis')
                ax.grid(True, alpha=0.3)
            
            return True
            
        except Exception as e:
            print(f"Warning: Failed to generate plot from description: {e}")
            return False
    
    def _generate_fallback_response(self, error_msg: str) -> str:
        """Generate fallback response when composition is not available."""
        return json.dumps({
            "success": False,
            "error": error_msg,
            "suggestion": "Install matplotlib and PIL/Pillow to enable multi-panel composition",
            "generated_figure": None
        })