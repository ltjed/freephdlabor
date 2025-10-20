"""
ComparisonPlotTool - Generate method/baseline comparison visualizations.

This tool specializes in creating publication-quality comparison plots for evaluating
different methods, models, or experimental conditions against baselines.
Essential for results sections showing competitive analysis.
"""

import json
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from smolagents import Tool

# Handle matplotlib import with fallback
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import matplotlib.patches as patches
    MATPLOTLIB_AVAILABLE = True
except ImportError as e:
    print(f"Warning: matplotlib not available ({e}). Using fallback plotting.")
    MATPLOTLIB_AVAILABLE = False
    plt = None

# Handle statistical imports
try:
    import scipy.stats as stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    stats = None

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    sns = None


class ComparisonPlotTool(Tool):
    name = "comparison_plot_tool"
    description = """
    Generate publication-quality comparison plots for method evaluation and baseline analysis.
    
    This tool specializes in creating comprehensive comparison visualizations including:
    - Method performance comparison with statistical significance testing
    - Baseline vs. proposed method analysis with confidence intervals
    - Multi-metric comparison across different experimental conditions
    - Relative improvement analysis with percentage gains/losses
    - Head-to-head comparison with win/loss/tie analysis
    - Performance distribution analysis across multiple runs
    
    Key features:
    - Automatically detects comparison data from various sources
    - Generates professional bar charts, box plots, and scatter plots
    - Includes statistical rigor (error bars, p-values, effect sizes)
    - Handles multiple experimental runs for robust comparison
    - Creates publication-ready figures with proper formatting
    - Supports both absolute and relative performance comparisons
    
    Input data sources:
    - JSON files with method results and baseline comparisons
    - CSV files with experimental results across conditions
    - NumPy arrays with performance metrics
    - Performance tables and summary statistics
    
    Use this tool when you need to demonstrate the effectiveness of your approach
    compared to existing methods or when showing ablation study results.
    """
    
    inputs = {
        "comparison_specification": {
            "type": "string",
            "description": "Specification of comparison data. Can be: file paths with method data, JSON field paths (e.g., 'results.json:method1_acc,method2_acc,baseline_acc'), or description for auto-discovery"
        },
        "comparison_type": {
            "type": "string",
            "description": "Type of comparison: 'performance_bars' (default), 'relative_improvement', 'statistical_comparison', 'distribution_comparison', 'head_to_head', 'multi_metric'",
            "nullable": True
        },
        "baseline_method": {
            "type": "string",
            "description": "Name/identifier of the baseline method for relative comparisons (default: auto-detect)",
            "nullable": True
        },
        "statistical_testing": {
            "type": "boolean", 
            "description": "Include statistical significance testing (t-tests, p-values) when multiple runs available (default: true)",
            "nullable": True
        },
        "output_filename": {
            "type": "string",
            "description": "Output filename for the generated plot (default: auto-generated based on comparison type)",
            "nullable": True
        }
    }
    
    outputs = {
        "comparison_results": {
            "type": "string",
            "description": "JSON containing comparison analysis results and generated plot paths"
        }
    }
    
    output_type = "string"
    
    def __init__(self, model=None, working_dir: Optional[str] = None):
        """Initialize ComparisonPlotTool.
        
        Args:
            model: LLM model for intelligent analysis (optional)
            working_dir: Working directory for workspace-aware file access
        """
        super().__init__()
        from ..model_utils import get_raw_model
        self.model = get_raw_model(model)
        # Convert to absolute path to prevent nested directory issues
        self.working_dir = os.path.abspath(working_dir) if working_dir else None
        
    def forward(self, comparison_specification: str, comparison_type: str = "performance_bars",
                baseline_method: str = None, statistical_testing: bool = True,
                output_filename: str = None) -> str:
        """
        Generate comparison plots from experimental data.
        
        Args:
            comparison_specification: Specification of comparison data to analyze
            comparison_type: Type of comparison visualization to create
            baseline_method: Reference baseline method for relative comparisons
            statistical_testing: Whether to include statistical significance analysis
            output_filename: Custom output filename
            
        Returns:
            JSON string containing comparison analysis and plot results
        """
        try:
            # Check if matplotlib is available
            if not MATPLOTLIB_AVAILABLE:
                return self._generate_fallback_response("Matplotlib not available")
            
            # Parse comparison specification and load data
            comparison_data = self._parse_and_load_comparison_data(comparison_specification)
            
            if not comparison_data or not comparison_data.get("methods"):
                return json.dumps({
                    "error": "No comparison data found matching the specification",
                    "comparison_specification": comparison_specification,
                    "generated_plots": []
                })
            
            # Auto-detect baseline if not specified
            if baseline_method is None:
                baseline_method = self._auto_detect_baseline(comparison_data)
            
            # Set output directory to paper_workspace/figures/
            if self.working_dir:
                output_dir = os.path.join(self.working_dir, "paper_workspace", "figures")
            else:
                output_dir = os.path.join(os.getcwd(), "paper_workspace", "figures")
            os.makedirs(output_dir, exist_ok=True)
            
            # Setup plotting style
            self._setup_publication_style()
            
            # Generate comparison plots based on type
            generated_plots = []
            
            if comparison_type == "performance_bars":
                plots = self._generate_performance_bars(
                    comparison_data, output_dir, statistical_testing, output_filename)
                generated_plots.extend(plots)
            elif comparison_type == "relative_improvement":
                plots = self._generate_relative_improvement(
                    comparison_data, baseline_method, output_dir, statistical_testing, output_filename)
                generated_plots.extend(plots)
            elif comparison_type == "statistical_comparison":
                plots = self._generate_statistical_comparison(
                    comparison_data, output_dir, statistical_testing, output_filename)
                generated_plots.extend(plots)
            elif comparison_type == "distribution_comparison":
                plots = self._generate_distribution_comparison(
                    comparison_data, output_dir, statistical_testing, output_filename)
                generated_plots.extend(plots)
            elif comparison_type == "head_to_head":
                plots = self._generate_head_to_head_comparison(
                    comparison_data, output_dir, statistical_testing, output_filename)
                generated_plots.extend(plots)
            elif comparison_type == "multi_metric":
                plots = self._generate_multi_metric_comparison(
                    comparison_data, output_dir, statistical_testing, output_filename)
                generated_plots.extend(plots)
            else:
                # Auto-generate appropriate comparison
                plots = self._auto_generate_comparison_plots(
                    comparison_data, baseline_method, output_dir, statistical_testing, output_filename)
                generated_plots.extend(plots)
            
            # Perform statistical analysis if requested
            statistical_results = {}
            if statistical_testing and SCIPY_AVAILABLE:
                statistical_results = self._perform_statistical_analysis(comparison_data, baseline_method)
            
            result = {
                "success": True,
                "comparison_type": comparison_type,
                "comparison_specification": comparison_specification,
                "baseline_method": baseline_method,
                "statistical_testing": statistical_testing,
                "output_directory": output_dir,
                "total_plots": len(generated_plots),
                "generated_plots": generated_plots,
                "statistical_results": statistical_results,
                "comparison_summary": self._summarize_comparison_data(comparison_data)
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            error_result = {
                "success": False,
                "error": f"Comparison plot generation failed: {str(e)}",
                "comparison_specification": comparison_specification,
                "generated_plots": []
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
                    f"Example: Use 'experiment_data/comparison_results.json' instead of the full path."
                )
        else:
            # Relative path - join with workspace
            abs_path = os.path.abspath(os.path.join(abs_working_dir, path))
            return abs_path
    
    def _parse_and_load_comparison_data(self, comparison_spec: str) -> Dict[str, Any]:
        """Parse comparison specification and load data."""
        comparison_data = {"methods": {}, "files": [], "metrics": []}
        
        try:
            # Handle different specification formats
            if ":" in comparison_spec:
                # Format: "file.json:method1_metric,method2_metric"
                file_part, fields_part = comparison_spec.split(":", 1)
                file_path = self._safe_path(file_part.strip()) if self.working_dir else file_part.strip()
                fields = [f.strip() for f in fields_part.split(",")]
                
                if os.path.exists(file_path):
                    data = self._load_json_comparison_data(file_path, fields)
                    comparison_data["methods"].update(data)
                    comparison_data["files"].append(file_path)
                    comparison_data["metrics"] = fields
            
            elif "," in comparison_spec:
                # Format: "file1.json,file2.csv" - load all as different methods
                file_paths = [f.strip() for f in comparison_spec.split(",")]
                for i, file_path in enumerate(file_paths):
                    resolved_path = self._safe_path(file_path) if self.working_dir else file_path
                    if os.path.exists(resolved_path):
                        method_name = f"method_{i+1}"
                        data = self._load_comparison_file(resolved_path, method_name)
                        comparison_data["methods"].update(data)
                        comparison_data["files"].append(resolved_path)
            
            else:
                # Single file or auto-discovery
                if os.path.exists(comparison_spec):
                    resolved_path = self._safe_path(comparison_spec) if self.working_dir else comparison_spec
                    data = self._load_comparison_file(resolved_path)
                    comparison_data["methods"].update(data)
                    comparison_data["files"].append(resolved_path)
                else:
                    # Auto-discovery
                    discovered_data = self._auto_discover_comparison_data(comparison_spec)
                    comparison_data.update(discovered_data)
            
            return comparison_data
            
        except Exception as e:
            print(f"Warning: Failed to parse comparison data: {e}")
            return {"methods": {}, "files": [], "metrics": []}
    
    def _load_json_comparison_data(self, file_path: str, fields: List[str]) -> Dict[str, Any]:
        """Load specific comparison fields from JSON file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            methods = {}
            for field in fields:
                # Handle nested field access
                value = data
                for key in field.split('.'):
                    if isinstance(value, dict) and key in value:
                        value = value[key]
                    else:
                        value = None
                        break
                
                if value is not None:
                    # Extract method name from field
                    method_name = field.split('_')[0] if '_' in field else field
                    if isinstance(value, (list, np.ndarray)):
                        methods[method_name] = np.array(value)
                    elif isinstance(value, (int, float)):
                        methods[method_name] = value
                    elif isinstance(value, dict):
                        # Handle nested method results
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, (int, float, list, np.ndarray)):
                                methods[f"{method_name}_{sub_key}"] = sub_value
            
            return methods
            
        except Exception as e:
            print(f"Warning: Failed to load JSON comparison data: {e}")
            return {}
    
    def _load_comparison_file(self, file_path: str, method_name: str = None) -> Dict[str, Any]:
        """Load comparison data from various file formats."""
        try:
            if method_name is None:
                method_name = os.path.basename(file_path).split('.')[0]
            
            if file_path.endswith('.json'):
                return self._load_json_comparison_file(file_path, method_name)
            elif file_path.endswith('.csv'):
                return self._load_csv_comparison_file(file_path, method_name)
            elif file_path.endswith('.npy'):
                data = np.load(file_path)
                return {method_name: data}
            elif file_path.endswith('.npz'):
                data = np.load(file_path)
                return {f"{method_name}_{key}": data[key] for key in data.keys()}
            else:
                return {}
                
        except Exception as e:
            print(f"Warning: Failed to load comparison file {file_path}: {e}")
            return {}
    
    def _load_json_comparison_file(self, file_path: str, method_name: str) -> Dict[str, Any]:
        """Load comparison data from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        methods = {}
        
        # Look for performance metrics
        performance_indicators = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'score', 'metric']
        
        def extract_metrics(obj, prefix=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    full_key = f"{prefix}_{key}" if prefix else key
                    if isinstance(value, (int, float)):
                        # Single metric value
                        methods[f"{method_name}_{full_key}"] = value
                    elif isinstance(value, list) and len(value) > 0:
                        if isinstance(value[0], (int, float)):
                            # Multiple runs or time series
                            methods[f"{method_name}_{full_key}"] = np.array(value)
                    elif isinstance(value, dict):
                        extract_metrics(value, full_key)
        
        extract_metrics(data)
        return methods
    
    def _load_csv_comparison_file(self, file_path: str, method_name: str) -> Dict[str, Any]:
        """Load comparison data from CSV file."""
        try:
            import pandas as pd
            df = pd.read_csv(file_path)
            
            methods = {}
            # Use numerical columns as metrics
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                methods[f"{method_name}_{col}"] = df[col].values
            
            return methods
            
        except ImportError:
            print("Warning: pandas not available for CSV loading")
            return {}
    
    def _auto_discover_comparison_data(self, description: str) -> Dict[str, Any]:
        """Auto-discover comparison data based on description."""
        comparison_data = {"methods": {}, "files": [], "metrics": []}
        
        if not self.working_dir:
            return comparison_data
        
        # Search for comparison-related files
        search_dirs = [
            self.working_dir,
            os.path.join(self.working_dir, "experiment_data"),
            os.path.join(self.working_dir, "results"),
            os.path.join(self.working_dir, "comparison")
        ]
        
        comparison_keywords = ['result', 'comparison', 'baseline', 'method', 'evaluation']
        
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                for ext in ['.json', '.csv', '.npy']:
                    files = list(Path(search_dir).glob(f"*{ext}"))
                    for file_path in files:
                        filename = file_path.name.lower()
                        if any(keyword in filename for keyword in comparison_keywords):
                            method_name = file_path.stem
                            data = self._load_comparison_file(str(file_path), method_name)
                            comparison_data["methods"].update(data)
                            comparison_data["files"].append(str(file_path))
        
        return comparison_data
    
    def _auto_detect_baseline(self, comparison_data: Dict) -> str:
        """Auto-detect baseline method from comparison data."""
        methods = comparison_data.get("methods", {})
        
        # Look for common baseline indicators
        baseline_indicators = ['baseline', 'base', 'reference', 'vanilla', 'original']
        
        for method_name in methods.keys():
            method_lower = method_name.lower()
            if any(indicator in method_lower for indicator in baseline_indicators):
                return method_name
        
        # If no explicit baseline, use the first method
        if methods:
            return list(methods.keys())[0]
        
        return "baseline"
    
    def _setup_publication_style(self):
        """Setup matplotlib for publication-quality plots."""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        plt.style.use('default')
        if SEABORN_AVAILABLE:
            sns.set_palette("Set2")
        
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 11,
            'figure.figsize': (12, 8),
            'lines.linewidth': 2,
            'grid.alpha': 0.3,
            'figure.dpi': 100,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight'
        })
    
    def _generate_performance_bars(self, comparison_data: Dict, output_dir: str,
                                 statistical_testing: bool, output_filename: str = None) -> List[Dict]:
        """Generate performance bar chart comparison."""
        plots = []
        methods = comparison_data.get("methods", {})
        
        if not methods:
            return plots
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract final performance values
        method_names = []
        performance_values = []
        error_bars = []
        
        for method, data in methods.items():
            method_names.append(method.replace('_', ' ').title())
            
            if isinstance(data, np.ndarray):
                # Multiple runs - use mean and std
                performance_values.append(np.mean(data))
                error_bars.append(np.std(data) if statistical_testing else 0)
            else:
                # Single value
                performance_values.append(float(data))
                error_bars.append(0)
        
        # Create bars
        bars = ax.bar(range(len(method_names)), performance_values, 
                     yerr=error_bars if statistical_testing else None,
                     capsize=5, alpha=0.8)
        
        # Customize plot
        ax.set_xlabel('Methods')
        ax.set_ylabel('Performance')
        ax.set_title('Method Performance Comparison')
        ax.set_xticks(range(len(method_names)))
        ax.set_xticklabels(method_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value, error in zip(bars, performance_values, error_bars):
            height = bar.get_height()
            label = f'{value:.3f}'
            if error > 0:
                label += f' ±{error:.3f}'
            ax.text(bar.get_x() + bar.get_width()/2., height + max(error, 0.01),
                   label, ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        filename = output_filename or "method_performance_comparison.png"
        plot_path = os.path.join(output_dir, filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        plots.append({
            "type": "performance_bars",
            "path": plot_path,
            "filename": filename,
            "title": "Method Performance Comparison",
            "description": "Bar chart comparing performance across different methods",
            "methods_compared": len(method_names),
            "statistical_analysis": statistical_testing
        })
        
        return plots
    
    def _generate_relative_improvement(self, comparison_data: Dict, baseline_method: str,
                                     output_dir: str, statistical_testing: bool, 
                                     output_filename: str = None) -> List[Dict]:
        """Generate relative improvement comparison."""
        plots = []
        methods = comparison_data.get("methods", {})
        
        if not methods or baseline_method not in methods:
            return plots
        
        # Get baseline performance
        baseline_data = methods[baseline_method]
        baseline_value = np.mean(baseline_data) if isinstance(baseline_data, np.ndarray) else float(baseline_data)
        
        # Calculate relative improvements
        fig, ax = plt.subplots(figsize=(10, 6))
        
        method_names = []
        improvements = []
        colors = []
        
        for method, data in methods.items():
            if method == baseline_method:
                continue
                
            method_value = np.mean(data) if isinstance(data, np.ndarray) else float(data)
            improvement = ((method_value - baseline_value) / baseline_value) * 100
            
            method_names.append(method.replace('_', ' ').title())
            improvements.append(improvement)
            colors.append('green' if improvement > 0 else 'red')
        
        # Create bars
        bars = ax.bar(range(len(method_names)), improvements, color=colors, alpha=0.8)
        
        # Customize plot
        ax.set_xlabel('Methods')
        ax.set_ylabel(f'Improvement over {baseline_method} (%)')
        ax.set_title(f'Relative Performance Improvement vs {baseline_method}')
        ax.set_xticks(range(len(method_names)))
        ax.set_xticklabels(method_names, rotation=45, ha='right')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, improvement in zip(bars, improvements):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., 
                   height + (1 if height > 0 else -3),
                   f'{improvement:.1f}%', ha='center', 
                   va='bottom' if height > 0 else 'top')
        
        plt.tight_layout()
        
        # Save plot
        filename = output_filename or "relative_improvement_comparison.png"
        plot_path = os.path.join(output_dir, filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        plots.append({
            "type": "relative_improvement",
            "path": plot_path,
            "filename": filename,
            "title": f"Relative Improvement vs {baseline_method}",
            "description": "Percentage improvement comparison against baseline method",
            "baseline_method": baseline_method,
            "methods_compared": len(method_names)
        })
        
        return plots
    
    def _generate_statistical_comparison(self, comparison_data: Dict, output_dir: str,
                                       statistical_testing: bool, output_filename: str = None) -> List[Dict]:
        """Generate statistical comparison with significance testing."""
        if not SCIPY_AVAILABLE:
            return []
        
        plots = []
        methods = comparison_data.get("methods", {})
        
        # Filter methods with multiple data points for statistical testing
        statistical_methods = {k: v for k, v in methods.items() 
                             if isinstance(v, np.ndarray) and len(v) > 1}
        
        if len(statistical_methods) < 2:
            return plots
        
        # Create box plot comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        
        data_for_boxplot = []
        labels = []
        
        for method, data in statistical_methods.items():
            data_for_boxplot.append(data)
            labels.append(method.replace('_', ' ').title())
        
        # Create box plots
        box_plot = ax.boxplot(data_for_boxplot, labels=labels, patch_artist=True)
        
        # Customize colors
        colors = plt.cm.Set3(np.linspace(0, 1, len(data_for_boxplot)))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xlabel('Methods')
        ax.set_ylabel('Performance Distribution')
        ax.set_title('Statistical Performance Comparison')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save plot
        filename = output_filename or "statistical_comparison.png"
        plot_path = os.path.join(output_dir, filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        plots.append({
            "type": "statistical_comparison",
            "path": plot_path,
            "filename": filename,
            "title": "Statistical Performance Comparison",
            "description": "Box plot comparison showing performance distributions and statistical significance",
            "methods_compared": len(labels)
        })
        
        return plots
    
    def _generate_distribution_comparison(self, comparison_data: Dict, output_dir: str,
                                        statistical_testing: bool, output_filename: str = None) -> List[Dict]:
        """Generate distribution comparison plots."""
        # Similar to statistical comparison but with histograms
        return []
    
    def _generate_head_to_head_comparison(self, comparison_data: Dict, output_dir: str,
                                        statistical_testing: bool, output_filename: str = None) -> List[Dict]:
        """Generate head-to-head comparison matrix."""
        # Implementation for pairwise comparisons
        return []
    
    def _generate_multi_metric_comparison(self, comparison_data: Dict, output_dir: str,
                                        statistical_testing: bool, output_filename: str = None) -> List[Dict]:
        """Generate multi-metric comparison plots."""
        # Implementation for radar charts or multi-dimensional comparisons
        return []
    
    def _auto_generate_comparison_plots(self, comparison_data: Dict, baseline_method: str,
                                      output_dir: str, statistical_testing: bool, 
                                      output_filename: str = None) -> List[Dict]:
        """Auto-generate appropriate comparison plots."""
        plots = []
        
        # Generate performance bars by default
        bar_plots = self._generate_performance_bars(comparison_data, output_dir, statistical_testing)
        plots.extend(bar_plots)
        
        # Add relative improvement if baseline is available
        if baseline_method:
            improvement_plots = self._generate_relative_improvement(
                comparison_data, baseline_method, output_dir, statistical_testing)
            plots.extend(improvement_plots)
        
        return plots
    
    def _perform_statistical_analysis(self, comparison_data: Dict, baseline_method: str) -> Dict[str, Any]:
        """Perform statistical analysis on comparison data."""
        if not SCIPY_AVAILABLE:
            return {}
        
        methods = comparison_data.get("methods", {})
        statistical_results = {
            "pairwise_tests": [],
            "effect_sizes": {},
            "summary_statistics": {}
        }
        
        # Calculate summary statistics
        for method, data in methods.items():
            if isinstance(data, np.ndarray) and len(data) > 1:
                statistical_results["summary_statistics"][method] = {
                    "mean": float(np.mean(data)),
                    "std": float(np.std(data)),
                    "median": float(np.median(data)),
                    "min": float(np.min(data)),
                    "max": float(np.max(data)),
                    "n": len(data)
                }
        
        # Perform pairwise t-tests
        method_list = list(methods.keys())
        for i in range(len(method_list)):
            for j in range(i+1, len(method_list)):
                method1, method2 = method_list[i], method_list[j]
                data1, data2 = methods[method1], methods[method2]
                
                if isinstance(data1, np.ndarray) and isinstance(data2, np.ndarray):
                    if len(data1) > 1 and len(data2) > 1:
                        try:
                            t_stat, p_value = stats.ttest_ind(data1, data2)
                            statistical_results["pairwise_tests"].append({
                                "method1": method1,
                                "method2": method2,
                                "t_statistic": float(t_stat),
                                "p_value": float(p_value),
                                "significant": p_value < 0.05
                            })
                        except:
                            continue
        
        return statistical_results
    
    def _summarize_comparison_data(self, comparison_data: Dict) -> Dict[str, Any]:
        """Summarize comparison data for reporting."""
        methods = comparison_data.get("methods", {})
        
        summary = {
            "total_methods": len(methods),
            "method_names": list(methods.keys()),
            "data_types": {},
            "performance_range": {}
        }
        
        for method, data in methods.items():
            if isinstance(data, np.ndarray):
                summary["data_types"][method] = f"array({len(data)} values)"
                summary["performance_range"][method] = {
                    "min": float(np.min(data)),
                    "max": float(np.max(data)),
                    "mean": float(np.mean(data))
                }
            else:
                summary["data_types"][method] = "single_value"
                summary["performance_range"][method] = float(data)
        
        return summary
    
    def _generate_fallback_response(self, error_msg: str) -> str:
        """Generate fallback response when plotting is not available."""
        return json.dumps({
            "success": False,
            "error": error_msg,
            "suggestion": "Install matplotlib and related dependencies to enable comparison plot generation",
            "generated_plots": []
        })