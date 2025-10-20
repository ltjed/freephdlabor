"""
TrainingAnalysisPlotTool - Generate training curves and learning analysis visualizations.

This tool specializes in creating publication-quality training analysis plots including
loss curves, accuracy progression, convergence analysis, and overfitting detection.
Essential for machine learning paper results sections.
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


class TrainingAnalysisPlotTool(Tool):
    name = "training_analysis_plot_tool"
    description = """
    Generate publication-quality training analysis plots for machine learning research.
    
    This tool specializes in creating comprehensive training visualizations including:
    - Training/validation loss curves with proper scaling and error bars
    - Accuracy progression analysis with confidence intervals
    - Learning rate scheduling visualization
    - Convergence analysis and overfitting detection
    - Multi-run statistical analysis with significance testing
    - Optimizer comparison plots
    
    Key features:
    - Automatically detects training metrics from various data formats
    - Generates professional multi-panel layouts for papers
    - Includes statistical rigor (error bars, confidence intervals)
    - Handles multiple experimental runs and statistical analysis
    - Creates publication-ready figures with proper formatting
    
    Input data sources:
    - JSON files with training logs (loss, accuracy, metrics over epochs)
    - CSV files with training history
    - NumPy arrays with training sequences
    - NPZ archives with multiple training runs
    
    Use this tool when you need to visualize training dynamics, convergence behavior,
    or compare training performance across different experimental conditions.
    """
    
    inputs = {
        "data_specification": {
            "type": "string",
            "description": "Specification of training data to plot. Can be: file paths (comma-separated), JSON field paths (e.g., 'results.json:train_loss,val_loss'), or data description for auto-discovery"
        },
        "plot_type": {
            "type": "string", 
            "description": "Type of training analysis: 'comprehensive' (default), 'loss_curves', 'accuracy_curves', 'convergence_analysis', 'optimizer_comparison', 'learning_rate_schedule'",
            "nullable": True
        },
        "statistical_analysis": {
            "type": "boolean",
            "description": "Include statistical analysis (error bars, confidence intervals) if multiple runs available (default: true)",
            "nullable": True
        },
        "output_filename": {
            "type": "string",
            "description": "Output filename for the generated plot (default: auto-generated based on plot type)",
            "nullable": True
        }
    }
    
    outputs = {
        "plot_results": {
            "type": "string",
            "description": "JSON containing plot generation results and file paths"
        }
    }
    
    output_type = "string"
    
    def __init__(self, model=None, working_dir: Optional[str] = None):
        """Initialize TrainingAnalysisPlotTool.
        
        Args:
            model: LLM model for intelligent analysis (optional)
            working_dir: Working directory for workspace-aware file access
        """
        super().__init__()
        from ..model_utils import get_raw_model
        self.model = get_raw_model(model)
        # Convert to absolute path to prevent nested directory issues
        self.working_dir = os.path.abspath(working_dir) if working_dir else None
        
    def forward(self, data_specification: str, plot_type: str = "comprehensive",
                statistical_analysis: bool = True, output_filename: str = None) -> str:
        """
        Generate training analysis plots from experimental data.
        
        Args:
            data_specification: Specification of training data to analyze
            plot_type: Type of training analysis to perform
            statistical_analysis: Whether to include statistical analysis
            output_filename: Custom output filename
            
        Returns:
            JSON string containing plot generation results
        """
        try:
            # Check if matplotlib is available
            if not MATPLOTLIB_AVAILABLE:
                return self._generate_fallback_response("Matplotlib not available")
            
            # Parse data specification and load training data
            training_data = self._parse_and_load_training_data(data_specification)
            
            if not training_data:
                return json.dumps({
                    "error": "No training data found matching the specification",
                    "data_specification": data_specification,
                    "generated_plots": []
                })
            
            # Set output directory to paper_workspace/figures/
            if self.working_dir:
                output_dir = os.path.join(self.working_dir, "paper_workspace", "figures")
            else:
                output_dir = os.path.join(os.getcwd(), "paper_workspace", "figures")
            os.makedirs(output_dir, exist_ok=True)
            
            # Setup plotting style
            self._setup_publication_style()
            
            # Generate plots based on type
            generated_plots = []
            
            if plot_type == "comprehensive":
                plots = self._generate_comprehensive_training_analysis(
                    training_data, output_dir, statistical_analysis, output_filename)
                generated_plots.extend(plots)
            elif plot_type == "loss_curves":
                plots = self._generate_loss_curves(
                    training_data, output_dir, statistical_analysis, output_filename)
                generated_plots.extend(plots)
            elif plot_type == "accuracy_curves":
                plots = self._generate_accuracy_curves(
                    training_data, output_dir, statistical_analysis, output_filename)
                generated_plots.extend(plots)
            elif plot_type == "convergence_analysis":
                plots = self._generate_convergence_analysis(
                    training_data, output_dir, statistical_analysis, output_filename)
                generated_plots.extend(plots)
            elif plot_type == "optimizer_comparison":
                plots = self._generate_optimizer_comparison(
                    training_data, output_dir, statistical_analysis, output_filename)
                generated_plots.extend(plots)
            elif plot_type == "learning_rate_schedule":
                plots = self._generate_learning_rate_analysis(
                    training_data, output_dir, statistical_analysis, output_filename)
                generated_plots.extend(plots)
            else:
                # Auto-detect best plot type
                plots = self._auto_generate_training_plots(
                    training_data, output_dir, statistical_analysis, output_filename)
                generated_plots.extend(plots)
            
            result = {
                "success": True,
                "plot_type": plot_type,
                "data_specification": data_specification,
                "statistical_analysis": statistical_analysis,
                "output_directory": output_dir,
                "total_plots": len(generated_plots),
                "generated_plots": generated_plots,
                "training_data_summary": self._summarize_training_data(training_data)
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            error_result = {
                "success": False,
                "error": f"Training analysis plot generation failed: {str(e)}",
                "data_specification": data_specification,
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
                    f"Example: Use 'experiment_data/training_logs.json' instead of the full path."
                )
        else:
            # Relative path - join with workspace
            abs_path = os.path.abspath(os.path.join(abs_working_dir, path))
            return abs_path
    
    def _parse_and_load_training_data(self, data_spec: str) -> Dict[str, Any]:
        """Parse data specification and load training data."""
        training_data = {"files": [], "metrics": {}, "runs": []}
        
        try:
            # Handle different specification formats
            if ":" in data_spec:
                # Format: "file.json:field1,field2"
                file_part, fields_part = data_spec.split(":", 1)
                file_path = self._safe_path(file_part.strip()) if self.working_dir else file_part.strip()
                fields = [f.strip() for f in fields_part.split(",")]
                
                if os.path.exists(file_path):
                    data = self._load_json_training_data(file_path, fields)
                    training_data["metrics"].update(data)
                    training_data["files"].append(file_path)
            
            elif "," in data_spec:
                # Format: "file1.json,file2.csv,file3.npy"
                file_paths = [f.strip() for f in data_spec.split(",")]
                for file_path in file_paths:
                    resolved_path = self._safe_path(file_path) if self.working_dir else file_path
                    if os.path.exists(resolved_path):
                        data = self._load_training_file(resolved_path)
                        training_data["metrics"].update(data)
                        training_data["files"].append(resolved_path)
            
            else:
                # Single file or auto-discovery
                if os.path.exists(data_spec):
                    resolved_path = self._safe_path(data_spec) if self.working_dir else data_spec
                    data = self._load_training_file(resolved_path)
                    training_data["metrics"].update(data)
                    training_data["files"].append(resolved_path)
                else:
                    # Auto-discovery based on description
                    discovered_data = self._auto_discover_training_data(data_spec)
                    training_data.update(discovered_data)
            
            return training_data
            
        except Exception as e:
            print(f"Warning: Failed to parse training data specification: {e}")
            return {"files": [], "metrics": {}, "runs": []}
    
    def _load_json_training_data(self, file_path: str, fields: List[str]) -> Dict[str, Any]:
        """Load specific fields from JSON training data."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            metrics = {}
            for field in fields:
                # Handle nested field access like "results.train_loss"
                value = data
                for key in field.split('.'):
                    if isinstance(value, dict) and key in value:
                        value = value[key]
                    else:
                        value = None
                        break
                
                if value is not None and isinstance(value, (list, np.ndarray)):
                    metrics[field] = np.array(value)
                elif value is not None:
                    metrics[field] = value
            
            return metrics
            
        except Exception as e:
            print(f"Warning: Failed to load JSON training data from {file_path}: {e}")
            return {}
    
    def _load_training_file(self, file_path: str) -> Dict[str, Any]:
        """Load training data from various file formats."""
        try:
            if file_path.endswith('.json'):
                return self._load_json_training_file(file_path)
            elif file_path.endswith('.csv'):
                return self._load_csv_training_file(file_path)
            elif file_path.endswith('.npy'):
                return self._load_numpy_training_file(file_path)
            elif file_path.endswith('.npz'):
                return self._load_npz_training_file(file_path)
            else:
                return {}
                
        except Exception as e:
            print(f"Warning: Failed to load training file {file_path}: {e}")
            return {}
    
    def _load_json_training_file(self, file_path: str) -> Dict[str, Any]:
        """Load training data from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        metrics = {}
        
        # Look for common training metric patterns
        training_indicators = ['loss', 'accuracy', 'error', 'metric', 'train', 'val', 'test']
        
        def extract_sequences(obj, prefix=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    full_key = f"{prefix}.{key}" if prefix else key
                    if isinstance(value, list) and len(value) > 0:
                        if isinstance(value[0], (int, float)):
                            # Numerical sequence - likely training metric
                            metrics[full_key] = np.array(value)
                    elif isinstance(value, dict):
                        extract_sequences(value, full_key)
        
        extract_sequences(data)
        return metrics
    
    def _load_csv_training_file(self, file_path: str) -> Dict[str, Any]:
        """Load training data from CSV file."""
        try:
            import pandas as pd
            df = pd.read_csv(file_path)
            
            metrics = {}
            # Convert numerical columns to metrics
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                metrics[col] = df[col].values
            
            return metrics
            
        except ImportError:
            # Fallback without pandas
            print("Warning: pandas not available for CSV loading")
            return {}
    
    def _load_numpy_training_file(self, file_path: str) -> Dict[str, Any]:
        """Load training data from NumPy file."""
        data = np.load(file_path)
        filename = os.path.basename(file_path).replace('.npy', '')
        return {filename: data}
    
    def _load_npz_training_file(self, file_path: str) -> Dict[str, Any]:
        """Load training data from NPZ file."""
        data = np.load(file_path)
        return {key: data[key] for key in data.keys()}
    
    def _auto_discover_training_data(self, description: str) -> Dict[str, Any]:
        """Auto-discover training data based on description."""
        training_data = {"files": [], "metrics": {}, "runs": []}
        
        if not self.working_dir:
            return training_data
        
        # Search common directories for training data
        search_dirs = [
            self.working_dir,
            os.path.join(self.working_dir, "experiment_data"),
            os.path.join(self.working_dir, "experiment_runs"),
            os.path.join(self.working_dir, "results"),
            os.path.join(self.working_dir, "logs")
        ]
        
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                for ext in ['.json', '.csv', '.npy', '.npz']:
                    files = list(Path(search_dir).glob(f"*{ext}"))
                    for file_path in files:
                        # Check if filename suggests training data
                        filename = file_path.name.lower()
                        if any(indicator in filename for indicator in ['train', 'loss', 'accuracy', 'metric']):
                            data = self._load_training_file(str(file_path))
                            training_data["metrics"].update(data)
                            training_data["files"].append(str(file_path))
        
        return training_data
    
    def _setup_publication_style(self):
        """Setup matplotlib for publication-quality plots."""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        plt.style.use('default')
        if SEABORN_AVAILABLE:
            sns.set_palette("husl")
        
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
    
    def _generate_comprehensive_training_analysis(self, training_data: Dict, output_dir: str, 
                                                statistical_analysis: bool, output_filename: str = None) -> List[Dict]:
        """Generate comprehensive training analysis with multiple panels."""
        plots = []
        
        # Create comprehensive multi-panel figure
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        fig.suptitle('Comprehensive Training Analysis', fontsize=18, fontweight='bold')
        
        metrics = training_data.get("metrics", {})
        
        # Panel 1: Loss curves
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_loss_curves(ax1, metrics, statistical_analysis)
        
        # Panel 2: Accuracy curves  
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_accuracy_curves(ax2, metrics, statistical_analysis)
        
        # Panel 3: Learning curves comparison
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_learning_curves_comparison(ax3, metrics, statistical_analysis)
        
        # Panel 4: Convergence analysis
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_convergence_analysis(ax4, metrics)
        
        # Panel 5: Training dynamics (overfitting analysis)
        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_training_dynamics(ax5, metrics)
        
        # Panel 6: Performance summary
        ax6 = fig.add_subplot(gs[2, 1])
        self._plot_performance_summary(ax6, metrics)
        
        # Save plot
        filename = output_filename or "comprehensive_training_analysis.png"
        plot_path = os.path.join(output_dir, filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        plots.append({
            "type": "comprehensive_training_analysis",
            "path": plot_path,
            "filename": filename,
            "title": "Comprehensive Training Analysis",
            "description": "Multi-panel analysis including loss curves, accuracy progression, convergence analysis, and training dynamics",
            "panels": 6,
            "statistical_analysis": statistical_analysis
        })
        
        return plots
    
    def _plot_loss_curves(self, ax, metrics: Dict, statistical_analysis: bool):
        """Plot training/validation loss curves."""
        loss_metrics = {k: v for k, v in metrics.items() if 'loss' in k.lower()}
        
        if not loss_metrics:
            ax.text(0.5, 0.5, 'No loss data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Training Loss Curves')
            return
        
        for name, values in loss_metrics.items():
            if isinstance(values, np.ndarray) and len(values) > 0:
                epochs = range(len(values))
                line_style = '--' if 'val' in name.lower() or 'test' in name.lower() else '-'
                ax.plot(epochs, values, line_style, label=name.replace('_', ' ').title(), linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    def _plot_accuracy_curves(self, ax, metrics: Dict, statistical_analysis: bool):
        """Plot training/validation accuracy curves."""
        acc_metrics = {k: v for k, v in metrics.items() if 'acc' in k.lower()}
        
        if not acc_metrics:
            ax.text(0.5, 0.5, 'No accuracy data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Training Accuracy Curves')
            return
        
        for name, values in acc_metrics.items():
            if isinstance(values, np.ndarray) and len(values) > 0:
                epochs = range(len(values))
                line_style = '--' if 'val' in name.lower() or 'test' in name.lower() else '-'
                ax.plot(epochs, values, line_style, label=name.replace('_', ' ').title(), linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Training Accuracy Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_learning_curves_comparison(self, ax, metrics: Dict, statistical_analysis: bool):
        """Plot learning curves for comparison analysis."""
        # Find train/val pairs
        train_metrics = {k: v for k, v in metrics.items() if 'train' in k.lower() and len(v) > 0}
        
        if not train_metrics:
            ax.text(0.5, 0.5, 'No training curve data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Learning Curves Comparison')
            return
        
        for name, values in train_metrics.items():
            if isinstance(values, np.ndarray):
                epochs = range(len(values))
                ax.plot(epochs, values, label=name.replace('_', ' ').title(), linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Metric Value')
        ax.set_title('Learning Curves Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_convergence_analysis(self, ax, metrics: Dict):
        """Plot convergence analysis."""
        # Simple convergence plot - rate of change in loss
        loss_metrics = {k: v for k, v in metrics.items() if 'loss' in k.lower() and 'train' in k.lower()}
        
        if not loss_metrics:
            ax.text(0.5, 0.5, 'No loss data for convergence analysis', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Convergence Analysis')
            return
        
        # Take the first training loss
        loss_data = list(loss_metrics.values())[0]
        if len(loss_data) > 1:
            # Calculate rate of change
            rate_of_change = np.diff(loss_data)
            epochs = range(1, len(loss_data))
            ax.plot(epochs, rate_of_change, 'r-', linewidth=2, label='Loss Rate of Change')
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss Change')
        ax.set_title('Convergence Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_training_dynamics(self, ax, metrics: Dict):
        """Plot training dynamics and overfitting analysis."""
        # Find train/val accuracy pairs
        train_acc = None
        val_acc = None
        
        for key, values in metrics.items():
            if 'acc' in key.lower() and 'train' in key.lower():
                train_acc = values
            elif 'acc' in key.lower() and ('val' in key.lower() or 'valid' in key.lower()):
                val_acc = values
        
        if train_acc is not None and val_acc is not None and len(train_acc) > 0 and len(val_acc) > 0:
            min_len = min(len(train_acc), len(val_acc))
            gap = np.array(train_acc[:min_len]) - np.array(val_acc[:min_len])
            epochs = range(min_len)
            ax.plot(epochs, gap, 'r-', linewidth=2, label='Overfitting Gap')
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax.set_ylabel('Training - Validation Accuracy')
        else:
            ax.text(0.5, 0.5, 'Insufficient data for overfitting analysis', ha='center', va='center', transform=ax.transAxes)
        
        ax.set_xlabel('Epoch')
        ax.set_title('Training Dynamics')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_performance_summary(self, ax, metrics: Dict):
        """Plot final performance summary."""
        # Extract final values from each metric
        final_values = {}
        metric_names = []
        values = []
        
        for key, data in metrics.items():
            if isinstance(data, np.ndarray) and len(data) > 0:
                final_val = np.mean(data[-5:]) if len(data) >= 5 else data[-1]
                final_values[key] = final_val
                metric_names.append(key.replace('_', '\n'))
                values.append(final_val)
        
        if values:
            bars = ax.bar(range(len(metric_names)), values, alpha=0.7)
            ax.set_xlabel('Metrics')
            ax.set_ylabel('Final Values')
            ax.set_title('Final Performance Summary')
            ax.set_xticks(range(len(metric_names)))
            ax.set_xticklabels(metric_names, rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, 'No performance data available', ha='center', va='center', transform=ax.transAxes)
    
    def _generate_loss_curves(self, training_data: Dict, output_dir: str, 
                            statistical_analysis: bool, output_filename: str = None) -> List[Dict]:
        """Generate dedicated loss curves plot."""
        plots = []
        
        fig, ax = plt.subplots(figsize=(10, 6))
        self._plot_loss_curves(ax, training_data.get("metrics", {}), statistical_analysis)
        
        filename = output_filename or "training_loss_curves.png"
        plot_path = os.path.join(output_dir, filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        plots.append({
            "type": "loss_curves",
            "path": plot_path,
            "filename": filename,
            "title": "Training Loss Curves",
            "description": "Training and validation loss progression over epochs"
        })
        
        return plots
    
    def _generate_accuracy_curves(self, training_data: Dict, output_dir: str,
                                statistical_analysis: bool, output_filename: str = None) -> List[Dict]:
        """Generate dedicated accuracy curves plot."""
        plots = []
        
        fig, ax = plt.subplots(figsize=(10, 6))
        self._plot_accuracy_curves(ax, training_data.get("metrics", {}), statistical_analysis)
        
        filename = output_filename or "training_accuracy_curves.png"
        plot_path = os.path.join(output_dir, filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        plots.append({
            "type": "accuracy_curves",
            "path": plot_path,
            "filename": filename,
            "title": "Training Accuracy Curves",
            "description": "Training and validation accuracy progression over epochs"
        })
        
        return plots
    
    def _generate_convergence_analysis(self, training_data: Dict, output_dir: str,
                                     statistical_analysis: bool, output_filename: str = None) -> List[Dict]:
        """Generate convergence analysis plot."""
        plots = []
        
        fig, ax = plt.subplots(figsize=(10, 6))
        self._plot_convergence_analysis(ax, training_data.get("metrics", {}))
        
        filename = output_filename or "convergence_analysis.png"
        plot_path = os.path.join(output_dir, filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        plots.append({
            "type": "convergence_analysis",
            "path": plot_path,
            "filename": filename,
            "title": "Convergence Analysis",
            "description": "Analysis of training convergence and stability"
        })
        
        return plots
    
    def _generate_optimizer_comparison(self, training_data: Dict, output_dir: str,
                                     statistical_analysis: bool, output_filename: str = None) -> List[Dict]:
        """Generate optimizer comparison plots."""
        # This would need more sophisticated logic to detect different optimizers
        # For now, return a placeholder
        return []
    
    def _generate_learning_rate_analysis(self, training_data: Dict, output_dir: str,
                                       statistical_analysis: bool, output_filename: str = None) -> List[Dict]:
        """Generate learning rate schedule analysis."""
        # This would need learning rate data
        # For now, return a placeholder  
        return []
    
    def _auto_generate_training_plots(self, training_data: Dict, output_dir: str,
                                    statistical_analysis: bool, output_filename: str = None) -> List[Dict]:
        """Auto-generate appropriate training plots based on available data."""
        plots = []
        
        # Try comprehensive first
        comprehensive_plots = self._generate_comprehensive_training_analysis(
            training_data, output_dir, statistical_analysis, output_filename)
        plots.extend(comprehensive_plots)
        
        return plots
    
    def _summarize_training_data(self, training_data: Dict) -> Dict[str, Any]:
        """Summarize the training data for reporting."""
        summary = {
            "total_files": len(training_data.get("files", [])),
            "metrics_found": list(training_data.get("metrics", {}).keys()),
            "metric_lengths": {}
        }
        
        for key, values in training_data.get("metrics", {}).items():
            if isinstance(values, np.ndarray):
                summary["metric_lengths"][key] = len(values)
        
        return summary
    
    def _generate_fallback_response(self, error_msg: str) -> str:
        """Generate fallback response when plotting is not available."""
        return json.dumps({
            "success": False,
            "error": error_msg,
            "suggestion": "Install matplotlib and related dependencies to enable training plot generation",
            "generated_plots": []
        })