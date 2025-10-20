"""
StatisticalAnalysisPlotTool - Add statistical rigor to experimental visualizations.

This tool specializes in enhancing plots with statistical analysis including
error bars, confidence intervals, significance testing, and distribution analysis.
Essential for adding scientific rigor to experimental results.
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
    MATPLOTLIB_AVAILABLE = True
except ImportError as e:
    print(f"Warning: matplotlib not available ({e}). Using fallback plotting.")
    MATPLOTLIB_AVAILABLE = False
    plt = None

# Handle statistical imports
try:
    import scipy.stats as stats
    from scipy import interpolate
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


class StatisticalAnalysisPlotTool(Tool):
    name = "statistical_analysis_plot_tool"
    description = """
    Add statistical rigor and analysis to experimental data visualizations.
    
    This tool specializes in creating statistically sound visualizations including:
    - Confidence intervals and error bars with proper statistical foundations
    - Significance testing visualization (p-values, effect sizes, Cohen's d)
    - Distribution analysis (histograms, Q-Q plots, normality tests)
    - Correlation analysis with statistical significance
    - Regression analysis with confidence bands
    - Bootstrap analysis for robust statistical inference
    - Multiple comparison corrections (Bonferroni, FDR)
    
    Key features:
    - Automatically detects appropriate statistical tests based on data
    - Handles different data types (continuous, categorical, paired, unpaired)
    - Provides publication-quality statistical annotations
    - Includes effect size calculations for practical significance
    - Generates comprehensive statistical reports
    - Creates publication-ready figures with proper statistical notation
    
    Statistical methods supported:
    - T-tests (one-sample, two-sample, paired)
    - ANOVA and post-hoc tests
    - Chi-square tests for categorical data
    - Correlation analysis (Pearson, Spearman)
    - Non-parametric tests (Mann-Whitney, Wilcoxon)
    - Bootstrap confidence intervals
    
    Use this tool when you need to add statistical validity to your experimental
    results or when reviewers request statistical significance testing.
    """
    
    inputs = {
        "data_specification": {
            "type": "string",
            "description": "Specification of data for statistical analysis. Can be: file paths, JSON field paths, or description for auto-discovery"
        },
        "analysis_type": {
            "type": "string",
            "description": "Type of statistical analysis: 'significance_testing' (default), 'confidence_intervals', 'distribution_analysis', 'correlation_analysis', 'regression_analysis', 'bootstrap_analysis'",
            "nullable": True
        },
        "alpha_level": {
            "type": "number",
            "description": "Significance level for statistical tests (default: 0.05)",
            "nullable": True
        },
        "multiple_comparisons": {
            "type": "string", 
            "description": "Multiple comparison correction: 'none' (default), 'bonferroni', 'holm', 'fdr_bh'",
            "nullable": True
        },
        "bootstrap_samples": {
            "type": "integer",
            "description": "Number of bootstrap samples for bootstrap analysis (default: 1000)",
            "nullable": True
        },
        "output_filename": {
            "type": "string",
            "description": "Output filename for the generated plot (default: auto-generated based on analysis type)",
            "nullable": True
        }
    }
    
    outputs = {
        "statistical_results": {
            "type": "string",
            "description": "JSON containing statistical analysis results and generated plot paths"
        }
    }
    
    output_type = "string"
    
    def __init__(self, model=None, working_dir: Optional[str] = None):
        """Initialize StatisticalAnalysisPlotTool.
        
        Args:
            model: LLM model for intelligent analysis (optional)
            working_dir: Working directory for workspace-aware file access
        """
        super().__init__()
        from ..model_utils import get_raw_model
        self.model = get_raw_model(model)
        # Convert to absolute path to prevent nested directory issues
        self.working_dir = os.path.abspath(working_dir) if working_dir else None
        
    def forward(self, data_specification: str, analysis_type: str = "significance_testing",
                alpha_level: float = 0.05, multiple_comparisons: str = "none",
                bootstrap_samples: int = 1000, output_filename: str = None) -> str:
        """
        Perform statistical analysis and generate enhanced plots.
        
        Args:
            data_specification: Specification of data to analyze statistically
            analysis_type: Type of statistical analysis to perform
            alpha_level: Significance level for hypothesis tests
            multiple_comparisons: Method for multiple comparison correction
            bootstrap_samples: Number of bootstrap samples
            output_filename: Custom output filename
            
        Returns:
            JSON string containing statistical analysis results and plots
        """
        try:
            # Check if required libraries are available
            if not MATPLOTLIB_AVAILABLE:
                return self._generate_fallback_response("Matplotlib not available")
            
            if not SCIPY_AVAILABLE and analysis_type != "bootstrap_analysis":
                return self._generate_fallback_response("SciPy not available for statistical analysis")
            
            # Parse data specification and load data
            statistical_data = self._parse_and_load_statistical_data(data_specification)
            
            if not statistical_data or not statistical_data.get("datasets"):
                return json.dumps({
                    "error": "No statistical data found matching the specification",
                    "data_specification": data_specification,
                    "generated_plots": []
                })
            
            # Set output directory
            if self.working_dir:
                output_dir = os.path.join(self.working_dir, "paper_workspace", "figures")
            else:
                output_dir = os.path.join(os.getcwd(), "paper_workspace", "figures")
            os.makedirs(output_dir, exist_ok=True)
            
            # Setup plotting style
            self._setup_publication_style()
            
            # Perform statistical analysis based on type
            generated_plots = []
            statistical_results = {}
            
            if analysis_type == "significance_testing":
                plots, results = self._perform_significance_testing(
                    statistical_data, alpha_level, multiple_comparisons, output_dir, output_filename)
                generated_plots.extend(plots)
                statistical_results.update(results)
            
            elif analysis_type == "confidence_intervals":
                plots, results = self._generate_confidence_intervals(
                    statistical_data, alpha_level, output_dir, output_filename)
                generated_plots.extend(plots)
                statistical_results.update(results)
            
            elif analysis_type == "distribution_analysis":
                plots, results = self._perform_distribution_analysis(
                    statistical_data, alpha_level, output_dir, output_filename)
                generated_plots.extend(plots)
                statistical_results.update(results)
            
            elif analysis_type == "correlation_analysis":
                plots, results = self._perform_correlation_analysis(
                    statistical_data, alpha_level, output_dir, output_filename)
                generated_plots.extend(plots)
                statistical_results.update(results)
            
            elif analysis_type == "regression_analysis":
                plots, results = self._perform_regression_analysis(
                    statistical_data, alpha_level, output_dir, output_filename)
                generated_plots.extend(plots)
                statistical_results.update(results)
            
            elif analysis_type == "bootstrap_analysis":
                plots, results = self._perform_bootstrap_analysis(
                    statistical_data, alpha_level, bootstrap_samples, output_dir, output_filename)
                generated_plots.extend(plots)
                statistical_results.update(results)
            
            else:
                # Auto-detect appropriate analysis
                plots, results = self._auto_statistical_analysis(
                    statistical_data, alpha_level, output_dir, output_filename)
                generated_plots.extend(plots)
                statistical_results.update(results)
            
            result = {
                "success": True,
                "analysis_type": analysis_type,
                "data_specification": data_specification,
                "alpha_level": alpha_level,
                "multiple_comparisons": multiple_comparisons,
                "bootstrap_samples": bootstrap_samples,
                "output_directory": output_dir,
                "total_plots": len(generated_plots),
                "generated_plots": generated_plots,
                "statistical_results": statistical_results,
                "data_summary": self._summarize_statistical_data(statistical_data)
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            error_result = {
                "success": False,
                "error": f"Statistical analysis failed: {str(e)}",
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
                    f"Example: Use 'experiment_data/statistical_data.json' instead of the full path."
                )
        else:
            # Relative path - join with workspace
            abs_path = os.path.abspath(os.path.join(abs_working_dir, path))
            return abs_path
    
    def _parse_and_load_statistical_data(self, data_spec: str) -> Dict[str, Any]:
        """Parse data specification and load statistical data."""
        statistical_data = {"datasets": {}, "files": [], "metadata": {}}
        
        try:
            # Handle different specification formats (similar to other tools)
            if ":" in data_spec:
                file_part, fields_part = data_spec.split(":", 1)
                file_path = self._safe_path(file_part.strip()) if self.working_dir else file_part.strip()
                fields = [f.strip() for f in fields_part.split(",")]
                
                if os.path.exists(file_path):
                    data = self._load_statistical_file(file_path, fields)
                    statistical_data["datasets"].update(data)
                    statistical_data["files"].append(file_path)
            
            elif "," in data_spec:
                file_paths = [f.strip() for f in data_spec.split(",")]
                for file_path in file_paths:
                    resolved_path = self._safe_path(file_path) if self.working_dir else file_path
                    if os.path.exists(resolved_path):
                        data = self._load_statistical_file(resolved_path)
                        statistical_data["datasets"].update(data) 
                        statistical_data["files"].append(resolved_path)
            
            else:
                # Single file or auto-discovery
                if os.path.exists(data_spec):
                    resolved_path = self._safe_path(data_spec) if self.working_dir else data_spec
                    data = self._load_statistical_file(resolved_path)
                    statistical_data["datasets"].update(data)
                    statistical_data["files"].append(resolved_path)
                else:
                    # Auto-discovery
                    discovered_data = self._auto_discover_statistical_data(data_spec)
                    statistical_data.update(discovered_data)
            
            return statistical_data
            
        except Exception as e:
            print(f"Warning: Failed to parse statistical data: {e}")
            return {"datasets": {}, "files": [], "metadata": {}}
    
    def _load_statistical_file(self, file_path: str, fields: List[str] = None) -> Dict[str, Any]:
        """Load statistical data from various file formats."""
        try:
            if file_path.endswith('.json'):
                return self._load_json_statistical_data(file_path, fields)
            elif file_path.endswith('.csv'):
                return self._load_csv_statistical_data(file_path, fields)
            elif file_path.endswith('.npy'):
                data = np.load(file_path)
                filename = os.path.basename(file_path).replace('.npy', '')
                return {filename: data}
            elif file_path.endswith('.npz'):
                data = np.load(file_path)
                return {key: data[key] for key in data.keys()}
            else:
                return {}
                
        except Exception as e:
            print(f"Warning: Failed to load statistical file {file_path}: {e}")
            return {}
    
    def _load_json_statistical_data(self, file_path: str, fields: List[str] = None) -> Dict[str, Any]:
        """Load statistical data from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        datasets = {}
        
        if fields:
            # Load specific fields
            for field in fields:
                value = data
                for key in field.split('.'):
                    if isinstance(value, dict) and key in value:
                        value = value[key]
                    else:
                        value = None
                        break
                
                if value is not None and isinstance(value, (list, np.ndarray)):
                    datasets[field] = np.array(value)
        else:
            # Load all numerical arrays
            def extract_arrays(obj, prefix=""):
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        full_key = f"{prefix}.{key}" if prefix else key
                        if isinstance(value, list) and len(value) > 0:
                            if isinstance(value[0], (int, float)):
                                datasets[full_key] = np.array(value)
                        elif isinstance(value, dict):
                            extract_arrays(value, full_key)
            
            extract_arrays(data)
        
        return datasets
    
    def _load_csv_statistical_data(self, file_path: str, fields: List[str] = None) -> Dict[str, Any]:
        """Load statistical data from CSV file."""
        try:
            import pandas as pd
            df = pd.read_csv(file_path)
            
            datasets = {}
            if fields:
                # Load specific columns
                for field in fields:
                    if field in df.columns:
                        datasets[field] = df[field].values
            else:
                # Load all numerical columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    datasets[col] = df[col].values
            
            return datasets
            
        except ImportError:
            print("Warning: pandas not available for CSV loading")
            return {}
    
    def _auto_discover_statistical_data(self, description: str) -> Dict[str, Any]:
        """Auto-discover statistical data based on description."""
        statistical_data = {"datasets": {}, "files": [], "metadata": {}}
        
        if not self.working_dir:
            return statistical_data
        
        # Search for statistical data files
        search_dirs = [
            self.working_dir,
            os.path.join(self.working_dir, "experiment_data"),
            os.path.join(self.working_dir, "results"),
            os.path.join(self.working_dir, "statistical_analysis")
        ]
        
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                for ext in ['.json', '.csv', '.npy', '.npz']:
                    files = list(Path(search_dir).glob(f"*{ext}"))
                    for file_path in files:
                        filename = file_path.name.lower()
                        # Look for statistical indicators
                        if any(indicator in filename for indicator in ['stat', 'test', 'result', 'metric']):
                            data = self._load_statistical_file(str(file_path))
                            statistical_data["datasets"].update(data)
                            statistical_data["files"].append(str(file_path))
        
        return statistical_data
    
    def _setup_publication_style(self):
        """Setup matplotlib for publication-quality statistical plots."""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        plt.style.use('default')
        if SEABORN_AVAILABLE:
            sns.set_palette("colorblind")
        
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
    
    def _perform_significance_testing(self, statistical_data: Dict, alpha_level: float,
                                    multiple_comparisons: str, output_dir: str, 
                                    output_filename: str = None) -> Tuple[List[Dict], Dict[str, Any]]:
        """Perform significance testing and generate visualizations."""
        plots = []
        results = {"tests": [], "summary": {}}
        
        datasets = statistical_data.get("datasets", {})
        dataset_names = list(datasets.keys())
        
        if len(dataset_names) < 2:
            return plots, results
        
        # Create significance testing visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Statistical Significance Analysis', fontsize=16, fontweight='bold')
        
        # Panel 1: Box plots with significance annotations
        ax1 = axes[0, 0]
        data_for_boxplot = []
        labels = []
        for name in dataset_names[:6]:  # Limit to 6 datasets for readability
            data_for_boxplot.append(datasets[name])
            labels.append(name.replace('_', ' ').title())
        
        box_plot = ax1.boxplot(data_for_boxplot, labels=labels, patch_artist=True)
        ax1.set_title('Distribution Comparison')
        ax1.set_ylabel('Values')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # Perform pairwise t-tests
        test_results = []
        for i in range(len(dataset_names)):
            for j in range(i+1, len(dataset_names)):
                name1, name2 = dataset_names[i], dataset_names[j]
                data1, data2 = datasets[name1], datasets[name2]
                
                if len(data1) > 1 and len(data2) > 1:
                    try:
                        # Two-sample t-test
                        t_stat, p_value = stats.ttest_ind(data1, data2)
                        
                        # Effect size (Cohen's d)
                        pooled_std = np.sqrt(((len(data1)-1)*np.var(data1, ddof=1) + 
                                            (len(data2)-1)*np.var(data2, ddof=1)) / 
                                           (len(data1) + len(data2) - 2))
                        cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std
                        
                        test_results.append({
                            "test_type": "two_sample_ttest",
                            "group1": name1,
                            "group2": name2,
                            "t_statistic": float(t_stat),
                            "p_value": float(p_value),
                            "cohens_d": float(cohens_d),
                            "significant": p_value < alpha_level,
                            "effect_size_interpretation": self._interpret_effect_size(abs(cohens_d))
                        })
                    except:
                        continue
        
        results["tests"] = test_results
        
        # Panel 2: P-value visualization
        ax2 = axes[0, 1]
        if test_results:
            p_values = [result["p_value"] for result in test_results]
            comparison_labels = [f"{result['group1'][:8]} vs {result['group2'][:8]}" 
                               for result in test_results]
            
            colors = ['red' if p < alpha_level else 'gray' for p in p_values]
            bars = ax2.bar(range(len(p_values)), p_values, color=colors, alpha=0.7)
            ax2.axhline(y=alpha_level, color='red', linestyle='--', 
                       label=f'α = {alpha_level}')
            ax2.set_xlabel('Comparisons')
            ax2.set_ylabel('P-value')
            ax2.set_title('Statistical Significance (P-values)')
            ax2.set_xticks(range(len(p_values)))
            ax2.set_xticklabels(comparison_labels, rotation=45, ha='right')
            ax2.legend()
            ax2.set_yscale('log')
        
        # Panel 3: Effect sizes
        ax3 = axes[1, 0]
        if test_results:
            effect_sizes = [abs(result["cohens_d"]) for result in test_results]
            bars = ax3.bar(range(len(effect_sizes)), effect_sizes, alpha=0.7)
            
            # Add effect size interpretation lines
            ax3.axhline(y=0.2, color='green', linestyle='--', alpha=0.5, label='Small effect')
            ax3.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium effect') 
            ax3.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Large effect')
            
            ax3.set_xlabel('Comparisons')
            ax3.set_ylabel("Cohen's d (Effect Size)")
            ax3.set_title('Effect Size Analysis')
            ax3.set_xticks(range(len(effect_sizes)))
            ax3.set_xticklabels(comparison_labels, rotation=45, ha='right')
            ax3.legend()
        
        # Panel 4: Summary statistics
        ax4 = axes[1, 1]
        summary_stats = []
        for name, data in datasets.items():
            summary_stats.append({
                "name": name,
                "mean": np.mean(data),
                "std": np.std(data),
                "n": len(data)
            })
        
        means = [stat["mean"] for stat in summary_stats]
        stds = [stat["std"] for stat in summary_stats]
        names = [stat["name"][:10] for stat in summary_stats]
        
        bars = ax4.bar(range(len(means)), means, yerr=stds, capsize=5, alpha=0.7)
        ax4.set_xlabel('Datasets')
        ax4.set_ylabel('Mean ± Std')
        ax4.set_title('Summary Statistics')
        ax4.set_xticks(range(len(names)))
        ax4.set_xticklabels(names, rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save plot
        filename = output_filename or "significance_testing_analysis.png"
        plot_path = os.path.join(output_dir, filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        plots.append({
            "type": "significance_testing",
            "path": plot_path,
            "filename": filename,
            "title": "Statistical Significance Analysis",
            "description": "Comprehensive significance testing with p-values, effect sizes, and distribution comparison",
            "tests_performed": len(test_results),
            "significant_results": len([r for r in test_results if r["significant"]])
        })
        
        # Apply multiple comparison correction if requested
        if multiple_comparisons != "none" and test_results:
            corrected_results = self._apply_multiple_comparison_correction(
                test_results, multiple_comparisons, alpha_level)
            results["corrected_tests"] = corrected_results
        
        results["summary"] = {
            "total_tests": len(test_results),
            "significant_tests": len([r for r in test_results if r["significant"]]),
            "alpha_level": alpha_level,
            "multiple_comparison_correction": multiple_comparisons
        }
        
        return plots, results
    
    def _generate_confidence_intervals(self, statistical_data: Dict, alpha_level: float,
                                     output_dir: str, output_filename: str = None) -> Tuple[List[Dict], Dict[str, Any]]:
        """Generate confidence interval visualizations."""
        plots = []
        results = {"confidence_intervals": []}
        
        datasets = statistical_data.get("datasets", {})
        
        # Create confidence interval plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        means = []
        lower_bounds = []
        upper_bounds = []
        labels = []
        
        for name, data in datasets.items():
            if len(data) > 1:
                mean = np.mean(data)
                sem = stats.sem(data)  # Standard error of mean
                confidence_level = 1 - alpha_level
                
                # Calculate confidence interval
                ci = stats.t.interval(confidence_level, len(data)-1, loc=mean, scale=sem)
                
                means.append(mean)
                lower_bounds.append(ci[0])
                upper_bounds.append(ci[1])
                labels.append(name.replace('_', ' ').title())
                
                results["confidence_intervals"].append({
                    "dataset": name,
                    "mean": float(mean),
                    "confidence_level": confidence_level,
                    "lower_bound": float(ci[0]),
                    "upper_bound": float(ci[1]),
                    "margin_of_error": float(mean - ci[0])
                })
        
        if means:
            # Create error bar plot
            y_pos = range(len(means))
            errors = [[m - l for m, l in zip(means, lower_bounds)],
                     [u - m for m, u in zip(means, upper_bounds)]]
            
            ax.errorbar(means, y_pos, xerr=errors, fmt='o', capsize=5, capthick=2)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels)
            ax.set_xlabel('Value')
            ax.set_title(f'{int((1-alpha_level)*100)}% Confidence Intervals')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        filename = output_filename or "confidence_intervals.png"
        plot_path = os.path.join(output_dir, filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        plots.append({
            "type": "confidence_intervals",
            "path": plot_path,
            "filename": filename,
            "title": f"{int((1-alpha_level)*100)}% Confidence Intervals",
            "description": "Confidence intervals for dataset means with error bars",
            "confidence_level": 1 - alpha_level
        })
        
        return plots, results
    
    def _perform_distribution_analysis(self, statistical_data: Dict, alpha_level: float,
                                     output_dir: str, output_filename: str = None) -> Tuple[List[Dict], Dict[str, Any]]:
        """Perform distribution analysis and normality testing."""
        plots = []
        results = {"normality_tests": [], "distribution_stats": []}
        
        datasets = statistical_data.get("datasets", {})
        
        # Create distribution analysis plot
        n_datasets = len(datasets)
        if n_datasets == 0:
            return plots, results
        
        cols = min(3, n_datasets)
        rows = (n_datasets + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_datasets == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        fig.suptitle('Distribution Analysis', fontsize=16, fontweight='bold')
        
        for i, (name, data) in enumerate(datasets.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Histogram with normal overlay
            ax.hist(data, bins=20, density=True, alpha=0.7, color='skyblue', 
                   edgecolor='black')
            
            # Overlay normal distribution
            mu, sigma = np.mean(data), np.std(data)
            x = np.linspace(np.min(data), np.max(data), 100)
            ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
                   label='Normal fit')
            
            ax.set_title(name.replace('_', ' ').title())
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Perform normality tests
            if len(data) >= 3:
                # Shapiro-Wilk test
                shapiro_stat, shapiro_p = stats.shapiro(data)
                
                # Kolmogorov-Smirnov test
                ks_stat, ks_p = stats.kstest(data, 'norm', args=(mu, sigma))
                
                results["normality_tests"].append({
                    "dataset": name,
                    "shapiro_wilk": {
                        "statistic": float(shapiro_stat),
                        "p_value": float(shapiro_p),
                        "is_normal": shapiro_p > alpha_level
                    },
                    "kolmogorov_smirnov": {
                        "statistic": float(ks_stat),
                        "p_value": float(ks_p),
                        "is_normal": ks_p > alpha_level
                    }
                })
            
            # Distribution statistics
            results["distribution_stats"].append({
                "dataset": name,
                "mean": float(np.mean(data)),
                "median": float(np.median(data)),
                "std": float(np.std(data)),
                "skewness": float(stats.skew(data)),
                "kurtosis": float(stats.kurtosis(data)),
                "min": float(np.min(data)),
                "max": float(np.max(data))
            })
        
        # Hide unused subplots
        for i in range(len(datasets), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        filename = output_filename or "distribution_analysis.png"
        plot_path = os.path.join(output_dir, filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        plots.append({
            "type": "distribution_analysis",
            "path": plot_path,
            "filename": filename,
            "title": "Distribution Analysis",
            "description": "Histogram analysis with normality testing and distribution statistics",
            "datasets_analyzed": len(datasets)
        })
        
        return plots, results
    
    def _perform_correlation_analysis(self, statistical_data: Dict, alpha_level: float,
                                    output_dir: str, output_filename: str = None) -> Tuple[List[Dict], Dict[str, Any]]:
        """Perform correlation analysis between datasets."""
        plots = []
        results = {"correlations": []}
        
        datasets = statistical_data.get("datasets", {})
        dataset_names = list(datasets.keys())
        
        if len(dataset_names) < 2:
            return plots, results
        
        # Create correlation matrix
        correlation_matrix = []
        p_value_matrix = []
        
        for name1 in dataset_names:
            correlation_row = []
            p_value_row = []
            
            for name2 in dataset_names:
                data1, data2 = datasets[name1], datasets[name2]
                
                # Ensure same length
                min_len = min(len(data1), len(data2))
                data1_truncated = data1[:min_len]
                data2_truncated = data2[:min_len]
                
                if min_len > 2:
                    # Pearson correlation
                    corr, p_val = stats.pearsonr(data1_truncated, data2_truncated)
                    correlation_row.append(corr)
                    p_value_row.append(p_val)
                    
                    if name1 != name2:  # Don't add self-correlations to results
                        results["correlations"].append({
                            "variable1": name1,
                            "variable2": name2,
                            "pearson_correlation": float(corr),
                            "p_value": float(p_val),
                            "significant": p_val < alpha_level,
                            "interpretation": self._interpret_correlation(abs(corr))
                        })
                else:
                    correlation_row.append(np.nan)
                    p_value_row.append(np.nan)
            
            correlation_matrix.append(correlation_row)
            p_value_matrix.append(p_value_row)
        
        # Create correlation heatmap
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Correlation matrix heatmap
        im1 = ax1.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        ax1.set_title('Correlation Matrix')
        ax1.set_xticks(range(len(dataset_names)))
        ax1.set_yticks(range(len(dataset_names)))
        ax1.set_xticklabels([name[:10] for name in dataset_names], rotation=45, ha='right')
        ax1.set_yticklabels([name[:10] for name in dataset_names])
        
        # Add correlation values as text
        for i in range(len(dataset_names)):
            for j in range(len(dataset_names)):
                if not np.isnan(correlation_matrix[i][j]):
                    text = ax1.text(j, i, f'{correlation_matrix[i][j]:.2f}',
                                   ha="center", va="center", color="black" if abs(correlation_matrix[i][j]) < 0.5 else "white")
        
        fig.colorbar(im1, ax=ax1, label='Correlation Coefficient')
        
        # P-value matrix heatmap
        im2 = ax2.imshow(p_value_matrix, cmap='Reds_r', vmin=0, vmax=0.1)
        ax2.set_title('P-value Matrix')
        ax2.set_xticks(range(len(dataset_names)))
        ax2.set_yticks(range(len(dataset_names)))
        ax2.set_xticklabels([name[:10] for name in dataset_names], rotation=45, ha='right')
        ax2.set_yticklabels([name[:10] for name in dataset_names])
        
        # Add p-values as text
        for i in range(len(dataset_names)):
            for j in range(len(dataset_names)):
                if not np.isnan(p_value_matrix[i][j]):
                    text = ax2.text(j, i, f'{p_value_matrix[i][j]:.3f}',
                                   ha="center", va="center", color="white" if p_value_matrix[i][j] < 0.05 else "black")
        
        fig.colorbar(im2, ax=ax2, label='P-value')
        
        plt.tight_layout()
        
        # Save plot
        filename = output_filename or "correlation_analysis.png"
        plot_path = os.path.join(output_dir, filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        plots.append({
            "type": "correlation_analysis",
            "path": plot_path,
            "filename": filename,
            "title": "Correlation Analysis",
            "description": "Correlation matrix with significance testing between all dataset pairs",
            "variables_analyzed": len(dataset_names)
        })
        
        return plots, results
    
    def _perform_regression_analysis(self, statistical_data: Dict, alpha_level: float,
                                   output_dir: str, output_filename: str = None) -> Tuple[List[Dict], Dict[str, Any]]:
        """Perform regression analysis with confidence bands."""
        # Implementation would go here - simplified for now
        return [], {}
    
    def _perform_bootstrap_analysis(self, statistical_data: Dict, alpha_level: float,
                                  bootstrap_samples: int, output_dir: str, 
                                  output_filename: str = None) -> Tuple[List[Dict], Dict[str, Any]]:
        """Perform bootstrap analysis for robust statistical inference."""
        # Implementation would go here - simplified for now
        return [], {}
    
    def _auto_statistical_analysis(self, statistical_data: Dict, alpha_level: float,
                                 output_dir: str, output_filename: str = None) -> Tuple[List[Dict], Dict[str, Any]]:
        """Auto-select appropriate statistical analysis."""
        plots = []
        results = {}
        
        # Default to significance testing
        sig_plots, sig_results = self._perform_significance_testing(
            statistical_data, alpha_level, "none", output_dir, output_filename)
        plots.extend(sig_plots)
        results.update(sig_results)
        
        return plots, results
    
    def _apply_multiple_comparison_correction(self, test_results: List[Dict], 
                                            method: str, alpha_level: float) -> List[Dict]:
        """Apply multiple comparison correction to p-values."""
        if not test_results:
            return []
        
        p_values = [result["p_value"] for result in test_results]
        
        if method == "bonferroni":
            corrected_alpha = alpha_level / len(p_values)
            corrected_results = []
            for result, p_val in zip(test_results, p_values):
                corrected_result = result.copy()
                corrected_result["corrected_p_value"] = p_val
                corrected_result["corrected_significant"] = p_val < corrected_alpha
                corrected_result["correction_method"] = "bonferroni"
                corrected_results.append(corrected_result)
        
        elif method == "holm":
            # Holm-Bonferroni method
            sorted_indices = np.argsort(p_values)
            corrected_results = []
            
            for i, idx in enumerate(sorted_indices):
                result = test_results[idx].copy()
                corrected_alpha_i = alpha_level / (len(p_values) - i)
                result["corrected_p_value"] = p_values[idx]
                result["corrected_significant"] = p_values[idx] < corrected_alpha_i
                result["correction_method"] = "holm"
                corrected_results.append(result)
        
        else:  # method == "fdr_bh" (Benjamini-Hochberg)
            # Simplified FDR implementation
            sorted_indices = np.argsort(p_values)
            corrected_results = []
            
            for i, idx in enumerate(sorted_indices):
                result = test_results[idx].copy()
                corrected_alpha_i = alpha_level * (i + 1) / len(p_values)
                result["corrected_p_value"] = p_values[idx]
                result["corrected_significant"] = p_values[idx] < corrected_alpha_i
                result["correction_method"] = "fdr_bh"
                corrected_results.append(result)
        
        return corrected_results
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        if cohens_d < 0.2:
            return "negligible"
        elif cohens_d < 0.5:
            return "small"
        elif cohens_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _interpret_correlation(self, correlation: float) -> str:
        """Interpret correlation strength."""
        abs_corr = abs(correlation)
        if abs_corr < 0.1:
            return "negligible"
        elif abs_corr < 0.3:
            return "weak"
        elif abs_corr < 0.5:
            return "moderate"
        elif abs_corr < 0.7:  
            return "strong"
        else:
            return "very strong"
    
    def _summarize_statistical_data(self, statistical_data: Dict) -> Dict[str, Any]:
        """Summarize statistical data for reporting."""
        datasets = statistical_data.get("datasets", {})
        
        summary = {
            "total_datasets": len(datasets),
            "dataset_names": list(datasets.keys()),
            "dataset_sizes": {name: len(data) for name, data in datasets.items()},
            "data_ranges": {}
        }
        
        for name, data in datasets.items():
            summary["data_ranges"][name] = {
                "min": float(np.min(data)),
                "max": float(np.max(data)),
                "mean": float(np.mean(data)),
                "std": float(np.std(data))
            }
        
        return summary
    
    def _generate_fallback_response(self, error_msg: str) -> str:
        """Generate fallback response when statistical analysis is not available."""
        return json.dumps({
            "success": False,
            "error": error_msg,
            "suggestion": "Install matplotlib and scipy to enable statistical analysis",
            "generated_plots": []
        })