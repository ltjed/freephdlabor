"""
DataDiscoveryTool - Discover and analyze experimental data files for plotting.

This tool analyzes the workspace to identify available experimental data files,
understand their structure, and provide recommendations for visualization.
Essential for data-driven figure generation in academic papers.
"""

import json
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from smolagents import Tool


class DataDiscoveryTool(Tool):
    name = "data_discovery_tool" 
    description = """
    Discover and analyze experimental data files in the workspace for visualization planning.
    
    This tool systematically explores the workspace to find experimental data files,
    analyzes their structure and content, and provides recommendations for plotting.
    
    Key capabilities:
    - Auto-discover data files (.json, .csv, .npy, .pkl) across common directories
    - Analyze data structure, dimensions, and content types
    - Identify time series, comparative metrics, and statistical data
    - Recommend appropriate plot types based on data characteristics
    - Extract metadata about experimental runs and configurations
    
    Output:
    - Comprehensive inventory of available data files
    - Data structure analysis and content summaries
    - Plot type recommendations for each dataset
    - Experimental metadata extraction
    
    Use this tool before creating any plots to understand what data is available
    and plan the most effective visualizations for your paper.
    """
    
    inputs = {
        "analysis_focus": {
            "type": "string",
            "description": "Focus area for data discovery: 'comprehensive' (default), 'training_data', 'comparison_data', 'statistical_data'",
            "nullable": True
        },
        "search_directories": {
            "type": "string", 
            "description": "Comma-separated list of directories to search (default: experiment_data,experimental_plots,results,data,.)",
            "nullable": True
        }
    }
    
    outputs = {
        "data_inventory": {
            "type": "string",
            "description": "JSON containing discovered data files, analysis, and recommendations"
        }
    }
    
    output_type = "string"
    
    def __init__(self, working_dir: Optional[str] = None):
        """Initialize DataDiscoveryTool.
        
        Args:
            working_dir: Working directory for workspace-aware file access
        """
        super().__init__()
        # Convert to absolute path to prevent nested directory issues
        self.working_dir = os.path.abspath(working_dir) if working_dir else None
        
    def forward(self, analysis_focus: str = "comprehensive", 
                search_directories: str = "experiment_data,experimental_plots,results,data,.") -> str:
        """
        Discover and analyze experimental data files for plotting.
        
        Args:
            analysis_focus: Focus area for the analysis
            search_directories: Directories to search for data files
            
        Returns:
            JSON string with data inventory and recommendations
        """
        try:
            # Parse search directories
            directories = [d.strip() for d in search_directories.split(",")]
            
            # Discover data files
            discovered_files = self._discover_data_files(directories)
            
            # Analyze each discovered file
            file_analyses = []
            for file_path in discovered_files:
                analysis = self._analyze_data_file(file_path)
                if analysis:
                    file_analyses.append(analysis)
            
            # Generate recommendations
            recommendations = self._generate_plot_recommendations(file_analyses, analysis_focus)
            
            # Create comprehensive inventory
            inventory = {
                "discovery_summary": {
                    "total_files_found": len(discovered_files),
                    "analyzed_files": len(file_analyses),
                    "search_directories": directories,
                    "analysis_focus": analysis_focus
                },
                "discovered_files": discovered_files,
                "file_analyses": file_analyses,
                "plot_recommendations": recommendations,
                "metadata": self._extract_experimental_metadata(file_analyses)
            }
            
            return json.dumps(inventory, indent=2)
            
        except Exception as e:
            error_result = {
                "error": f"Data discovery failed: {str(e)}",
                "discovered_files": [],
                "recommendations": []
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
                    f"Example: Use 'experiment_data/results.json' instead of the full path."
                )
        else:
            # Relative path - join with workspace  
            abs_path = os.path.abspath(os.path.join(abs_working_dir, path))
            return abs_path
    
    def _discover_data_files(self, directories: List[str]) -> List[str]:
        """Discover data files in specified directories."""
        discovered_files = []
        data_extensions = ['.json', '.csv', '.npy', '.pkl', '.npz']
        plot_extensions = ['.png', '.pdf', '.svg', '.jpg', '.jpeg']  # Add plot discovery
        all_extensions = data_extensions + plot_extensions
        
        for directory in directories:
            try:
                search_path = self._safe_path(directory) if self.working_dir else directory
                
                if os.path.exists(search_path) and os.path.isdir(search_path):
                    # Search for data files and plots with deep recursive search
                    for root, dirs, files in os.walk(search_path):
                        # Limit depth to avoid infinite recursion
                        level = root.replace(search_path, '').count(os.sep)
                        if level < 8:  # Max 8 levels deep for experimental directories
                            for file in files:
                                file_path = os.path.join(root, file)
                                for ext in all_extensions:
                                    if file.endswith(ext):
                                        discovered_files.append(file_path)
                                        break
                                    
            except Exception as e:
                # Skip directories that can't be accessed
                continue
        
        return list(set(discovered_files))  # Remove duplicates
    
    def _analyze_data_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Analyze a single data file."""
        try:
            file_info = {
                "file_path": file_path,
                "filename": os.path.basename(file_path),
                "extension": os.path.splitext(file_path)[1],
                "size_bytes": os.path.getsize(file_path),
                "data_type": "unknown",
                "structure": {},
                "content_summary": {},
                "plot_potential": []
            }
            
            # Analyze based on file type
            if file_path.endswith('.json'):
                analysis = self._analyze_json_file(file_path)
                file_info.update(analysis)
            elif file_path.endswith('.csv'):
                analysis = self._analyze_csv_file(file_path)
                file_info.update(analysis)
            elif file_path.endswith('.npy'):
                analysis = self._analyze_numpy_file(file_path)
                file_info.update(analysis)
            elif file_path.endswith(('.png', '.pdf', '.svg', '.jpg', '.jpeg')):
                analysis = self._analyze_plot_file(file_path)
                file_info.update(analysis)
            elif file_path.endswith('.npz'):
                analysis = self._analyze_npz_file(file_path)
                file_info.update(analysis)
            
            return file_info
            
        except Exception as e:
            return {
                "file_path": file_path,
                "error": f"Analysis failed: {str(e)}",
                "plot_potential": []
            }
    
    def _analyze_json_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze JSON data file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            analysis = {
                "data_type": "json",
                "structure": self._analyze_json_structure(data),
                "content_summary": self._summarize_json_content(data),
                "plot_potential": self._identify_json_plot_potential(data)
            }
            
            return analysis
            
        except Exception as e:
            return {"error": f"JSON analysis failed: {str(e)}"}
    
    def _analyze_csv_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze CSV data file."""
        try:
            # Try to use pandas if available
            try:
                import pandas as pd
                df = pd.read_csv(file_path)
                
                analysis = {
                    "data_type": "csv",
                    "structure": {
                        "rows": len(df),
                        "columns": len(df.columns),
                        "column_names": list(df.columns),
                        "column_types": {col: str(df[col].dtype) for col in df.columns}
                    },
                    "content_summary": self._summarize_dataframe(df),
                    "plot_potential": self._identify_csv_plot_potential(df)
                }
                
            except ImportError:
                # Fallback without pandas
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    
                analysis = {
                    "data_type": "csv",
                    "structure": {
                        "rows": len(lines) - 1,  # Assume header
                        "estimated_columns": len(lines[0].split(',')) if lines else 0
                    },
                    "content_summary": {"note": "Limited analysis without pandas"},
                    "plot_potential": ["time_series", "comparison"]
                }
            
            return analysis
            
        except Exception as e:
            return {"error": f"CSV analysis failed: {str(e)}"}
    
    def _analyze_numpy_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze NumPy data file."""
        try:
            data = np.load(file_path)
            
            analysis = {
                "data_type": "numpy",
                "structure": {
                    "shape": data.shape,
                    "dtype": str(data.dtype),
                    "dimensions": len(data.shape),
                    "size": data.size
                },
                "content_summary": {
                    "min": float(np.min(data)) if data.size > 0 else None,
                    "max": float(np.max(data)) if data.size > 0 else None,
                    "mean": float(np.mean(data)) if data.size > 0 else None,
                    "std": float(np.std(data)) if data.size > 0 else None
                },
                "plot_potential": self._identify_numpy_plot_potential(data)
            }
            
            return analysis
            
        except Exception as e:
            return {"error": f"NumPy analysis failed: {str(e)}"}
    
    def _analyze_npz_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze NumPy compressed archive."""
        try:
            data = np.load(file_path)
            arrays = {key: data[key] for key in data.keys()}
            
            analysis = {
                "data_type": "npz",
                "structure": {
                    "arrays": list(data.keys()),
                    "array_shapes": {key: arrays[key].shape for key in arrays},
                    "array_dtypes": {key: str(arrays[key].dtype) for key in arrays}
                },
                "content_summary": {
                    key: {
                        "min": float(np.min(arrays[key])) if arrays[key].size > 0 else None,
                        "max": float(np.max(arrays[key])) if arrays[key].size > 0 else None,
                        "mean": float(np.mean(arrays[key])) if arrays[key].size > 0 else None
                    } for key in arrays
                },
                "plot_potential": self._identify_npz_plot_potential(arrays)
            }
            
            return analysis
            
        except Exception as e:
            return {"error": f"NPZ analysis failed: {str(e)}"}
    
    def _analyze_plot_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze existing plot files for reuse in papers."""
        try:
            filename = os.path.basename(file_path)
            extension = os.path.splitext(file_path)[1]
            
            # Categorize plot type based on filename patterns
            plot_category = self._categorize_plot_from_filename(filename)
            
            # Determine experimental context from path
            experimental_context = self._extract_experimental_context(file_path)
            
            analysis = {
                "data_type": "existing_plot",
                "plot_category": plot_category,
                "experimental_context": experimental_context,
                "structure": {
                    "format": extension,
                    "ready_for_latex": extension in ['.png', '.pdf'],
                    "needs_conversion": extension in ['.svg', '.jpg', '.jpeg']
                },
                "content_summary": {
                    "plot_type": plot_category,
                    "source_directory": os.path.dirname(file_path),
                    "experimental_run": experimental_context.get("run_id"),
                    "experiment_type": experimental_context.get("experiment_type")
                },
                "plot_potential": ["existing_figure_reuse", plot_category]
            }
            
            return analysis
            
        except Exception as e:
            return {"error": f"Plot analysis failed: {str(e)}"}
    
    def _categorize_plot_from_filename(self, filename: str) -> str:
        """Categorize plot type based on filename patterns."""
        filename_lower = filename.lower()
        
        # Training-related plots
        if any(pattern in filename_lower for pattern in ['loss', 'train', 'accuracy', 'metric']):
            return "training_analysis"
        
        # Comparison plots
        if any(pattern in filename_lower for pattern in ['comparison', 'baseline', 'ablation', 'vs']):
            return "method_comparison"
        
        # Statistical plots
        if any(pattern in filename_lower for pattern in ['distribution', 'histogram', 'correlation', 'scatter']):
            return "statistical_analysis"
        
        # Experimental plots
        if any(pattern in filename_lower for pattern in ['experiment', 'result', 'performance']):
            return "experimental_results"
        
        # Learning curves
        if any(pattern in filename_lower for pattern in ['curve', 'learning', 'convergence']):
            return "learning_curves"
        
        # Architecture/model plots
        if any(pattern in filename_lower for pattern in ['architecture', 'model', 'network']):
            return "model_visualization"
        
        return "general_plot"
    
    def _extract_experimental_context(self, file_path: str) -> Dict[str, Any]:
        """Extract experimental context from file path."""
        path_parts = file_path.split(os.sep)
        context = {
            "run_id": None,
            "experiment_type": None,
            "configuration": None
        }
        
        # Look for experiment IDs in path
        for part in path_parts:
            if 'experiment_' in part:
                context["run_id"] = part
            if any(exp_type in part.lower() for exp_type in ['ablation', 'baseline', 'comparison']):
                context["experiment_type"] = part
            if any(config in part.lower() for config in ['lr_', 'batch_', 'epochs_']):
                context["configuration"] = part
                
        return context
    
    def _analyze_json_structure(self, data: Any) -> Dict[str, Any]:
        """Analyze JSON data structure."""
        if isinstance(data, dict):
            return {
                "type": "dictionary",
                "keys": list(data.keys()),
                "nested_structure": {k: type(v).__name__ for k, v in data.items()}
            }
        elif isinstance(data, list):
            return {
                "type": "list",
                "length": len(data),
                "element_types": list(set(type(item).__name__ for item in data[:10]))
            }
        else:
            return {"type": type(data).__name__}
    
    def _summarize_json_content(self, data: Any) -> Dict[str, Any]:
        """Summarize JSON content for plotting insights."""
        summary = {}
        
        if isinstance(data, dict):
            # Look for common patterns
            numerical_keys = []
            list_keys = []
            
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    numerical_keys.append(key)
                elif isinstance(value, list) and value:
                    list_keys.append(key)
                    if isinstance(value[0], (int, float)):
                        summary[f"{key}_stats"] = {
                            "length": len(value),
                            "min": min(value),
                            "max": max(value),
                            "type": "numerical_sequence"
                        }
            
            summary["numerical_keys"] = numerical_keys
            summary["sequence_keys"] = list_keys
        
        return summary
    
    def _identify_json_plot_potential(self, data: Any) -> List[str]:
        """Identify potential plot types for JSON data."""
        plot_types = []
        
        if isinstance(data, dict):
            # Check for time series patterns
            time_indicators = ['epoch', 'step', 'iteration', 'time', 'loss', 'accuracy', 'error']
            if any(key.lower() in [t.lower() for t in time_indicators] for key in data.keys()):
                plot_types.append("training_curves")
            
            # Check for comparison data
            if len([k for k, v in data.items() if isinstance(v, (int, float))]) > 2:
                plot_types.append("bar_comparison")
            
            # Check for nested experimental data
            nested_dicts = [v for v in data.values() if isinstance(v, dict)]
            if nested_dicts:
                plot_types.append("nested_comparison")
        
        return plot_types or ["general_visualization"]
    
    def _identify_csv_plot_potential(self, df) -> List[str]:
        """Identify potential plot types for CSV data."""
        plot_types = []
        
        try:
            # Check for time series
            time_cols = [col for col in df.columns if any(indicator in col.lower() 
                        for indicator in ['time', 'epoch', 'step', 'iteration'])]
            if time_cols:
                plot_types.append("time_series")
            
            # Check for numerical columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                plot_types.append("correlation_analysis")
                plot_types.append("distribution_analysis")
            
            # Check for categorical data
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                plot_types.append("categorical_comparison")
                
        except:
            plot_types = ["general_csv_plots"]
        
        return plot_types
    
    def _summarize_dataframe(self, df) -> Dict[str, Any]:
        """Summarize pandas DataFrame."""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            return {
                "numeric_columns": list(numeric_cols),
                "categorical_columns": list(df.select_dtypes(include=['object']).columns),
                "summary_stats": df[numeric_cols].describe().to_dict() if len(numeric_cols) > 0 else {}
            }
        except:
            return {"note": "Summary generation failed"}
    
    def _identify_numpy_plot_potential(self, data: np.ndarray) -> List[str]:
        """Identify potential plot types for NumPy data."""
        plot_types = []
        
        if len(data.shape) == 1:
            plot_types.append("line_plot")
            plot_types.append("histogram")
        elif len(data.shape) == 2:
            if data.shape[1] < 10:  # Multiple sequences
                plot_types.append("multi_line_plot")
            else:  # Matrix/heatmap
                plot_types.append("heatmap")
        
        return plot_types
    
    def _identify_npz_plot_potential(self, arrays: Dict[str, np.ndarray]) -> List[str]:
        """Identify potential plot types for NPZ data."""
        plot_types = []
        
        # Check for training-related arrays
        training_indicators = ['loss', 'accuracy', 'error', 'metric']
        for key in arrays.keys():
            if any(indicator in key.lower() for indicator in training_indicators):
                plot_types.append("training_analysis")
                break
        
        # Check for comparison data
        if len(arrays) > 1:
            plot_types.append("multi_array_comparison")
        
        return plot_types or ["array_visualization"]
    
    def _generate_plot_recommendations(self, file_analyses: List[Dict], focus: str) -> Dict[str, Any]:
        """Generate plot recommendations based on discovered data."""
        recommendations = {
            "priority_plots": [],
            "suggested_combinations": [],
            "focus_specific": {}
        }
        
        # Analyze all plot potentials
        all_potentials = []
        for analysis in file_analyses:
            all_potentials.extend(analysis.get("plot_potential", []))
        
        # Generate priority recommendations
        potential_counts = {}
        for potential in all_potentials:
            potential_counts[potential] = potential_counts.get(potential, 0) + 1
        
        # Sort by frequency and relevance
        sorted_potentials = sorted(potential_counts.items(), key=lambda x: x[1], reverse=True)
        
        for plot_type, count in sorted_potentials[:5]:
            recommendations["priority_plots"].append({
                "plot_type": plot_type,
                "data_files_supporting": count,
                "priority": "high" if count > 1 else "medium"
            })
        
        # Focus-specific recommendations
        if focus == "training_data":
            training_files = [f for f in file_analyses 
                            if any("training" in p for p in f.get("plot_potential", []))]
            recommendations["focus_specific"]["training_analysis"] = len(training_files)
        
        elif focus == "comparison_data":
            comparison_files = [f for f in file_analyses 
                              if any("comparison" in p for p in f.get("plot_potential", []))]
            recommendations["focus_specific"]["comparison_analysis"] = len(comparison_files)
        
        return recommendations
    
    def _extract_experimental_metadata(self, file_analyses: List[Dict]) -> Dict[str, Any]:
        """Extract experimental metadata from discovered files."""
        metadata = {
            "experiment_indicators": [],
            "potential_configurations": [],
            "data_time_ranges": []
        }
        
        # Look for experimental indicators in filenames and content
        for analysis in file_analyses:
            filename = analysis.get("filename", "")
            
            # Check for common experimental patterns
            if any(pattern in filename.lower() for pattern in 
                   ['train', 'test', 'valid', 'experiment', 'run', 'trial']):
                metadata["experiment_indicators"].append(filename)
            
            # Extract configuration hints
            if analysis.get("data_type") == "json":
                structure = analysis.get("structure", {})
                if "keys" in structure:
                    config_keys = [k for k in structure["keys"] 
                                 if any(term in k.lower() for term in 
                                       ['config', 'param', 'setting', 'hyperparameter'])]
                    metadata["potential_configurations"].extend(config_keys)
        
        return metadata