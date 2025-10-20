"""
Experimental Results Extractor Tool - Extract authentic results from AI-Scientist experiments.

This tool extends the DataDiscoveryTool pattern to extract actual experimental results
from AI-Scientist runs and generate authentic results files, preventing data fabrication.
"""

import json
import os
import numpy as np
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional
from smolagents import Tool


class ExperimentalResultsExtractorTool(Tool):
    name = "experimental_results_extractor_tool"
    description = """
    Extract authentic experimental results from AI-Scientist runs and generate results files.
    
    This tool extends the DataDiscoveryTool pattern to focus on extracting actual experimental
    results rather than planning visualizations. It prevents data fabrication by:
    
    - Discovering actual experiments conducted by AI-Scientist-v2
    - Extracting real metrics, loss values, and performance numbers
    - Generating authentic results.csv and experiment_results.json files
    - Replacing fabricated data with genuine experimental findings
    - Providing detailed analysis of actual experimental setup and outcomes
    
    Key capabilities:
    - Auto-discover AI-Scientist experiment runs in workspace
    - Extract actual loss functions, optimizers, hyperparameters tested
    - Generate CSV data based on real experimental comparisons
    - Validate against fabricated results and replace with authentic data
    - Create experiment summaries based on actual findings
    
    Use this tool to ensure all experimental results in papers are grounded in
    actual experiments rather than hallucinated or template-based data.
    """
    
    inputs = {
        "workspace_dir": {
            "type": "string",
            "description": "Root workspace directory containing AI-Scientist runs"
        },
        "extract_mode": {
            "type": "string",
            "description": "Extraction mode: 'replace_fabricated' (default), 'extract_only', 'validate_existing'",
            "nullable": True
        },
        "output_format": {
            "type": "string", 
            "description": "Output format: 'csv', 'json', 'both' (default)",
            "nullable": True
        }
    }
    
    outputs = {
        "extraction_report": {
            "type": "string",
            "description": "Comprehensive report of experimental results extraction and authentic data generation"
        }
    }
    
    output_type = "string"

    def __init__(self, working_dir: Optional[str] = None):
        """Initialize Experimental Results Extractor Tool."""
        super().__init__()
        self.working_dir = os.path.abspath(working_dir) if working_dir else None
    
    def forward(self, workspace_dir: str, extract_mode: str = "replace_fabricated", 
                output_format: str = "both") -> str:
        """
        Extract authentic experimental results and generate results files.
        
        Args:
            workspace_dir: Root workspace directory
            extract_mode: How to handle existing results
            output_format: Format for output files
            
        Returns:
            JSON string with extraction report and generated files
        """
        try:
            work_dir = os.path.abspath(workspace_dir)
            if not os.path.exists(work_dir):
                return json.dumps({
                    "error": f"Workspace directory not found: {workspace_dir}",
                    "extraction_report": None
                })
            
            # Discover actual experiments using DataDiscoveryTool pattern
            experiment_findings = self._discover_experiments(work_dir)
            
            # Extract results from actual experiments
            extracted_results = self._extract_experimental_results(experiment_findings)
            
            # Generate authentic results files
            generated_files = self._generate_results_files(work_dir, extracted_results, output_format)
            
            # Validate against existing fabricated data if requested
            validation_results = {}
            if extract_mode in ["replace_fabricated", "validate_existing"]:
                validation_results = self._validate_against_fabricated(work_dir, extracted_results)
            
            # Compile comprehensive report
            report = self._generate_extraction_report(
                experiment_findings, extracted_results, generated_files, validation_results, work_dir
            )
            
            return json.dumps(report, indent=2)
            
        except Exception as e:
            return json.dumps({
                "error": f"Experimental results extraction failed: {str(e)}",
                "extraction_report": None
            })
    
    def _discover_experiments(self, workspace_dir: str) -> Dict[str, Any]:
        """Discover AI-Scientist experiments using DataDiscoveryTool pattern."""
        
        findings = {
            "experiment_runs": [],
            "experiment_dirs": [],
            "data_files": [],
            "plot_files": [],
            "total_experiments": 0
        }

        # Search for AI-Scientist run directories (extends DataDiscoveryTool pattern)
        search_patterns = [
            "experiment_runs/*/experiments/*/experiment_results/*",
            "*/experiment_*.npy",
            "*/experiment_*.py", 
            "experimental_plots/*.png"
        ]
        
        for pattern in search_patterns:
            full_pattern = os.path.join(workspace_dir, pattern)
            matches = glob.glob(full_pattern, recursive=True)
            
            for match in matches:
                if "experiment_results" in match and os.path.isdir(match):
                    findings["experiment_dirs"].append(match)
                elif match.endswith(".npy"):
                    findings["data_files"].append(match)
                elif match.endswith(".png"):
                    findings["plot_files"].append(match)
        
        findings["total_experiments"] = len(findings["experiment_dirs"])
        
        return findings
    
    def _extract_experimental_results(self, findings: Dict[str, Any]) -> Dict[str, Any]:
        """Extract actual results from discovered experiments."""
        
        results = {
            "experiment_summary": {},
            "extracted_metrics": {},
            "experimental_setup": {},
            "authentic_comparisons": []
        }
        
        # Process each experiment directory
        for exp_dir in findings.get("experiment_dirs", []):
            exp_results = self._process_experiment_directory(exp_dir)
            if exp_results:
                # Merge results
                exp_id = os.path.basename(exp_dir)
                results["experiment_summary"][exp_id] = exp_results
                
                # Extract metrics for aggregation
                for metric_name, values in exp_results.get("metrics", {}).items():
                    if metric_name not in results["extracted_metrics"]:
                        results["extracted_metrics"][metric_name] = []
                    results["extracted_metrics"][metric_name].extend(values)
        
        # Generate comparative analysis from actual experiments
        if results["extracted_metrics"]:
            results["authentic_comparisons"] = self._generate_authentic_comparisons(results["extracted_metrics"])
        
        return results
    
    def _process_experiment_directory(self, exp_dir: str) -> Optional[Dict[str, Any]]:
        """Process a single experiment directory to extract results."""
        
        exp_data = {
            "experiment_path": exp_dir,
            "experiment_type": "unknown",
            "metrics": {},
            "parameters": {}
        }
        
        # Look for experiment code to understand setup
        code_file = os.path.join(exp_dir, "experiment_code.py")
        if os.path.exists(code_file):
            code_analysis = self._analyze_experiment_code(code_file)
            exp_data.update(code_analysis)
        
        # Look for experimental data
        data_file = os.path.join(exp_dir, "experiment_data.npy")
        if os.path.exists(data_file):
            data_results = self._extract_data_metrics(data_file)
            exp_data["metrics"].update(data_results)
        
        return exp_data if exp_data["metrics"] else None
    
    def _analyze_experiment_code(self, code_file: str) -> Dict[str, Any]:
        """Analyze experiment code to understand setup (extends DataDiscoveryTool pattern)."""
        
        analysis = {
            "experiment_type": "supervised_learning",
            "parameters": {},
            "methods_tested": []
        }
        
        try:
            with open(code_file, 'r', encoding='utf-8') as f:
                code = f.read()
            
            # Extract experimental setup details
            if "MSELoss" in code or "L1Loss" in code or "SmoothL1Loss" in code:
                analysis["experiment_type"] = "loss_function_comparison"
                if "MSELoss" in code:
                    analysis["methods_tested"].append("MSE")
                if "L1Loss" in code:
                    analysis["methods_tested"].append("MAE") 
                if "SmoothL1Loss" in code:
                    analysis["methods_tested"].append("Huber")
            
            if "SGD" in code or "Adam" in code or "RMSprop" in code:
                analysis["experiment_type"] = "optimizer_comparison"
                if "SGD" in code:
                    analysis["methods_tested"].append("SGD")
                if "Adam" in code:
                    analysis["methods_tested"].append("Adam")
            
            # Extract hyperparameters
            import re
            lr_matches = re.findall(r'lr=([0-9.]+)', code)
            if lr_matches:
                analysis["parameters"]["learning_rates"] = [float(lr) for lr in lr_matches]
            
            return analysis
            
        except Exception:
            return analysis
    
    def _extract_data_metrics(self, data_file: str) -> Dict[str, List[float]]:
        """Extract metrics from experiment data files."""
        
        metrics = {}
        
        try:
            data = np.load(data_file, allow_pickle=True).item()
            
            if isinstance(data, dict):
                # Navigate nested data structure to find final metrics
                for category, category_data in data.items():
                    if isinstance(category_data, dict):
                        for method, method_data in category_data.items():
                            if isinstance(method_data, dict):
                                # Extract final losses
                                if "losses" in method_data:
                                    losses = method_data["losses"]
                                    if isinstance(losses, dict):
                                        if "train" in losses and losses["train"]:
                                            final_train_loss = losses["train"][-1]
                                            metrics[f"{method}_train_loss"] = [final_train_loss]
                                        if "val" in losses and losses["val"]:
                                            final_val_loss = losses["val"][-1]
                                            metrics[f"{method}_val_loss"] = [final_val_loss]
                                
                                # Extract accuracy if available
                                if "metrics" in method_data:
                                    method_metrics = method_data["metrics"]
                                    if isinstance(method_metrics, dict):
                                        if "train" in method_metrics and method_metrics["train"]:
                                            final_acc = method_metrics["train"][-1]
                                            if isinstance(final_acc, (int, float)):
                                                metrics[f"{method}_accuracy"] = [final_acc]
            
            return metrics
            
        except Exception:
            return {}
    
    def _generate_authentic_comparisons(self, metrics: Dict[str, List[float]]) -> List[Dict[str, Any]]:
        """Generate authentic comparative results from extracted metrics."""
        
        comparisons = []
        
        # Group metrics by method
        methods = {}
        for metric_name, values in metrics.items():
            if "_" in metric_name:
                method = metric_name.split("_")[0]
                metric_type = "_".join(metric_name.split("_")[1:])
                
                if method not in methods:
                    methods[method] = {}
                
                if values:
                    methods[method][metric_type] = sum(values) / len(values)  # Average
        
        # Create comparative analysis
        for method, method_metrics in methods.items():
            comparison = {
                "method": method.upper(),
                "metrics": method_metrics
            }
            comparisons.append(comparison)
        
        return comparisons
    
    def _generate_results_files(self, workspace_dir: str, results: Dict[str, Any], 
                               output_format: str) -> Dict[str, Any]:
        """Generate authentic results files based on extracted data."""
        
        generated_files = {
            "files_created": [],
            "csv_data": None,
            "json_data": None
        }
        
        writeup_dir = os.path.join(workspace_dir, "writeup_agent")
        os.makedirs(writeup_dir, exist_ok=True)
        
        # Generate CSV file
        if output_format in ["csv", "both"]:
            csv_data = self._create_csv_data(results)
            if csv_data:
                csv_path = os.path.join(writeup_dir, "results.csv")
                with open(csv_path, 'w', encoding='utf-8') as f:
                    f.write(csv_data)
                generated_files["files_created"].append(csv_path)
                generated_files["csv_data"] = csv_data
        
        # Generate JSON file
        if output_format in ["json", "both"]:
            json_data = self._create_json_data(results)
            if json_data:
                json_path = os.path.join(writeup_dir, "experiment_results.json")
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2)
                generated_files["files_created"].append(json_path)
                generated_files["json_data"] = json_data
        
        return generated_files
    
    def _create_csv_data(self, results: Dict[str, Any]) -> Optional[str]:
        """Create CSV data from authentic experimental results."""
        
        comparisons = results.get("authentic_comparisons", [])
        if not comparisons:
            return None
        
        # Create CSV based on actual experimental comparisons
        csv_lines = []
        
        # Determine header based on available metrics
        headers = ["Method"]
        metric_types = set()
        for comp in comparisons:
            metric_types.update(comp.get("metrics", {}).keys())
        
        # Add common metric headers
        common_metrics = ["train_loss", "val_loss", "accuracy"]
        for metric in common_metrics:
            if any(metric in str(metric_types) for metric in metric_types):
                headers.append(metric.replace("_", " ").title())
        
        csv_lines.append(",".join(headers))
        
        # Add data rows
        for comp in comparisons:
            method = comp["method"]
            row = [method]
            
            metrics = comp.get("metrics", {})
            for header in headers[1:]:  # Skip "Method" header
                metric_key = header.lower().replace(" ", "_")
                
                # Find matching metric
                value = None
                for metric_name, metric_value in metrics.items():
                    if metric_key in metric_name:
                        value = f"{metric_value:.4f}"
                        break
                
                row.append(value or "N/A")
            
            csv_lines.append(",".join(row))
        
        return "\n".join(csv_lines)
    
    def _create_json_data(self, results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create JSON data from authentic experimental results."""
        
        comparisons = results.get("authentic_comparisons", [])
        if not comparisons:
            return None
        
        json_data = {}
        
        for comp in comparisons:
            method = comp["method"]
            metrics = comp.get("metrics", {})
            
            # Use primary metric (e.g., validation loss) as the main value
            primary_value = None
            for metric_name, metric_value in metrics.items():
                if "val_loss" in metric_name:
                    primary_value = metric_value
                    break
            
            if primary_value is None and metrics:
                # Fallback to first available metric
                primary_value = list(metrics.values())[0]
            
            if primary_value is not None:
                json_data[f"{method} Loss Function"] = primary_value
        
        return json_data
    
    def _validate_against_fabricated(self, workspace_dir: str, extracted_results: Dict[str, Any]) -> Dict[str, Any]:
        """Check for fabricated data and recommend replacement."""
        
        validation = {
            "fabricated_files_detected": [],
            "fabrication_indicators": [],
            "replacement_recommendations": []
        }
        
        # Check for known fabricated files
        fabricated_patterns = [
            ("results.csv", ["SRTR", "70.0", "25.0", "45.0", "55.0"]),
            ("experiment_results.json", ["SRTR-Full", "0.7", "0.25", "0.45", "0.55"])
        ]
        
        writeup_dir = os.path.join(workspace_dir, "writeup_agent")
        
        for filename, indicators in fabricated_patterns:
            file_path = os.path.join(writeup_dir, filename)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check for fabrication indicators
                    found_indicators = [ind for ind in indicators if ind in content]
                    if found_indicators:
                        validation["fabricated_files_detected"].append(filename)
                        validation["fabrication_indicators"].extend(found_indicators)
                        validation["replacement_recommendations"].append(
                            f"Replace {filename} with authentic experimental data"
                        )
                except Exception:
                    pass
        
        return validation
    
    def _generate_extraction_report(self, findings: Dict[str, Any], results: Dict[str, Any], 
                                  generated_files: Dict[str, Any], validation: Dict[str, Any], 
                                  workspace_dir: str) -> Dict[str, Any]:
        """Generate comprehensive extraction report."""
        
        report = {
            "workspace_directory": workspace_dir,
            "extraction_summary": {
                "experiments_found": findings.get("total_experiments", 0),
                "metrics_extracted": len(results.get("extracted_metrics", {})),
                "comparisons_generated": len(results.get("authentic_comparisons", [])),
                "files_generated": len(generated_files.get("files_created", []))
            },
            "discovered_experiments": findings,
            "extracted_results": results,
            "generated_files": generated_files,
            "fabrication_validation": validation,
            "recommendations": []
        }
        
        # Generate recommendations
        if findings.get("total_experiments", 0) > 0:
            report["recommendations"].append("✅ Authentic experimental data successfully extracted")
            
            if generated_files.get("files_created"):
                report["recommendations"].append(f"📊 Generated {len(generated_files['files_created'])} authentic results files")
            
            if validation.get("fabricated_files_detected"):
                report["recommendations"].append("🚨 CRITICAL: Fabricated data detected - replace with authentic results")
                report["recommendations"].extend(validation.get("replacement_recommendations", []))
            else:
                report["recommendations"].append("🔍 No fabricated data detected")
        else:
            report["recommendations"].append("⚠️ No experimental data found - verify AI-Scientist execution")
        
        return report