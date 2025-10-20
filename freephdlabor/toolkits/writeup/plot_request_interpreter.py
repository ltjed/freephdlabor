"""
Plot Request Interpreter - Convert natural language requests to plot specifications.
"""

import json
import os
from typing import Dict, Any, List, Optional

def interpret_request_with_llm(model, request: str, data_source: str = None, working_dir: str = None) -> Dict[str, Any]:
    """Use LLM to interpret natural language plotting requests."""
    
    # Auto-detect available data if not specified
    data_context = ""
    if working_dir:
        data_context = _scan_workspace_data(working_dir)
    
    interpretation_prompt = f"""
You are a scientific plotting expert. Parse this DETAILED natural language request for creating NeurIPS-quality plots.

User Request: "{request}"

Data Source: {data_source if data_source else "Auto-detect from workspace"}

Available Data Context:
{data_context}

The user request should contain COMPLETE specifications including exact data sources, statistical requirements, layout plans, and output specifications. Parse and extract:

1. **Exact Data Sources**: Extract specific file paths and data field specifications
   - File paths mentioned (e.g., "experiment_data/results.json")  
   - JSON field paths (e.g., "['training']['loss']")
   - Data patterns (e.g., "*.npy files", "columns 0-5")
   - Array/matrix specifications

2. **Detailed Plot Specifications**: Extract layout and content requirements
   - Specific plot types (training curves, comparisons, ablations, etc.)
   - Multi-panel layouts (2x2 grid, horizontal arrangement, etc.)
   - Subplot assignments (top-left: X, top-right: Y, etc.)
   - Statistical analysis requirements (error bars, confidence intervals, p-values)

3. **Statistical Requirements**: Extract statistical analysis specifications
   - Error bar types (standard deviation, standard error, confidence intervals)
   - Significance testing requirements (t-tests, ANOVA, p-values)
   - Trend analysis (regression lines, R² values)
   - Professional annotations

4. **Output Requirements**: Extract file naming and format specifications
   - Exact file names requested
   - Resolution/format requirements
   - Intended usage (paper Figure X, supplementary, etc.)
   - Multiple file generation plans

5. **Implementation Plan**: How to execute this specific request
   - Data loading strategy for specified sources
   - Plot generation sequence
   - Statistical computation approach
   - Layout implementation details

Respond in JSON format with EXACT specifications extracted:
{{
    "data_sources": {{
        "specific_files": ["exact/file/path1.json", "exact/file/path2.csv"],
        "json_fields": ["specific.field.path", "results['accuracy']"],
        "data_patterns": ["*.npy", "experiment_*_results.json"],
        "array_specs": ["columns 0-5", "rows 10:100"]
    }},
    "plot_specifications": {{
        "plot_types": ["training_curves", "comparison_plot", "ablation_study"],
        "layout_plan": "2x2 grid: top-left training loss, top-right validation accuracy, etc.",
        "subplot_details": ["panel_1: training curves with error bars", "panel_2: final comparison"],
        "statistical_analysis": ["95% confidence intervals", "t-test p-values", "trend lines with R²"]
    }},
    "output_requirements": {{
        "file_names": ["exact_filename1.png", "exact_filename2.png"],
        "resolution": "publication quality (300 DPI)",
        "intended_use": "paper Figure 2",
        "format_specs": "PNG with transparent background"
    }},
    "implementation_strategy": {{
        "data_loading": "load JSON field X, aggregate CSV columns Y-Z",
        "statistical_computation": "calculate std dev across runs, perform t-tests",
        "layout_execution": "create 2x2 matplotlib subplot grid",
        "styling": "apply publication formatting with consistent colors"
    }},
    "extracted_specifications": "detailed summary of exactly what was requested"
}}
"""
    
    try:
        # Use proper ChatMessage format for model calls
        from smolagents.types import ChatMessage
        messages = [ChatMessage(role="user", content=interpretation_prompt)]
        response = model.generate(messages).content
        plan = json.loads(response)
        return plan
    except Exception as e:
        # Fallback interpretation
        return {
            "plot_types": ["training_curves", "comparison"],
            "data_requirements": {
                "primary_data": "experimental results from workspace",
                "patterns": ["*.csv", "*.json", "*.npy"],
                "processing": "auto-detect and aggregate"
            },
            "statistical_features": ["error_bars", "trend_analysis"],
            "layout": {
                "style": "publication",
                "multi_panel": True,
                "figure_count": 2
            },
            "output_specs": {
                "figure_descriptions": ["Training analysis", "Method comparison"],
                "file_names": ["training_analysis.png", "method_comparison.png"]
            },
            "interpretation": f"Fallback interpretation due to error: {str(e)}"
        }

def interpret_request_heuristic(request: str, data_source: str = None) -> Dict[str, Any]:
    """Heuristic interpretation when LLM is not available."""
    request_lower = request.lower()
    
    # Detect plot types from keywords
    plot_types = []
    if any(word in request_lower for word in ["training", "loss", "accuracy", "epoch"]):
        plot_types.append("training_curves")
    if any(word in request_lower for word in ["comparison", "compare", "vs", "versus"]):
        plot_types.append("comparison")
    if any(word in request_lower for word in ["ablation", "parameter", "sweep"]):
        plot_types.append("ablation")
    if any(word in request_lower for word in ["distribution", "histogram", "box"]):
        plot_types.append("distribution")
    
    # Detect statistical features
    statistical_features = []
    if any(word in request_lower for word in ["error", "confidence", "std", "variance"]):
        statistical_features.append("error_bars")
    if any(word in request_lower for word in ["significance", "p-value", "statistical"]):
        statistical_features.append("significance_testing")
    if any(word in request_lower for word in ["trend", "regression", "fit"]):
        statistical_features.append("trend_analysis")
    
    # Detect style preferences
    style = "publication"
    if any(word in request_lower for word in ["professional", "publication", "paper"]):
        style = "publication"
    elif any(word in request_lower for word in ["presentation", "slide"]):
        style = "presentation"
    
    # Default to comprehensive analysis if nothing specific detected
    if not plot_types:
        plot_types = ["training_curves", "comparison"]
    if not statistical_features:
        statistical_features = ["error_bars"]
    
    return {
        "plot_types": plot_types,
        "data_requirements": {
            "primary_data": "experimental results",
            "patterns": ["*loss*", "*acc*", "*train*", "*valid*"],
            "processing": "auto-aggregate"
        },
        "statistical_features": statistical_features,
        "layout": {
            "style": style,
            "multi_panel": len(plot_types) > 1,
            "figure_count": len(plot_types)
        },
        "output_specs": {
            "figure_descriptions": [f"{plot_type.replace('_', ' ').title()}" for plot_type in plot_types],
            "file_names": [f"{plot_type}.png" for plot_type in plot_types]
        },
        "interpretation": f"Heuristic interpretation: detected {', '.join(plot_types)} with {', '.join(statistical_features)}"
    }

def _scan_workspace_data(working_dir: str) -> str:
    """Scan workspace to understand available data."""
    try:
        data_summary = []
        
        # Look for common data directories
        data_dirs = ["experiment_data", "experimental_plots", "results", "data"]
        for data_dir in data_dirs:
            full_path = os.path.join(working_dir, data_dir)
            if os.path.exists(full_path):
                files = os.listdir(full_path)
                data_summary.append(f"- {data_dir}/: {len(files)} files")
        
        # Look for common data files in root
        data_files = []
        for ext in ['.csv', '.json', '.npy', '.pkl', '.txt']:
            files = [f for f in os.listdir(working_dir) if f.endswith(ext)]
            if files:
                data_files.extend(files[:5])  # First 5 files of each type
        
        if data_files:
            data_summary.append(f"- Root data files: {', '.join(data_files)}")
        
        return "\n".join(data_summary) if data_summary else "No obvious data files detected"
        
    except Exception:
        return "Unable to scan workspace data"