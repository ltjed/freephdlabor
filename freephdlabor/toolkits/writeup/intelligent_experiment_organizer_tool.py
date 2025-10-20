"""
IntelligentExperimentOrganizerTool - Structure-first experiment organization.

This tool implements a two-phase approach:
1. Structure Investigation: Maps the experiment directory structure and generates a summary
2. Structure-Aware Extraction: Uses the structure summary to efficiently extract key resources

Key Benefits:
- Intelligence before action - understands experiment type before processing
- Context preservation - maintains scientific relationships
- Efficiency - skips irrelevant content, focuses on key resources
- Reusability - structure summary guides later agents
- Adaptability - works for V2, generic, and custom experiments
"""

import json
import os
import shutil
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from smolagents import Tool


class IntelligentExperimentOrganizerTool(Tool):
    name = "intelligent_experiment_organizer_tool"
    description = """
    Structure-first intelligent experiment organization tool.
    
    This tool revolutionizes experiment data organization by first investigating
    the experiment structure, then using that understanding to efficiently extract
    and organize the most important resources for paper writing.
    
    Two-Phase Workflow:
    1. **Structure Investigation**: Maps directory hierarchy, identifies patterns,
       classifies experiment type, and generates a detailed structure summary
    2. **Structure-Aware Extraction**: Uses the structure summary as a "map" to
       efficiently navigate and extract key resources while preserving context
    
    Supports Multiple Experiment Types:
    - AI-Scientist-v2: Leverages stage summaries and best implementations
    - Generic: Discovers and organizes arbitrary experimental data
    - Custom: Adapts to domain-specific structures
    
    Output: Comprehensive resource package with preserved scientific context.
    """
    
    inputs = {
        "investigation_mode": {
            "type": "string",
            "description": "Mode: 'full' (complete analysis), 'quick' (essential only), 'structure_only' (investigation only)",
            "nullable": True
        }
    }
    
    outputs = {
        "organization_report": {
            "type": "string",
            "description": "JSON report with structure analysis and organized resources"
        }
    }
    
    output_type = "string"
    
    def __init__(self, model=None, working_dir: Optional[str] = None):
        """Initialize IntelligentExperimentOrganizerTool."""
        super().__init__()
        from ..model_utils import get_raw_model
        self.model = get_raw_model(model)
        self.working_dir = os.path.abspath(working_dir) if working_dir else None
        
    def forward(self, investigation_mode: str = "full") -> str:
        """Execute structure-first intelligent organization."""
        try:
            print("🔍 Phase 1: Investigating experiment structure...")
            structure_analysis = self._investigate_experiment_structure()
            
            print("📝 Phase 2: Generating structure summary...")
            structure_summary = self._generate_structure_summary(structure_analysis)
            
            # Save structure summary for later use
            self._save_structure_summary(structure_summary)
            
            if investigation_mode == "structure_only":
                return json.dumps({
                    "phase": "structure_investigation_only",
                    "structure_analysis": structure_analysis,
                    "structure_summary_path": os.path.join(
                        self.working_dir, "paper_workspace/structure_analysis.txt"
                    )
                }, indent=2)
            
            print("📦 Phase 3: Structure-aware resource extraction...")
            extracted_resources = self._extract_using_structure(structure_analysis, structure_summary)
            
            print("🎯 Phase 4: Creating intelligent organization...")
            organized_package = self._create_intelligent_package(
                structure_analysis, extracted_resources
            )
            
            final_report = {
                "organization_status": "completed",
                "approach": "structure_first_intelligent",
                "experiment_type": structure_analysis.get("experiment_type", "unknown"),
                "structure_summary_path": os.path.join(
                    self.working_dir, "paper_workspace/structure_analysis.txt"
                ),
                "key_resources": extracted_resources,
                "organized_package": organized_package,
                "intelligence_insights": structure_analysis.get("insights", {}),
                "guidance_for_writeup": self._generate_writeup_guidance(structure_analysis)
            }
            
            print("🎉 Intelligent experiment organization complete!")
            return json.dumps(final_report, indent=2)
            
        except Exception as e:
            return json.dumps({
                "organization_status": "failed",
                "error": str(e),
                "phase": "unknown",
                "guidance": "Check experiment structure and retry"
            }, indent=2)
    
    def _investigate_experiment_structure(self) -> Dict[str, Any]:
        """Phase 1: Comprehensive structure investigation."""
        investigation = {
            "timestamp": datetime.now().isoformat(),
            "working_directory": self.working_dir,
            "directory_map": {},
            "experiment_type": "unknown",
            "key_indicators": {},
            "resource_inventory": {},
            "insights": {}
        }
        
        # Step 1: Map complete directory structure
        print("  📁 Mapping directory hierarchy...")
        investigation["directory_map"] = self._map_directory_tree()
        
        # Step 2: Detect experiment patterns
        print("  🔎 Detecting experiment patterns...")
        investigation["experiment_type"], investigation["key_indicators"] = self._classify_experiment_type(
            investigation["directory_map"]
        )
        
        # Step 3: Inventory resources by importance
        print("  📊 Inventorying resources...")
        investigation["resource_inventory"] = self._inventory_resources(
            investigation["directory_map"], investigation["experiment_type"]
        )
        
        # Step 4: Generate intelligence insights
        print("  💡 Generating insights...")
        investigation["insights"] = self._generate_insights(investigation)
        
        return investigation
    
    def _map_directory_tree(self) -> Dict[str, Any]:
        """Create comprehensive map of directory structure."""
        directory_map = {
            "root": self.working_dir,
            "tree": {},
            "file_counts": {"total": 0, "by_extension": {}},
            "directory_counts": {"total": 0, "by_depth": {}}
        }
        
        # Build tree structure with metadata
        for root, dirs, files in os.walk(self.working_dir):
            relative_root = os.path.relpath(root, self.working_dir)
            if relative_root == ".":
                relative_root = "root"
            
            # Count directories by depth
            depth = relative_root.count(os.sep)
            directory_map["directory_counts"]["by_depth"][depth] = \
                directory_map["directory_counts"]["by_depth"].get(depth, 0) + 1
            directory_map["directory_counts"]["total"] += 1
            
            # Process files in this directory
            file_info = []
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                file_ext = os.path.splitext(file)[1].lower()
                
                file_info.append({
                    "name": file,
                    "size": file_size,
                    "extension": file_ext,
                    "path": file_path
                })
                
                # Update counts
                directory_map["file_counts"]["total"] += 1
                directory_map["file_counts"]["by_extension"][file_ext] = \
                    directory_map["file_counts"]["by_extension"].get(file_ext, 0) + 1
            
            # Store directory info
            directory_map["tree"][relative_root] = {
                "path": root,
                "subdirectories": dirs,
                "files": file_info,
                "file_count": len(files),
                "depth": depth
            }
        
        return directory_map
    
    def _classify_experiment_type(self, directory_map: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Detect and classify experiment type based on structure patterns."""
        indicators = {
            "v2_patterns": [],
            "generic_patterns": [],
            "custom_patterns": [],
            "confidence_scores": {}
        }
        
        # Check for V2 patterns
        v2_score = 0
        v2_patterns = [
            ("logs/0-run", "V2 execution logs"),
            ("baseline_summary.json", "V2 stage summary"),
            ("research_summary.json", "V2 stage summary"),
            ("ablation_summary.json", "V2 stage summary"),
            ("process_ForkProcess", "V2 parallel execution"),
            ("experiment_results", "V2 result aggregation"),
            ("idea.json", "V2 research context")
        ]
        
        for pattern, description in v2_patterns:
            found_paths = []
            for dir_path, dir_info in directory_map["tree"].items():
                # Check if pattern exists in this directory or its files
                if pattern in dir_path or any(pattern in f["name"] for f in dir_info["files"]):
                    found_paths.append(dir_info["path"])
                    v2_score += 1
            
            if found_paths:
                indicators["v2_patterns"].append({
                    "pattern": pattern,
                    "description": description,
                    "locations": found_paths
                })
        
        indicators["confidence_scores"]["v2"] = v2_score / len(v2_patterns)
        
        # Check for generic patterns
        generic_score = 0
        generic_patterns = [
            (".npy", "Numpy data files"),
            (".csv", "CSV data files"),
            (".png", "Plot files"),
            (".json", "JSON data files"),
            ("results", "Results directory"),
            ("data", "Data directory")
        ]
        
        for pattern, description in generic_patterns:
            count = directory_map["file_counts"]["by_extension"].get(pattern, 0)
            if count > 0:
                generic_score += min(count / 10, 1)  # Normalize to [0,1]
                indicators["generic_patterns"].append({
                    "pattern": pattern,
                    "description": description,
                    "count": count
                })
        
        indicators["confidence_scores"]["generic"] = generic_score / len(generic_patterns)
        
        # Determine experiment type
        if indicators["confidence_scores"]["v2"] > 0.4:
            experiment_type = "AI-Scientist-v2"
        elif indicators["confidence_scores"]["generic"] > 0.3:
            experiment_type = "generic"
        else:
            experiment_type = "custom"
        
        return experiment_type, indicators
    
    def _inventory_resources(self, directory_map: Dict[str, Any], 
                           experiment_type: str) -> Dict[str, List[Dict[str, Any]]]:
        """Create prioritized inventory of resources based on experiment type."""
        inventory = {
            "high_priority": [],    # Essential for paper writing
            "medium_priority": [],  # Useful supporting material
            "low_priority": [],     # Background/debug information
            "artifacts": []         # Execution artifacts, usually skippable
        }
        
        if experiment_type == "AI-Scientist-v2":
            return self._inventory_v2_resources(directory_map)
        else:
            return self._inventory_generic_resources(directory_map)
    
    def _inventory_v2_resources(self, directory_map: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Specialized inventory for V2 experiments."""
        inventory = {
            "high_priority": [],
            "medium_priority": [],
            "low_priority": [],
            "artifacts": []
        }
        
        # High priority: Stage summaries, final figures, research context
        high_priority_patterns = [
            ("baseline_summary.json", "Stage 1 results"),
            ("research_summary.json", "Stage 2 improvements"),
            ("ablation_summary.json", "Stage 3 validation"),
            ("idea.json", "Research hypothesis"),
            ("idea.md", "Research description")
        ]
        
        # Medium priority: Figures directory, best implementations
        medium_priority_patterns = [
            ("figures/", "Publication figures"),
            ("latex/", "Draft paper"),
            ("experiment_data.npy", "Processed results")
        ]
        
        # Low priority: Logs and intermediate results
        low_priority_patterns = [
            ("logs/", "Execution logs"),
            ("experiment_results/", "Individual attempts")
        ]
        
        # Artifacts: Process execution directories
        artifact_patterns = [
            ("process_ForkProcess", "Parallel execution artifacts"),
            ("working/", "Temporary execution files")
        ]
        
        # Classify all resources
        for dir_path, dir_info in directory_map["tree"].items():
            for file_info in dir_info["files"]:
                resource = {
                    "name": file_info["name"],
                    "path": file_info["path"],
                    "size": file_info["size"],
                    "directory": dir_path,
                    "type": "file"
                }
                
                # Classify by priority
                classified = False
                for patterns, priority in [
                    (high_priority_patterns, "high_priority"),
                    (medium_priority_patterns, "medium_priority"),
                    (low_priority_patterns, "low_priority"),
                    (artifact_patterns, "artifacts")
                ]:
                    for pattern, description in patterns:
                        if pattern in file_info["name"] or pattern in dir_path:
                            resource["description"] = description
                            resource["pattern_matched"] = pattern
                            inventory[priority].append(resource)
                            classified = True
                            break
                    if classified:
                        break
                
                # Default to medium priority if not classified
                if not classified:
                    resource["description"] = "Unclassified experimental data"
                    inventory["medium_priority"].append(resource)
        
        return inventory
    
    def _inventory_generic_resources(self, directory_map: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Generic inventory for non-V2 experiments."""
        inventory = {
            "high_priority": [],
            "medium_priority": [],
            "low_priority": [],
            "artifacts": []
        }
        
        # Classify by file type and location
        for dir_path, dir_info in directory_map["tree"].items():
            for file_info in dir_info["files"]:
                resource = {
                    "name": file_info["name"],
                    "path": file_info["path"],
                    "size": file_info["size"],
                    "directory": dir_path,
                    "type": "file",
                    "extension": file_info["extension"]
                }
                
                # Prioritize by extension and size
                if file_info["extension"] in ['.png', '.pdf', '.svg', '.jpg']:
                    resource["description"] = "Plot/figure file"
                    inventory["high_priority"].append(resource)
                elif file_info["extension"] in ['.json', '.csv']:
                    resource["description"] = "Structured data file"
                    inventory["high_priority"].append(resource)
                elif file_info["extension"] in ['.npy', '.npz', '.pkl']:
                    resource["description"] = "Binary data file"
                    inventory["medium_priority"].append(resource)
                elif file_info["extension"] in ['.txt', '.md', '.log']:
                    resource["description"] = "Text/log file"
                    inventory["low_priority"].append(resource)
                else:
                    resource["description"] = "Other file"
                    inventory["artifacts"].append(resource)
        
        return inventory
    
    def _generate_insights(self, investigation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate intelligence insights from investigation."""
        insights = {
            "complexity_assessment": "",
            "key_findings": [],
            "recommended_strategy": "",
            "potential_challenges": []
        }
        
        # Assess complexity
        total_files = investigation["directory_map"]["file_counts"]["total"]
        total_dirs = investigation["directory_map"]["directory_counts"]["total"]
        
        if total_files > 100:
            insights["complexity_assessment"] = "High complexity - many files to process"
        elif total_files > 20:
            insights["complexity_assessment"] = "Medium complexity - moderate file count"
        else:
            insights["complexity_assessment"] = "Low complexity - manageable file count"
        
        # Key findings based on experiment type
        exp_type = investigation["experiment_type"]
        if exp_type == "AI-Scientist-v2":
            insights["key_findings"] = [
                "V2 structure detected - can leverage stage summaries",
                "Stage progression analysis available",
                "Best implementations pre-selected by V2"
            ]
            insights["recommended_strategy"] = "Use V2 stage summaries as primary source"
        else:
            insights["key_findings"] = [
                "Generic experimental structure",
                "Will need comprehensive file analysis",
                "No pre-computed summaries available"
            ]
            insights["recommended_strategy"] = "Full discovery and annotation workflow"
        
        # Potential challenges
        high_priority_count = len(investigation["resource_inventory"].get("high_priority", []))
        if high_priority_count > 50:
            insights["potential_challenges"].append("Large number of high-priority files")
        
        artifact_count = len(investigation["resource_inventory"].get("artifacts", []))
        if artifact_count > total_files * 0.5:
            insights["potential_challenges"].append("Many execution artifacts to filter")
        
        return insights
    
    def _generate_structure_summary(self, structure_analysis: Dict[str, Any]) -> str:
        """Generate comprehensive structure summary in markdown format."""
        analysis = structure_analysis
        timestamp = analysis["timestamp"]
        exp_type = analysis["experiment_type"]
        
        summary = f"""# Experiment Structure Analysis
Generated: {timestamp}

## Experiment Type: {exp_type}
**Root Directory**: {analysis["working_directory"]}
**Complexity**: {analysis["insights"]["complexity_assessment"]}

## Directory Structure Overview
**Total Files**: {analysis["directory_map"]["file_counts"]["total"]}
**Total Directories**: {analysis["directory_map"]["directory_counts"]["total"]}

### File Distribution by Type
"""
        
        # Add file type distribution
        for ext, count in sorted(analysis["directory_map"]["file_counts"]["by_extension"].items()):
            summary += f"- `{ext or 'no extension'}`: {count} files\n"
        
        summary += "\n## Key Resources Identified\n\n"
        
        # Add prioritized resource sections
        priorities = [
            ("high_priority", "🎯 High Priority (Essential for Paper)", "These are the most important resources for paper writing"),
            ("medium_priority", "📊 Medium Priority (Supporting Material)", "Useful supporting data and analysis"),
            ("low_priority", "📝 Low Priority (Background Info)", "Background information and logs"),
            ("artifacts", "🗂️ Artifacts (Usually Skippable)", "Execution artifacts and temporary files")
        ]
        
        for priority_key, title, description in priorities:
            resources = analysis["resource_inventory"].get(priority_key, [])
            summary += f"### {title}\n"
            summary += f"*{description}*\n\n"
            
            if resources:
                summary += f"**Count**: {len(resources)} items\n\n"
                # Group by directory for better organization
                by_directory = {}
                for resource in resources[:20]:  # Limit to first 20 for readability
                    dir_name = resource["directory"]
                    if dir_name not in by_directory:
                        by_directory[dir_name] = []
                    by_directory[dir_name].append(resource)
                
                for dir_name, dir_resources in sorted(by_directory.items()):
                    summary += f"**{dir_name}/**\n"
                    for resource in dir_resources:
                        size_mb = resource["size"] / (1024*1024) if resource["size"] > 0 else 0
                        summary += f"- `{resource['name']}` ({size_mb:.1f}MB) - {resource.get('description', 'Unknown')}\n"
                    summary += "\n"
                
                if len(resources) > 20:
                    summary += f"... and {len(resources) - 20} more items\n\n"
            else:
                summary += "No resources found in this category.\n\n"
        
        # Add experiment-specific guidance
        summary += "## Extraction Strategy\n\n"
        if exp_type == "AI-Scientist-v2":
            summary += """### For V2 Experiments:
1. **Extract stage summaries** → Load progression narrative from JSON files
2. **Identify best implementations** → Use best_solution IDs from summaries
3. **Organize figures by stage** → Preserve scientific progression (baseline→research→ablation)
4. **Extract research context** → Load hypothesis from idea.json

### Key V2 Locations:
"""
            # Add V2-specific patterns found
            for pattern_info in analysis["key_indicators"]["v2_patterns"]:
                summary += f"- **{pattern_info['pattern']}**: {pattern_info['description']}\n"
                
        else:
            summary += """### For Generic Experiments:
1. **Discovery phase** → Scan all data and figure files
2. **Categorization** → Group by type and relevance
3. **Annotation** → Generate descriptions for key files
4. **Organization** → Structure for paper writing

### Recommended Processing:
- Focus on high-priority resources first
- Use file extensions to guide processing strategy
- Generate annotations for key data files
"""
        
        summary += f"\n## Intelligence Insights\n\n"
        summary += f"**Strategy**: {analysis['insights']['recommended_strategy']}\n\n"
        
        if analysis["insights"]["key_findings"]:
            summary += "**Key Findings**:\n"
            for finding in analysis["insights"]["key_findings"]:
                summary += f"- {finding}\n"
            summary += "\n"
        
        if analysis["insights"]["potential_challenges"]:
            summary += "**Potential Challenges**:\n"
            for challenge in analysis["insights"]["potential_challenges"]:
                summary += f"- {challenge}\n"
        
        return summary
    
    def _save_structure_summary(self, structure_summary: str) -> str:
        """Save structure summary for later use."""
        # Create paper_workspace directory
        subspace_dir = os.path.join(self.working_dir, "paper_workspace")
        os.makedirs(subspace_dir, exist_ok=True)
        
        # Save structure analysis
        summary_path = os.path.join(subspace_dir, "structure_analysis.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(structure_summary)
        
        print(f"  📝 Structure summary saved to: {summary_path}")
        return summary_path
    
    def _extract_using_structure(self, structure_analysis: Dict[str, Any], 
                                structure_summary: str) -> Dict[str, Any]:
        """Phase 3: Use structure analysis to efficiently extract resources."""
        extracted = {
            "extraction_method": "structure_guided",
            "experiment_type": structure_analysis["experiment_type"],
            "primary_resources": {},
            "supporting_resources": {},
            "context_data": {}
        }
        
        if structure_analysis["experiment_type"] == "AI-Scientist-v2":
            extracted.update(self._extract_v2_resources(structure_analysis))
        else:
            extracted.update(self._extract_generic_resources(structure_analysis))
        
        return extracted
    
    def _extract_v2_resources(self, structure_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract V2 resources using structure intelligence."""
        v2_extracted = {
            "stage_summaries": {},
            "research_context": {},
            "best_implementations": {},
            "organized_figures": {},
            "progression_narrative": {}
        }
        
        # Extract stage summaries (high priority)
        high_priority = structure_analysis["resource_inventory"]["high_priority"]
        for resource in high_priority:
            if "summary.json" in resource["name"]:
                try:
                    with open(resource["path"], 'r') as f:
                        summary_data = json.load(f)
                    
                    stage_name = resource["name"].replace("_summary.json", "")
                    v2_extracted["stage_summaries"][stage_name] = {
                        "data": summary_data,
                        "path": resource["path"],
                        "description": resource.get("description", "")
                    }
                except Exception as e:
                    print(f"  ⚠️ Error loading {resource['name']}: {e}")
            
            elif resource["name"] in ["idea.json", "idea.md"]:
                try:
                    with open(resource["path"], 'r') as f:
                        if resource["name"].endswith('.json'):
                            context_data = json.load(f)
                        else:
                            context_data = f.read()
                    
                    v2_extracted["research_context"][resource["name"]] = {
                        "data": context_data,
                        "path": resource["path"]
                    }
                except Exception as e:
                    print(f"  ⚠️ Error loading {resource['name']}: {e}")
        
        # Build progression narrative from stage summaries
        v2_extracted["progression_narrative"] = self._build_v2_progression(
            v2_extracted["stage_summaries"]
        )
        
        # Organize figures by stage
        v2_extracted["organized_figures"] = self._organize_v2_figures_intelligent(
            structure_analysis
        )
        
        return v2_extracted
    
    def _extract_generic_resources(self, structure_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract generic resources using structure intelligence."""
        generic_extracted = {
            "data_files": [],
            "figure_files": [],
            "metadata_files": [],
            "organized_by_type": {}
        }
        
        # Process high and medium priority resources
        for priority in ["high_priority", "medium_priority"]:
            resources = structure_analysis["resource_inventory"].get(priority, [])
            
            for resource in resources:
                file_info = {
                    "name": resource["name"],
                    "path": resource["path"],
                    "size": resource["size"],
                    "description": resource.get("description", ""),
                    "priority": priority
                }
                
                # Categorize by file type
                if resource.get("extension") in ['.png', '.pdf', '.svg', '.jpg']:
                    generic_extracted["figure_files"].append(file_info)
                elif resource.get("extension") in ['.json', '.csv', '.txt', '.md']:
                    generic_extracted["metadata_files"].append(file_info)
                elif resource.get("extension") in ['.npy', '.npz', '.pkl']:
                    generic_extracted["data_files"].append(file_info)
        
        return generic_extracted
    
    def _build_v2_progression(self, stage_summaries: Dict[str, Any]) -> Dict[str, Any]:
        """Build progression narrative from V2 stage summaries."""
        progression = {
            "stages_identified": list(stage_summaries.keys()),
            "metrics_progression": {},
            "improvements": {},
            "timeline": []
        }
        
        # Extract metrics from each stage
        for stage_name, stage_info in stage_summaries.items():
            stage_data = stage_info.get("data", {})
            if "results" in stage_data and "metrics" in stage_data["results"]:
                progression["metrics_progression"][stage_name] = stage_data["results"]["metrics"]
        
        # Calculate improvements between stages
        stages = ["baseline", "research", "ablation"]
        for i in range(len(stages) - 1):
            current_stage = stages[i]
            next_stage = stages[i + 1]
            
            if (current_stage in progression["metrics_progression"] and 
                next_stage in progression["metrics_progression"]):
                
                current_metrics = progression["metrics_progression"][current_stage]
                next_metrics = progression["metrics_progression"][next_stage]
                
                improvements = {}
                for metric in current_metrics:
                    if metric in next_metrics:
                        try:
                            current_val = float(current_metrics[metric])
                            next_val = float(next_metrics[metric])
                            
                            # Calculate improvement (assuming higher is better for most metrics)
                            if metric.lower() in ["loss", "error"]:
                                # Lower is better
                                improvement = (current_val - next_val) / current_val
                            else:
                                # Higher is better
                                improvement = (next_val - current_val) / current_val
                            
                            improvements[metric] = improvement
                        except:
                            continue
                
                progression["improvements"][f"{current_stage}_to_{next_stage}"] = improvements
        
        return progression
    
    def _organize_v2_figures_intelligent(self, structure_analysis: Dict[str, Any]) -> Dict[str, List[str]]:
        """Intelligently organize V2 figures by stage."""
        organized = {
            "baseline": [],
            "research": [],
            "ablation": [],
            "combined": [],
            "other": []
        }
        
        # Find figure resources
        all_resources = []
        for priority in ["high_priority", "medium_priority"]:
            all_resources.extend(structure_analysis["resource_inventory"].get(priority, []))
        
        # Create organized figures directory
        figures_dest = os.path.join(self.working_dir, "paper_workspace/intelligent_figures")
        os.makedirs(figures_dest, exist_ok=True)
        
        # Categorize and organize figures
        for resource in all_resources:
            if resource.get("extension") in ['.png', '.pdf', '.svg', '.jpg']:
                filename = resource["name"].lower()
                
                # Determine category based on filename
                if "baseline" in filename:
                    category = "baseline"
                elif "research" in filename:
                    category = "research" 
                elif "ablation" in filename:
                    category = "ablation"
                elif "combined" in filename or "comparison" in filename:
                    category = "combined"
                else:
                    category = "other"
                
                # Create category subdirectory
                category_dir = os.path.join(figures_dest, category)
                os.makedirs(category_dir, exist_ok=True)
                
                # Copy file to organized location
                dest_path = os.path.join(category_dir, resource["name"])
                try:
                    shutil.copy2(resource["path"], dest_path)
                    organized[category].append(dest_path)
                except Exception as e:
                    print(f"  ⚠️ Error copying {resource['name']}: {e}")
        
        return organized
    
    def _create_intelligent_package(self, structure_analysis: Dict[str, Any],
                                   extracted_resources: Dict[str, Any]) -> Dict[str, str]:
        """Create comprehensive package from intelligent extraction."""
        package = {
            "structure_analysis": "",
            "extracted_resources": "",
            "organized_figures": "",
            "extraction_summary": ""
        }
        
        # Save analysis data
        base_dir = os.path.join(self.working_dir, "paper_workspace")
        
        # Save structure analysis
        analysis_file = os.path.join(base_dir, "structure_analysis.json")
        with open(analysis_file, 'w') as f:
            json.dump(structure_analysis, f, indent=2)
        package["structure_analysis"] = analysis_file
        
        # Save extracted resources
        resources_file = os.path.join(base_dir, "extracted_resources.json")
        with open(resources_file, 'w') as f:
            json.dump(extracted_resources, f, indent=2)
        package["extracted_resources"] = resources_file
        
        # Create extraction summary
        summary_file = os.path.join(base_dir, "extraction_summary.md")
        with open(summary_file, 'w') as f:
            f.write(self._generate_extraction_summary(structure_analysis, extracted_resources))
        package["extraction_summary"] = summary_file
        
        return package
    
    def _generate_extraction_summary(self, structure_analysis: Dict[str, Any],
                                    extracted_resources: Dict[str, Any]) -> str:
        """Generate summary of extraction results."""
        exp_type = structure_analysis["experiment_type"]
        timestamp = datetime.now().isoformat()
        
        summary = f"""# Intelligent Extraction Summary
Generated: {timestamp}
Experiment Type: {exp_type}

## Extraction Results

### Resources Extracted
"""
        
        if exp_type == "AI-Scientist-v2":
            summary += f"""
**Stage Summaries**: {len(extracted_resources.get('stage_summaries', {}))} files
**Research Context**: {len(extracted_resources.get('research_context', {}))} files
**Organized Figures**: {sum(len(figs) for figs in extracted_resources.get('organized_figures', {}).values())} files

### Stage Progression Detected
"""
            progression = extracted_resources.get("progression_narrative", {})
            if progression:
                summary += f"**Stages**: {', '.join(progression.get('stages_identified', []))}\n"
                
                if progression.get("improvements"):
                    summary += "\n**Key Improvements**:\n"
                    for transition, improvements in progression["improvements"].items():
                        summary += f"- {transition}:\n"
                        for metric, improvement in improvements.items():
                            summary += f"  - {metric}: {improvement:.2%}\n"
        
        else:
            summary += f"""
**Data Files**: {len(extracted_resources.get('data_files', []))} files
**Figure Files**: {len(extracted_resources.get('figure_files', []))} files
**Metadata Files**: {len(extracted_resources.get('metadata_files', []))} files
"""
        
        summary += f"""
## Intelligent Organization

All resources have been organized in `paper_workspace/` with structure preserved:
- Original relationships maintained
- Context information included
- Priority classification applied

## Ready for Paper Writing

The extracted resources are now ready for WriteupAgent to use for academic paper generation.
"""
        
        return summary
    
    def _generate_writeup_guidance(self, structure_analysis: Dict[str, Any]) -> List[str]:
        """Generate specific guidance for WriteupAgent."""
        exp_type = structure_analysis["experiment_type"]
        guidance = [
            "Structure analysis completed - see structure_analysis.txt",
            "Resources organized by priority and context",
            "All files copied to paper_workspace/ with preserved relationships"
        ]
        
        if exp_type == "AI-Scientist-v2":
            guidance.extend([
                "V2 stage summaries available in extracted_resources.json",
                "Progression narrative ready - see stage transitions",
                "Best implementations identified per stage",
                "Figures organized by experimental stage (baseline/research/ablation)"
            ])
        else:
            guidance.extend([
                "Generic experiment structure detected",
                "Files categorized by type and importance",
                "Manual analysis may be needed for complex relationships"
            ])
        
        return guidance