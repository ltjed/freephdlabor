from smolagents import Tool
import json
import subprocess
import tempfile
import os
import shutil
import uuid
from datetime import datetime

class RunExperimentTool(Tool):
    """
    A tool that takes a research idea in JSON format and executes a full, 
    autonomous research workflow using the AI-Scientist-v2 framework. 
    It handles code implementation, experimentation, and result analysis, 
    returning a summary of the findings.
    """
    name: str = "RunExperimentTool"
    description: str = (
        "A tool that takes a research idea in JSON format and executes a full, "
        "autonomous research workflow using the AI-Scientist-v2 framework. "
        "It handles code implementation, experimentation, and result analysis, "
        "returning a summary of the findings. "
        "CRITICAL PREREQUISITE: You MUST use IdeaStandardizationTool BEFORE calling this tool to convert "
        "the research idea to AI-Scientist-v2 format. This ensures correct model selection "
        "(e.g., Pythia-410M not DistilBERT) and dataset usage (e.g., C4/MMLU not synthetic data). "
        "The idea_json must have: Name, Title, Short Hypothesis, Abstract, Experiments fields."
    )
    
    inputs = {
        "idea_json": {
            "type": "string",
            "description": "A JSON string representing the research idea, matching the output format of GenerateIdeaTool."
        },
        "code_model": {
            "type": "string",
            "description": "LLM model to use for code generation (default: from RUN_EXPERIMENT_CODE_MODEL env or 'gpt-5')",
            "nullable": True
        },
        "feedback_model": {
            "type": "string", 
            "description": "LLM model to use for feedback and evaluation (default: from RUN_EXPERIMENT_FEEDBACK_MODEL env or 'gpt-5')",
            "nullable": True
        },
        "vlm_model": {
            "type": "string",
            "description": "VLM model to use for visual feedback on plots (default: from RUN_EXPERIMENT_VLM_MODEL env or 'gpt-5')",
            "nullable": True
        },
        "report_model": {
            "type": "string",
            "description": "LLM model to use for report generation (default: from RUN_EXPERIMENT_REPORT_MODEL env or 'gpt-5')",
            "nullable": True
        },
        "end_stage": {
            "type": "integer",
            "description": "Final stage to run (1-4). Stage 1: initial_implementation (basic working baseline), Stage 2: baseline_tuning (hyperparameter optimization), Stage 3: creative_research (novel improvements), Stage 4: ablation_studies (systematic component analysis). Default: 4 (run all stages)",
            "nullable": True
        }
    }
    output_type = "string"

    def __init__(self, workspace_dir=None):
        """
        Initializes the RunExperimentTool.
        The path to the AI-Scientist-v2 directory is hardcoded relative to this file's location.
        
        Args:
            workspace_dir: Optional workspace directory to save experiment results
        """
        super().__init__()
        # The tool is in freephdlabor/toolkits, so we go up two levels to the repo root
        # and then into external_tools/run_experiment_tool.
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        self.ai_scientist_path = os.path.join(repo_root, "external_tools", "run_experiment_tool")
        # Convert workspace_dir to absolute path immediately to avoid issues when working directory changes
        self.workspace_dir = os.path.abspath(workspace_dir) if workspace_dir else None
    
    def _get_python_executable(self):
        """
        Auto-detect the correct Python executable based on the current environment.
        Tries multiple strategies to find the right Python path.
        """
        import sys
        
        # Strategy 1: Use current Python interpreter (most reliable)
        current_python = sys.executable
        if current_python and os.path.exists(current_python):
            return current_python
        
        # Strategy 2: Look for conda environment Python
        conda_env = os.environ.get('CONDA_DEFAULT_ENV')
        if conda_env:
            # Try common conda paths
            possible_paths = [
                os.path.expanduser(f"~/.conda/envs/{conda_env}/bin/python"),
                os.path.expanduser(f"~/anaconda3/envs/{conda_env}/bin/python"),
                os.path.expanduser(f"~/miniconda3/envs/{conda_env}/bin/python"),
                f"/opt/conda/envs/{conda_env}/bin/python"
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    return path
        
        # Strategy 3: Check for specific environment names
        env_names = ['ai_scientist', 'ai_scientist_v2', 'as_env']
        for env_name in env_names:
            possible_paths = [
                os.path.expanduser(f"~/.conda/envs/{env_name}/bin/python"),
                os.path.expanduser(f"~/anaconda3/envs/{env_name}/bin/python"),
                os.path.expanduser(f"~/miniconda3/envs/{env_name}/bin/python"),
                f"/opt/conda/envs/{env_name}/bin/python"
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    return path
        
        # Strategy 4: Fall back to system python
        return "python"

    def forward(self, idea_json: str, code_model: str = None, 
                feedback_model: str = None, vlm_model: str = None,
                report_model: str = None, end_stage: int = None) -> str:
        """
        Executes the AI-Scientist-v2 workflow for a given research idea.
        
        Model parameters are read from environment variables if not provided.
        This allows configuration via .llm_config.yaml file.
        """
        # Read models from environment if not provided
        code_model = code_model or os.environ.get('RUN_EXPERIMENT_CODE_MODEL', 'gpt-5')
        feedback_model = feedback_model or os.environ.get('RUN_EXPERIMENT_FEEDBACK_MODEL', 'gpt-5')
        vlm_model = vlm_model or os.environ.get('RUN_EXPERIMENT_VLM_MODEL', 'gpt-5')
        report_model = report_model or os.environ.get('RUN_EXPERIMENT_REPORT_MODEL', 'gpt-5')
        
        # Validate and set end_stage parameter
        end_stage = end_stage or 4
        if end_stage not in [1, 2, 3, 4]:
            return json.dumps({
                "status": "failure",
                "summary": f"Invalid end_stage: {end_stage}. Must be 1, 2, 3, or 4.",
                "valid_stages": {
                    1: "initial_implementation", 
                    2: "baseline_tuning",
                    3: "creative_research", 
                    4: "ablation_studies"
                }
            })

        print(f"🎯 RunExperimentTool: Configured to run stages 1-{end_stage}")
        
        # Create a unique run directory in workspace for accessibility
        # Store results directly in workspace so WriteupAgent can access them
        if self.workspace_dir:
            # Ensure workspace_dir is absolute to avoid path resolution issues in subprocess
            workspace_abs = os.path.abspath(self.workspace_dir)
            run_base_dir = os.path.join(workspace_abs, "experiment_runs")
        else:
            run_base_dir = "/tmp/experiment_runs"
        os.makedirs(run_base_dir, exist_ok=True)
        run_dir = os.path.join(run_base_dir, str(uuid.uuid4()))
        os.makedirs(run_dir)

        try:
            # 2. Prepare Inputs
            idea_path = os.path.join(run_dir, "idea.json")
            with open(idea_path, "w") as f:
                # Handle both JSON strings and dict objects
                if isinstance(idea_json, str):
                    try:
                        parsed_json = json.loads(idea_json)
                    except json.JSONDecodeError:
                        # If it's not valid JSON, try to evaluate it as a Python literal
                        try:
                            import ast
                            parsed_json = ast.literal_eval(idea_json)
                        except (ValueError, SyntaxError):
                            return json.dumps({"status": "failure", "summary": f"Invalid idea_json format. Expected JSON string or dict, got: {type(idea_json)}"})
                else:
                    # Assume it's already a dict/object
                    parsed_json = idea_json
                
                # IdeaStandardizationTool returns JSON array [idea_object] (AI-Scientist-v2 format)
                # Extract the single idea object for validation and processing
                if isinstance(parsed_json, list) and len(parsed_json) > 0:
                    idea_data = parsed_json[0]  # Take first idea from array
                elif isinstance(parsed_json, dict):
                    idea_data = parsed_json  # Already a single object
                else:
                    return json.dumps({"status": "failure", "summary": f"Invalid idea format. Expected JSON array or object, got: {type(parsed_json)}"})
                
                # Validate that the idea is in the correct format
                # The ExperimentationAgent MUST use IdeaStandardizationTool before calling this tool
                required_fields = ["Name", "Title", "Short Hypothesis", "Abstract", "Experiments"]
                missing_fields = [field for field in required_fields if field not in idea_data]
                
                if missing_fields:
                    error_msg = (
                        f"Research idea is missing required AI-Scientist-v2 fields: {missing_fields}. "
                        f"The ExperimentationAgent MUST use IdeaStandardizationTool BEFORE calling RunExperimentTool. "
                        f"This ensures experiments use the correct models and datasets specified in the research idea, "
                        f"not generic defaults like DistilBERT and synthetic data."
                    )
                    import logging
                    logging.getLogger(__name__).error(f"ERROR: {error_msg}")
                    
                    # Return clear error instead of silently using wrong conversion
                    return json.dumps({
                        "status": "failure",
                        "summary": "Invalid research idea format - IdeaStandardizationTool was not used",
                        "error": error_msg,
                        "missing_fields": missing_fields,
                        "recommendation": "Use IdeaStandardizationTool to convert the idea before calling RunExperimentTool"
                    })
                
                json.dump([idea_data], f)
            
            # Extract idea name for logging directory
            idea_name = idea_data.get("Name", idea_data.get("Title", "unknown_idea"))
            idea_name = idea_name.replace(" ", "_").replace("/", "_")[:50]  # Sanitize

            # Check if we're in POC mode and use simplified config
            is_poc = os.environ.get('POC_MODE', 'false').lower() == 'true'
            if is_poc:
                config_template_path = os.path.join(self.ai_scientist_path, "bfts_config_poc.yaml")
                if not os.path.exists(config_template_path):
                    # Fall back to regular config if POC config doesn't exist
                    config_template_path = os.path.join(self.ai_scientist_path, "bfts_config.yaml")
            else:
                config_template_path = os.path.join(self.ai_scientist_path, "bfts_config.yaml")
            
            if not os.path.exists(config_template_path):
                return json.dumps({"status": "failure", "summary": f"Config template not found at {config_template_path}"})
            
            # Copy config and modify to use GPT-5 for all models
            config_path = os.path.join(run_dir, "bfts_config.yaml")
            shutil.copy(config_template_path, config_path)
            
            # Update config to use specified models (default to GPT-5)
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Set models for all configurations
            config['agent']['code']['model'] = code_model
            config['agent']['feedback']['model'] = feedback_model
            config['agent']['vlm_feedback']['model'] = vlm_model
            config['report']['model'] = report_model
            
            # Write updated config
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            # Log model configuration
            model_info = f"code={code_model}, feedback={feedback_model}, vlm={vlm_model}, report={report_model}"
            print(f"🤖 Configured AI-Scientist-v2 models: {model_info}")
            
            # Special note for GPT-5 models
            if any(model.startswith('gpt-5') for model in [code_model, feedback_model, vlm_model, report_model]):
                print(f"✨ GPT-5 models will use reasoning_effort='high' automatically")

            # 3. Execute AI-Scientist-v2
            # Auto-detect the correct python executable based on environment
            python_executable = self._get_python_executable()
            script_path = os.path.join(self.ai_scientist_path, "launch_scientist_bfts.py")
            cmd = [
                python_executable,
                script_path,
                "--load_ideas",
                idea_path,
                "--idea_idx",
                "0",
                "--debug",  # Enable debug mode to preserve all temporary files
                "--skip_writeup",  # Skip writeup stage to avoid LaTeX failures
                "--skip_review",   # Skip review stage since no writeup is generated
                "--end_stage", str(end_stage),  # ADD THIS LINE
            ]
            
            # Pass environment variables to subprocess
            env = os.environ.copy()
            
            # Create log directory in workspace if available
            log_dir = None
            if self.workspace_dir:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_dir = os.path.join(
                    self.workspace_dir, 
                    "experimentation_agent",
                    f"{timestamp}_{idea_name}"
                )
                os.makedirs(log_dir, exist_ok=True)
                
                # Save the command that was run
                with open(os.path.join(log_dir, "command.txt"), "w") as f:
                    f.write(" ".join(cmd))
                
                # Save run directory info for reference
                with open(os.path.join(log_dir, "run_directory.txt"), "w") as f:
                    f.write(f"AI-Scientist-v2 working directory: {run_dir}\n")
                    f.write(f"Results will be in: {run_dir}/experiments/\n")
                
                print(f"📝 Log directory created: {log_dir}")
                print(f"🔬 AI-Scientist-v2 working directory: {run_dir}")
            
            # Run without capturing output - stream to terminal
            print(f"\n🚀 Starting AI-Scientist-v2 experiment...")
            print(f"   You will see real-time progress below:\n")
            
            process = subprocess.run(
                cmd,
                cwd=run_dir,
                text=True,
                check=False,
                env=env
            )

            if process.returncode != 0:
                return json.dumps({
                    "status": "failure",
                    "summary": f"AI-Scientist-v2 script failed to execute. Python executable: {python_executable}",
                    "results_directory": run_dir,
                    "log_directory": log_dir,
                    "command": " ".join(cmd),
                    "note": "Check terminal output above for error details"
                })

            # 4. Parse and Return Results
            experiments_path = os.path.join(run_dir, "experiments")
            if not os.path.isdir(experiments_path) or not os.listdir(experiments_path):
                return json.dumps({
                    "status": "failure",
                    "summary": "AI-Scientist-v2 did not produce an experiment directory.",
                    "results_directory": run_dir,
                    "log_directory": log_dir,
                    "note": "Check terminal output above for details"
                })

            exp_dir_name = os.listdir(experiments_path)[0]
            exp_dir = os.path.join(experiments_path, exp_dir_name)
            
            # Find the latest run directory (highest index)
            logs_dir = os.path.join(exp_dir, "logs")
            if not os.path.isdir(logs_dir):
                return json.dumps({
                    "status": "failure",
                    "summary": f"Logs directory not found: {logs_dir}",
                    "results_directory": exp_dir,
                    "log_directory": log_dir,
                    "note": "AI-Scientist-v2 may not have completed properly"
                })
            
            # Find all run directories and select the latest one
            run_dirs = [d for d in os.listdir(logs_dir) if d.endswith('-run') and os.path.isdir(os.path.join(logs_dir, d))]
            if not run_dirs:
                return json.dumps({
                    "status": "failure", 
                    "summary": f"No run directories found in {logs_dir}",
                    "results_directory": exp_dir,
                    "log_directory": log_dir,
                    "note": "Expected to find directories like '0-run', '1-run', etc."
                })
            
            # Get the latest run (highest index)
            latest_run = max(run_dirs, key=lambda x: int(x.split('-')[0]))
            summary_path = os.path.join(logs_dir, latest_run, "research_summary.json")
            if not os.path.exists(summary_path):
                return json.dumps({
                    "status": "failure",
                    "summary": f"research_summary.json not found in {os.path.join(logs_dir, latest_run)}",
                    "results_directory": exp_dir,
                    "log_directory": log_dir,
                    "latest_run": latest_run,
                    "note": "Check terminal output above for details"
                })

            with open(summary_path, "r") as f:
                research_summary = json.load(f)

            # Find best code and save it
            best_code = None
            if "aggregated results of nodes with different seeds" in research_summary and \
               "code" in research_summary["aggregated results of nodes with different seeds"]:
                best_code = research_summary["aggregated results of nodes with different seeds"]["code"]
            elif "best node" in research_summary and "code" in research_summary["best node"]:
                best_code = research_summary["best node"]["code"]

            best_code_path = None
            if best_code:
                best_code_path = os.path.join(exp_dir, "logs", "best_code.py")
                with open(best_code_path, "w") as f:
                    f.write(best_code)

            # Copy comprehensive experiment results to workspace if available
            if self.workspace_dir and log_dir:
                # If run_dir is already in workspace, results are accessible - just create symlinks
                if run_dir.startswith(self.workspace_dir):
                    print(f"📁 Experiment results already in workspace: {exp_dir}")
                    # Create a convenient symlink in the log directory
                    symlink_path = os.path.join(log_dir, "experiment_results")
                    try:
                        if not os.path.exists(symlink_path):
                            os.symlink(exp_dir, symlink_path)
                        print(f"🔗 Created symlink to experiment results: {symlink_path}")
                    except OSError as e:
                        print(f"⚠️  Could not create symlink: {e}")
                else:
                    # Fallback: Copy results if they're outside workspace
                    logs_dir = os.path.join(exp_dir, "logs", "0-run")
                    if os.path.exists(logs_dir):
                        summary_files = ["draft_summary.json", "baseline_summary.json", 
                                       "research_summary.json", "ablation_summary.json"]
                        for summary_file in summary_files:
                            src_path = os.path.join(logs_dir, summary_file)
                            if os.path.exists(src_path):
                                dst_path = os.path.join(log_dir, summary_file)
                                shutil.copy2(src_path, dst_path)
                                print(f"📋 Copied {summary_file} to workspace")
                    
                    # Copy the entire experiment directory for complete results
                    workspace_exp_dir = os.path.join(log_dir, "full_experiment_results")
                    try:
                        shutil.copytree(exp_dir, workspace_exp_dir, dirs_exist_ok=True)
                        print(f"📁 Copied full experiment results to workspace")
                    except Exception as e:
                        print(f"⚠️  Warning: Could not copy full experiment results: {e}")
                
                # Save the experiment path for reference
                with open(os.path.join(log_dir, "experiment_path.txt"), "w") as f:
                    f.write(f"Experiment directory: {exp_dir}\n")
                    f.write(f"Run directory: {run_dir}\n")
                    f.write(f"Results are accessible within workspace\n")

            return json.dumps({
                "status": "success",
                "results_directory": exp_dir,
                "log_directory": log_dir,
                "summary": research_summary,
                "best_code_path": best_code_path
            })

        except Exception as e:
            python_executable = self._get_python_executable()
            return json.dumps({
                "status": "failure",
                "summary": f"An error occurred in RunExperimentTool: {str(e)}",
                "results_directory": run_dir,
                "python_executable_used": python_executable,
                "error_type": "execution_failure",
                "next_steps": [
                    "The experimental execution failed due to environment or configuration issues",
                    "Consider creating a detailed implementation plan instead of running experiments",
                    "Focus on theoretical analysis and methodology design",
                    "Clearly report this as a limitation in any research output"
                ],
                "important_note": "DO NOT generate synthetic experimental results. Report the failure honestly and focus on methodology and theoretical contributions."
            }) 