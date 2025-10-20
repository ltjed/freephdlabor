import argparse
import os
import sys
import json
import functools
from datetime import datetime
import yaml

from phoenix.otel import register
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from freephdlabor.interaction.callback_tools import setup_user_input_socket, make_user_input_step_callback

from freephdlabor.args import parse_arguments

# Debug: Check Phoenix collector endpoint before registration
import os
phoenix_endpoint = os.environ.get('PHOENIX_COLLECTOR_ENDPOINT', 'NOT SET')
print(f"🔍 DEBUG: PHOENIX_COLLECTOR_ENDPOINT = {phoenix_endpoint}")
print(f"🔍 DEBUG: About to register Phoenix tracer...")

register(set_global_tracer_provider=False, verbose=True)
print(f"✅ Phoenix tracer registered successfully")

SmolagentsInstrumentor().instrument()
print(f"✅ Smolagents instrumentation complete")


from smolagents import LiteLLMModel
from dotenv import load_dotenv
import litellm
from freephdlabor.config import load_llm_config, filter_model_params
from freephdlabor.interpreters import WorkspacePythonExecutor

# Load environment variables from .env file
load_dotenv(override=True)

# Configure LiteLLM for GPT-5 compatibility and debugging
litellm.drop_params = True
# Enable LiteLLM debugging (using new method to avoid deprecation warning)
os.environ['LITELLM_LOG'] = 'DEBUG'

# Apply the parameter filtering to LiteLLM completion function
litellm.completion = filter_model_params(litellm.completion)

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from freephdlabor.utils import create_model, initialize_agent_system

def main():
    """Main entry point for the smolagents launcher."""
    # Create single timestamp for logs and workspace
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Auto-save logs to separate files
    os.makedirs('logs', exist_ok=True)
    sys.stdout = open(f'logs/freephdlabor_{timestamp}.out', 'w', buffering=1)
    sys.stderr = open(f'logs/freephdlabor_{timestamp}.err', 'w', buffering=1)

    # Parse command line arguments
    args = parse_arguments()
    
    # Load LLM config file if it exists
    llm_config = load_llm_config()

    # Setup Interrupt Channel
    input_queue = setup_user_input_socket(args.callback_host, args.callback_port)

    print(f"✅ Interruption Port available at https://{args.callback_host}:{args.callback_port}")

    # Get interrupt callback
    interrupt_callback = make_user_input_step_callback(input_queue=input_queue)
    
    # Determine which model and settings to use
    # Precedence: CLI arguments > Config file > Defaults

    # Start with hard-coded defaults
    model_name = 'gpt-5'
    reasoning_effort = 'high'
    verbosity = 'medium'
    budget_tokens = None

    # Override with config file if present
    if llm_config and 'main_agents' in llm_config:
        model_name = llm_config['main_agents'].get('model', model_name)
        reasoning_effort = llm_config['main_agents'].get('reasoning_effort', reasoning_effort)
        verbosity = llm_config['main_agents'].get('verbosity', verbosity)
        budget_tokens = llm_config['main_agents'].get('budget_tokens', budget_tokens)
        print(f"📋 Loaded config file settings for main agents")

    # Override with CLI arguments if explicitly provided (different from argparse defaults)
    if args.model != 'gpt-5':
        model_name = args.model
        print(f"📋 CLI override: using model {model_name}")
    if args.reasoning_effort != 'high':
        reasoning_effort = args.reasoning_effort
        print(f"📋 CLI override: reasoning_effort={reasoning_effort}")
    if args.verbosity != 'medium':
        verbosity = args.verbosity
        print(f"📋 CLI override: verbosity={verbosity}")

    print("\nSmolagents Research System Initialized!")
    print("🚀 Starting autonomous research task...")
    print(f"✅ Configuration: {model_name} model")

    # Create a model
    model = create_model(model_name, reasoning_effort, verbosity, budget_tokens)
    print(f"✅ Created model: {model.model_id}")
    if model_name.startswith("gpt-5"):
        print(f"✅ GPT-5 configuration: reasoning_effort={reasoning_effort}, verbosity={verbosity}")
    elif "claude" in model_name and budget_tokens:
        print(f"✅ Claude Extended Thinking configuration: budget_tokens={budget_tokens}")
    
    # Pass experiment config to environment for RunExperimentTool
    if llm_config and 'run_experiment_tool' in llm_config:
        exp_config = llm_config['run_experiment_tool']
        os.environ['RUN_EXPERIMENT_CODE_MODEL'] = exp_config.get('code_model', 'gpt-5')
        os.environ['RUN_EXPERIMENT_FEEDBACK_MODEL'] = exp_config.get('feedback_model', 'gpt-5')
        os.environ['RUN_EXPERIMENT_VLM_MODEL'] = exp_config.get('vlm_model', 'gpt-5')
        os.environ['RUN_EXPERIMENT_REPORT_MODEL'] = exp_config.get('report_model', 'gpt-5')
        os.environ['RUN_EXPERIMENT_REASONING_EFFORT'] = exp_config.get('reasoning_effort', 'high')
        print(f"🔬 RunExperimentTool config: {exp_config.get('code_model', 'gpt-5')} with reasoning_effort={exp_config.get('reasoning_effort', 'high')}")

    # Handle resume mode or create new workspace
    if args.resume:
        # Resume from existing workspace
        results_base_dir = os.path.abspath(args.resume)

        # Basic validation: check if directory exists
        if not os.path.exists(results_base_dir):
            print(f"❌ Workspace directory does not exist: {results_base_dir}")
            return 1

        if not os.path.isdir(results_base_dir):
            print(f"❌ Path is not a directory: {results_base_dir}")
            return 1

        print(f"🔄 Resuming from existing workspace: {results_base_dir}")

        # Determine task to use
        if args.task:
            task = args.task
            print(f"📋 Using task from --task argument")
        else:
            task = "Continue working on the a previous research task. Please meticulously analyze the existing files to and then plan how to call the relevant agents to further progress the research task and deliverable better research outputs."
            print(f"📋 No specific task provided, using continuation task")
    else:
        # Determine task
        if args.task:
            task = args.task
            print(f"📋 Using task from --task argument")
        else:
            task = "Investigate whether training small language models with multiple paraphrased responses reduces hallucinations and improves response quality. Generate research ideas exploring: (1) Fine-tuning GPT-2 Small (124M) on instruction-following datasets with single vs multiple response variants, using small subsets (5K samples) for rapid experimentation, (2) Comparing response diversity and factual accuracy between single-response and multi-response training regimes, (3) Testing whether exposure to diverse correct answers during training acts as implicit regularization against repetitive or factually incorrect outputs, (4) Measuring trade-offs between response quality, diversity, and training efficiency with automated metrics. Use small-scale datasets like Alpaca-5K with automated paraphrase generation. Focus on fast automated evaluation including response diversity via Self-BLEU, factual consistency via rule-based checks, and response quality via ROUGE scores. Target achieving measurable improvements in response diversity while maintaining quality, with each experimental run completing in under 1 hour."
            print(f"📋 Using default task (no --task argument provided)")

        # Create new workspace directory
        results_base_dir = os.path.join("results", f"freephdlabor_{timestamp}")

        os.makedirs(results_base_dir, exist_ok=True)
        print(f"📁 Created new workspace: {results_base_dir}")

    # Set environment variables for agents to use
    os.environ["RESULTS_BASE_DIR"] = results_base_dir

    print(f"📁 Active workspace: {results_base_dir}")
    print(f"📝 Task: {task[:100]}{'...' if len(task) > 100 else ''}")

    # Create the ManagerAgent
    try:
        # Essential imports for tool-centric agents (no direct ML library access)
        essential_imports = [
            # Standard library essentials
            "json", "os", "posixpath", "ntpath", "sys", "datetime", "uuid", "typing", "pathlib", "shutil", 
            "functools", "copy", "pickle", "logging", "warnings", "gc",
            # Development & configuration
            "argparse", "configparser", "yaml", "toml", "requests", "urllib",
            # Data & datasets
            "datasets", "transformers", "huggingface_hub", "tokenizers",
            # Research utilities
            "wandb", "tensorboard", "tqdm", "requests", "urllib", "zipfile", "tarfile",
            # Development & testing
            "argparse", "configparser", "yaml", "toml",
        ]
        
        # Create workspace-aware Python executor for agent code
        # This ensures agent-generated code runs in the workspace directory
        workspace_executor = WorkspacePythonExecutor(
            workspace_dir=results_base_dir,
            additional_authorized_imports=essential_imports
        )
        print(f"✅ Configured workspace executor for: {results_base_dir}")
        
        # Initialize the complete multi-agent system
        manager = initialize_agent_system(
            model=model,
            workspace_dir=results_base_dir,
            workspace_interpreter=workspace_executor,
            essential_imports=essential_imports,
            enable_planning=args.enable_planning,
            planning_interval=args.planning_interval,
            interrupt_callback=interrupt_callback
        )
        
        print("\n" + "=" * 50)
        print("🔬 Running ManagerAgent with the research task...")
        print(f"📝 Task: {task}")
        print("=" * 50)

        result = manager.run(task)

        print("\n" + "=" * 50)
        print("✅ Task finished!")
        print(f"📋 Final Result from ManagerAgent:\n{result}")
        print("=" * 50)
        print("Saving Agents Memory...")
        from freephdlabor.utils import save_agent_memory
        save_agent_memory(manager)
        print("✅ Memory Saved!")

    except Exception as e:
        print(f"❌ Error during ManagerAgent execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    main()