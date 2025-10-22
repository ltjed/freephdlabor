<p align="center">
  <img src="assets/logo.png" alt="freephdlabor logo" width="600">
</p>

<h1 align="center">customizing your own research lab to do scientific research in your field 24/7</h1>

<p align="center">
  <a href="https://freephdlabor.github.io">
    <img src="https://img.shields.io/badge/Blog-GitHub_Pages-222?style=for-the-badge&logo=github" alt="Blog">
  </a>
  <a href="https://arxiv.org/abs/2510.15624">
    <img src="https://img.shields.io/badge/arXiv-2510.15624-b31b1b?style=for-the-badge&logo=arxiv" alt="arXiv">
  </a>
  <a href="https://github.com/ltjed/freephdlabor/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
  </a>
  <br>
  <img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fltjed%2Ffreephdlabor&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=visitors&edge_flat=false" alt="Visitors">
</p>

---

## Overview

**freephdlabor** is an open-source multiagent framework that automates the complete scientific research lifecycle—from hypothesis generation through experimentation to publication-ready manuscripts.

<p align="center">
  <img src="assets/features.png" alt="freephdlabor core features" width="800">
</p>

### Use It Two Ways

**1. Out of the Box**: Give it a research idea in the evening, wake up to a complete paper with real experiments, figures, and citations. No configuration needed—just run it.

**2. Customize for Your Domain**: Adapt it to materials science, biology, economics, or any other scientific domains. All you need is to define domain-specific tools or find out-of-the-box tools from other repos. The architecture and support features of freephdlabor can handle the rest-coordination, memory, workflow etc.-automatically.

### See It In Action

https://github.com/user-attachments/assets/d07f551f-a749-4235-ad0c-2e10367236d0

*[Watch the full demo on YouTube](https://www.youtube.com/watch?v=xwG7dpUtues)*

### What Makes It Different

| Feature | freephdlabor | Existing Systems |
|---------|--------------|---------------|
| **Dynamic Workflows** | ✅ Adapts in real-time to research findings | ❌ Predetermined workflows with little to no flexibility |
| **Fully Customizable** | ✅ Add/modify/remove agents with ease via built-in customization support | ❌ Difficult to adapt without complete redesign the entire system |
| **Human-in-the-Loop** | ✅ Naturally integrates human feedback in real-time | ❌ Intervention only at fixed workflow checkpoints |
| **Continual Research** | ✅ Context management enables sustained exploration | ❌ One-off, single-run attempts |

**Learn More**: For an accessible introduction and the intuition behind the various designs, see our [blog post](https://freephdlabor.github.io/). For specifics regarding implementation, see our [technical report](https://arxiv.org/abs/2510.15624).

---
## 🛠️ Installation

### Prerequisites

- Python 3.11+
- Conda environment manager
- CUDA-compatible GPU (recommended if you want to run experiments that require computation)
- API keys for your chosen LLM providers

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/ltjed/freephdlabor.git
cd freephdlabor
```

2. **Create and activate conda environment:**
```bash
conda env create -f environment.yml
conda activate freephdlabor
```

3. **Set up API keys:**
Modify the `.env` file with your actual API keys:
```bash
OPENAI_API_KEY=your_openai_key

ANTHROPIC_API_KEY=your_anthropic_key

GOOGLE_API_KEY=your_google_key
...
```
Then optionally edit in `.llm_config.yaml` to customize model selection for different components.

## 🚀 Quick Start

Run the system with a research task:

```bash
python launch_multiagent.py --task "Your research idea or direction here"
```

With the `--task` parameter, you're describing the research direction or idea you want the system to carry out. The **ManagerAgent** will take care of the rest by orchestrating specialized agents (IdeationAgent, WriteupAgent, etc.) to execute your research autonomously.

**Example:**
```bash
python launch_multiagent.py --task "investigate the effect of learning rate schedules on neural network generalization and report any interesting discoveries in a paper"
```
Or, if you have a research plan in mind...

```bash
python launch_multiagent.py --task "Complete a full research project on detecting and predicting training phase transitions in neural networks using Hidden Markov Models (HMMs).

RESEARCH OBJECTIVES: (1) Generate novel research ideas for training phase detection, (2) Train a single Pythia-160M model on a small subset of data while logging detailed metrics every 10 steps, (3) Extract features like gradient norms, weight magnitudes, and activation statistics to characterize training dynamics, (4) Fit HMMs to identify distinct training phases (e.g., rapid memorization, feature learning, refinement), (5) Build a predictor that can forecast performance jumps or phase transitions at least 500 steps in advance. Use small data subsets (10K samples) to ensure fast training cycles within 2 hours. Focus on clear automated metrics: phase detection accuracy, prediction lead time, and correlation with actual performance changes. Target achieving >85% accuracy in predicting major training phase transitions."
```

Check out an [example paper](https://github.com/ltjed/freephdlabor/blob/main/assets/example_paper.pdf).

## 📁 Codebase Structure

The main entry point is `launch_multiagent.py`, which initializes the multiagent system. Here's how the codebase is organized:

```
freephdlabor/
├── launch_multiagent.py          # Main entry point - starts the multiagent system
├── freephdlabor/                 # Core package
│   ├── agents/                   # Agent implementations
│   │   ├── manager_agent.py      # Central orchestrator
│   │   ├── ideation_agent.py     # Research idea generation
│   │   ├── experimentation_agent.py # Conducts experiments
│   │   ├── writeup_agent.py      # Paper writing
│   │   ├── reviewer_agent.py     # Research review
│   │   └── proofreading_agent.py # Paper proofreading
│   ├── toolkits/                 # Specialized tools for agents
│   │   ├── writeup/              # LaTeX and paper writing tools
│   │   ├── general_tools/        # File editing, web search, etc.
│   │   └── run_experiment_tool.py # Experiment execution
│   ├── prompts/                  # Agent instructions and templates
│   ├── interpreters/             # Code execution environment
│   └── config.py                 # Configuration management
├── external_tools/               # External tools (out-of-the-box Python functions)
├── results/                      # Workspaces for each run (agents share and communicate here)
├── environment.yml               # Conda environment
├── .llm_config.yaml              # LLM configuration
└── launch_multiagent_slurm.sh    # SLURM template for HPC clusters
```

### How It Works

1. **`launch_multiagent.py`** - The main entry point that:
   - Parses command line arguments
   - Loads LLM configuration
   - Creates the workspace directory
   - Initializes the ManagerAgent with specialized tools
   - Runs the research task

2. **ManagerAgent** - Orchestrates the research workflow by delegating tasks to specialized agents.

3. **Specialized Agents** - Each agent has specific capabilities:
   - **IdeationAgent**: Generates and refines research ideas
   - **ExperimentationAgent**: Draft code, execute experiment
   - **WriteupAgent**: Creates academic papers with LaTeX
   - **ReviewerAgent**: Reviews and provides feedback
   - **ProofreadingAgent**: Fixes typos and formatting issues

### Modular Design Principles

**Agent Independence**: Each agent is self-contained with its own:
- Tool set (`freephdlabor/toolkits/`)
- Instructions (`freephdlabor/prompts/`)
- Memory and state management
- Communication protocols

**Dynamic Integration**: New agents can be added by:
- Extending the `BaseResearchAgent` class
- Implementing required methods (`run()`, `get_tools()`, etc.)
- Registering with the ManagerAgent
- The system automatically handles agent coordination

**Interruption & Resume**: The system maintains:
- Complete workspace state
- Agent memory and context
- Research progress tracking
- Seamless resumption capabilities

## 🔧 Customizing for Your Research Domain

What if you work in a completely different domain of science other than AI/ML, such as materials science, biology, or economics? Fortunately, **freephdlabor** is designed with modularity at its core, allowing you to adapt the multiagent system to your specific research domain.

One easy way to get started is to modify the **ExperimentationAgent** to conduct experiments in your field based on research ideas. And doing so could be as simple as replacing the existing **RunExperimentTool** with a custom tool that takes research ideas as input and outputs materials experiment results (e.g., crystal structure predictions, property calculations, or synthesis protocols if you are material scientist). With automatic prompt optimization (detailed in Step 5), customization often just requires giving agents the right tools - the prompt engineering is handled automatically.

The following steps explain how to perform such customizations in general:

### Step 1: Modify existing or create new specialized agents

You can either modify existing agent classes or create your own specialized agents. Here's an example of creating a custom agent in `freephdlabor/agents/`:

```python
# freephdlabor/agents/my_custom_agent.py
from .base_research_agent import BaseResearchAgent
from typing import List
from smolagents import Tool

class MyCustomAgent(BaseResearchAgent):
    """Custom agent for specialized research tasks."""
    
    def __init__(self, model, workspace_dir: str, workspace_interpreter=None, **kwargs):
        # Define your custom tools
        tools = [
            MyCustomTool(),
            # Add other tools as needed
        ]
        
        # Create system prompt for your domain
        system_prompt = """
        You are MyCustomAgent, specialized in [your domain].
        
        Your capabilities include:
        - [Capability 1]
        - [Capability 2] 
        - [Capability 3]
        
        Use your tools to accomplish research tasks in [your domain].
        Always coordinate with other agents and maintain research continuity.
        """
        
        super().__init__(
            model=model,
            agent_name="my_custom_agent",
            workspace_dir=workspace_dir,
            workspace_interpreter=workspace_interpreter,
            tools=tools,
            system_prompt=system_prompt,
            **kwargs
        )
```

### Step 2: Modify existing tools or create new specialized tools

You can either modify existing tools or create new ones tailored to your domain. To create a new tool, add it to `freephdlabor/toolkits/`:

```python
# freephdlabor/toolkits/my_custom_tool.py
from smolagents import Tool
from typing import Any, Dict

class MyCustomTool(Tool):
    """Custom tool for specialized research tasks."""
    
    def __init__(self):
        super().__init__(
            name="my_custom_tool",
            description="Description of what this tool does"
        )
    
    def __call__(self, input_data: str, **kwargs) -> str:
        """Execute the tool with given input."""
        # Implement your tool logic here
        return "Tool execution result"
```

### Step 3: Modify or create your custom initialize function

You can either modify existing initialization logic in `launch_multiagent.py` or create your own initialize function that sets up your custom multiagent system:

```python
# my_custom_system.py
from freephdlabor.agents.manager_agent import ManagerAgent
from freephdlabor.agents.my_custom_agent import MyCustomAgent
from freephdlabor.agents.ideation_agent import IdeationAgent
from freephdlabor.agents.experimentation_agent import ExperimentationAgent
from freephdlabor.agents.writeup_agent import WriteupAgent
from freephdlabor.interpreters.workspace_executor import WorkspacePythonExecutor

def initialize_my_custom_system(model, workspace_dir, workspace_interpreter, essential_imports, **kwargs):
    """
    Initialize your custom multiagent system.
    
    This function gives you complete control over:
    - Which agents to include
    - How they're configured
    - How they interact with each other
    - Custom workflows and coordination
    """
    print("🔧 Initializing custom multiagent system...")
    
    # Create your custom agents
    custom_agent = MyCustomAgent(
        model=model,
        workspace_dir=workspace_dir,
        workspace_interpreter=workspace_interpreter,
        additional_authorized_imports=essential_imports
    )
    
    # Include standard agents if needed
    ideation_agent = IdeationAgent(
        model=model,
        workspace_dir=workspace_dir,
        workspace_interpreter=workspace_interpreter,
        additional_authorized_imports=essential_imports
    )
    
    experimentation_agent = ExperimentationAgent(
        model=model,
        workspace_dir=workspace_dir,
        workspace_interpreter=workspace_interpreter,
        additional_authorized_imports=essential_imports
    )
    
    writeup_agent = WriteupAgent(
        model=model,
        workspace_dir=workspace_dir,
        workspace_interpreter=workspace_interpreter,
        additional_authorized_imports=essential_imports
    )
    
    # Create your custom manager with your agent configuration
    manager = ManagerAgent(
        model=model,
        workspace_dir=workspace_dir,
        workspace_interpreter=workspace_interpreter,
        additional_authorized_imports=essential_imports,
        # Pass your custom agents
        custom_agents=[custom_agent],
        # Configure which standard agents to include
        include_ideation=True,
        include_experimentation=True,
        include_writeup=True,
        include_reviewer=False,  # Skip reviewer if not needed
        include_proofreading=False,  # Skip proofreading if not needed
        **kwargs
    )
    
    return manager
```

### Step 4: Modify or create your custom launch script

You can either modify `launch_multiagent.py` directly or create your own launch script that uses your custom initialize function:

```python
# launch_my_custom_system.py
import argparse
import os
import sys
from datetime import datetime

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from freephdlabor.utils import create_model
from my_custom_system import initialize_my_custom_system

def main():
    """Main entry point for your custom multiagent system."""
    parser = argparse.ArgumentParser(description="Custom Multiagent Research System")
    parser.add_argument("--model", type=str, default="gpt-5", help="LLM model to use")
    parser.add_argument("--workspace", type=str, default="my_research", help="Workspace directory")
    parser.add_argument("--task", type=str, help="Research task to execute")
    args = parser.parse_args()
    
    # Create model
    model = create_model(args.model)
    
    # Create workspace directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    workspace_dir = os.path.join("results", f"{timestamp}_{args.workspace}")
    os.makedirs(workspace_dir, exist_ok=True)
    
    # Essential imports for agents
    essential_imports = [
        "json", "os", "sys", "datetime", "typing", "pathlib", "shutil", 
        "functools", "copy", "pickle", "logging", "warnings", "gc",
        "argparse", "configparser", "yaml", "toml", "requests", "urllib",
        "datasets", "transformers", "huggingface_hub", "tokenizers",
        "wandb", "tensorboard", "tqdm", "requests", "urllib", "zipfile", "tarfile"
    ]
    
    # Create workspace executor
    workspace_executor = WorkspacePythonExecutor(
        workspace_dir=workspace_dir,
        additional_authorized_imports=essential_imports
    )
    
    # Initialize your custom system
    manager = initialize_my_custom_system(
        model=model,
        workspace_dir=workspace_dir,
        workspace_interpreter=workspace_executor,
        essential_imports=essential_imports
    )
    
    # Define your task
    task = args.task or "Execute your custom research workflow"
    
    print(f"🚀 Starting custom multiagent system...")
    print(f"📁 Workspace: {workspace_dir}")
    print(f"📝 Task: {task}")
    
    # Run your custom system
    result = manager.run(task)
    
    print(f"✅ Custom system completed!")
    print(f"📋 Result: {result}")

if __name__ == "__main__":
    main()
```

### Step 5: Run and iteratively improve your custom multiagent system

Execute your custom multiagent system:

```bash
# Run with default settings
python launch_my_custom_system.py --task "YOUR TASK HERE"
```

**Automatically Optimizing System Prompts (Optional):**

The system automatically logs all LM calls to `results/{workspace}/agent_llm_calls.jsonl`. You can use built-in slash commands with Claude Code to analyze these logs and automatically suggest improvements to agent prompts. Thus, often customization your own multiagent system often becomes as simple as giving agents the right tools - the prompts engineering is handled automatically.

```bash
# Analyze when a specific agent didn't have enough context
/analyze_agent_context agent_name=writeup results_dir=20250501_143022_my_research

# Automatically refine agent prompts based on the analysis
/refine_agent_prompt agent_name=writeup results_dir=20250501_143022_my_research
```

This iterative analysis helps identify communication gaps between agents and improves system prompts for better collaboration. See `.claude/commands/` for command details.

### Advanced Customization Examples

#### Example 1: Domain-Specific Research System

```python
def initialize_computer_vision_system(model, workspace_dir, workspace_interpreter, essential_imports, **kwargs):
    """Initialize a computer vision research system."""
    
    # Create CV-specific agents
    cv_analysis_agent = ComputerVisionAnalysisAgent(
        model=model, workspace_dir=workspace_dir, workspace_interpreter=workspace_interpreter
    )
    
    cv_experiment_agent = ComputerVisionExperimentAgent(
        model=model, workspace_dir=workspace_dir, workspace_interpreter=workspace_interpreter
    )
    
    cv_visualization_agent = ComputerVisionVisualizationAgent(
        model=model, workspace_dir=workspace_dir, workspace_interpreter=workspace_interpreter
    )
    
    # Create custom manager for CV research
    manager = ComputerVisionManagerAgent(
        model=model,
        workspace_dir=workspace_dir,
        workspace_interpreter=workspace_interpreter,
        cv_agents=[cv_analysis_agent, cv_experiment_agent, cv_visualization_agent]
    )
    
    return manager
```

#### Example 2: Minimal Research System

```python
def initialize_minimal_system(model, workspace_dir, workspace_interpreter, essential_imports, **kwargs):
    """Initialize a minimal research system with only essential agents."""
    
    # Only include ideation and writeup agents
    ideation_agent = IdeationAgent(
        model=model, workspace_dir=workspace_dir, workspace_interpreter=workspace_interpreter
    )
    
    writeup_agent = WriteupAgent(
        model=model, workspace_dir=workspace_dir, workspace_interpreter=workspace_interpreter
    )
    
    # Create minimal manager
    manager = MinimalManagerAgent(
        model=model,
        workspace_dir=workspace_dir,
        workspace_interpreter=workspace_interpreter,
        agents=[ideation_agent, writeup_agent]
    )
    
    return manager
```

### Best Practices for Custom Systems

1. **Design for Your Domain**: Create agents and workflows specific to your research area
2. **Modular Architecture**: Keep agents independent and reusable
3. **Clear Interfaces**: Define clear communication protocols between agents
4. **Error Handling**: Implement robust error handling and recovery
5. **Logging and Monitoring**: Add comprehensive logging for debugging
6. **Testing**: Create test cases for your custom agents and workflows


## 🚀 Advanced Features

Once you're comfortable with basic usage and customization, **freephdlabor** offers powerful advanced capabilities for dynamic research workflows.

### Interrupt and Interact with Running Agents

This feature allows you to interrupt and provide instructions to a running agent through a terminal connection, enabling real-time guidance without restarting the entire research process.

#### Starting the Agent with Callback Server

**On your local machine:**
```bash
python launch_multiagent.py --callback_host="127.0.0.1" --callback_port=5001
```

Using `127.0.0.1 (localhost)` when running locally.

**On remote server (e.g., SLURM cluster):**
```bash
python launch_multiagent.py --callback_host="0.0.0.0" --callback_port=5001
```

Using `0.0.0.0` to allow connections from other machines/nodes (or specify your server's IP address).

#### Connecting the Interruption Handler

**On your local machine:**
```bash
nc localhost 5001
```

**On remote server:**
```bash
nc <compute_node_name> 5001
```

#### Interacting with Agents

**Interrupt the workflow:**
When the terminal is connected to the handler, type `interrupt` and wait for the step_callback function to be called.

**Give your instruction:**
Type instructions line by line. When you finish the last line, press ENTER twice. You will see an option asking whether the instruction is a modification of the current task or a new task. Type `m` for modification or `n` for a new task, then press ENTER. The instructions will be added to the agent's memory and the workflow will continue.

### Resume from Previous Executions

The system maintains complete workspace state, allowing you to resume research sessions:

```bash
# Start a research task
python launch_multiagent.py --task "Your research task here"

# Later, interrupt and resume with modifications
python launch_multiagent.py --resume results/WORKSPACE_NAME --task "Continue with additional analysis"
```


### Command Line Options Reference

```bash
python launch_multiagent.py [OPTIONS]

Options:
  --model <string>              LLM model to use
  --interpreter <string>        Python interpreter path
  --debug                       Enable debug logging
  --reasoning-effort <string>   GPT-5 reasoning effort: minimal|low|medium|high
  --verbosity <string>          GPT-5 verbosity: low|medium|high
  --callback-host <string>      Callback server host
  --callback-port <integer>     Callback server port
  --task <string>               Research task description
  --resume <string>             Resume from existing workspace path
  --enable-planning             Enable planning mode
  --planning-interval <integer> Planning interval in steps
```

## 🏃‍♂️ Running on HPC Clusters

For SLURM-based HPC clusters, use the annotated template `launch_multiagent_slurm.sh`. This template includes detailed comments explaining each configuration section. Note that HPCs likely differ in parameters names etc., so the template may not work on your HPC out-of-the-box.


## 🔧 Configuration

### LLM Configuration (`.llm_config.yaml`)

The configuration file provides fine-grained control over which models are used for different components:

```yaml
main_agents:
  model: gemini-2.5-pro
  thinking_budget: 32768

run_experiment_tool:
  code_model: gpt-5
  feedback_model: gpt-5
  vlm_model: gpt-5
  report_model: gpt-5
  reasoning_effort: high
```

- **`main_agents`**: Configures the LLM used by agents in the multiagent system (ManagerAgent, IdeationAgent, WriteupAgent, etc.)
- **`run_experiment_tool`**: Configures models used internally within the RunExperimentTool

This demonstrates the fine-grained control available - you can use different models for different components. For example, you might use a fast, cost-effective model for main agents while using a more powerful reasoning model for experiment execution.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- The multiagent system is built on the [smolagents](https://github.com/huggingface/smolagents) framework
- Incorporates a modified part of [AI-Scientist-v2](https://github.com/SakanaAI/AI-Scientist-v2) as a tool for ExperimentationAgent
- Uses [Phoenix](https://github.com/Arize-ai/phoenix) for telemetry and monitoring

## 📞 Support

For questions and support, please open an issue on the GitHub repository.

## Citation

To cite this work, please cite https://arxiv.org/abs/2510.15624 with the following bibtex entry:

```bibtex
@misc{li2025freephdlabor,
      title={Build Your Personalized Research Group: A Multiagent Framework for Continual and Interactive Science Automation},
      author={Ed Li and Junyu Ren and Xintian Pan and Cat Yan and Chuanhao Li and Dirk Bergemann and Zhuoran Yang},
      year={2025},
      eprint={2510.15624},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2510.15624},
}
```