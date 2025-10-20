"""
Args for system initialization
"""
import argparse
from .utils import AVAILABLE_MODELS

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Smolagents Research System - Multi-Agent AI Research Automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create new workspace (uses default: gpt-5 with high reasoning effort)
  python launch_multiagent.py --task "Research transformer attention mechanisms"


  # Resume from existing workspace
  python launch_multiagent.py --resume results/freephdlabor_20250929_143022/ --model claude-sonnet-4-5
  python launch_multiagent.py --resume results/freephdlabor_20250929_143022/ --task "Continue writing the conclusion section"
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        choices=AVAILABLE_MODELS,
        default="gpt-5",
        help="LLM model to use for all agents"
    )

    parser.add_argument(
        "--interpreter",
        type=str,
        default="python",
        help="Python interpreter path for experiment execution"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    # GPT-5 specific parameters
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        choices=["minimal", "low", "medium", "high"],
        default="high",
        help="GPT-5 reasoning effort level (controls thinking depth)"
    )
    
    parser.add_argument(
        "--verbosity",
        type=str,
        choices=["low", "medium", "high"], 
        default="medium",
        help="GPT-5 verbosity level (controls response detail)"
    )

    parser.add_argument(
        "--callback_host",
        type=str,
        default="127.0.0.1",
        help="Host for the callback server"
    )

    parser.add_argument(
        "--callback_port",
        type=int,
        default=5001,
        help="Port for the callback server"
    )

    parser.add_argument(
        "--enable-planning",
        action="store_true",
        help="Enable planning feature for research agents (creates systematic step-by-step plans)"
    )

    parser.add_argument(
        "--planning-interval",
        type=int,
        default=3,
        help="Interval for planning steps (e.g., 3 = replan every 3 steps). Only used if --enable-planning is set."
    )

    parser.add_argument(
        "--resume",
        type=str,
        help="Resume from an existing workspace directory (e.g., results/20250128_143022_task_name/). "
             "If specified, will continue from the existing workspace instead of creating a new one."
    )

    parser.add_argument(
        "--task",
        type=str,
        help="The research task description to be carried out by the multiagent system. Can be used with --resume to tell the system to continue working on a previous task."
    )

    return parser.parse_args()