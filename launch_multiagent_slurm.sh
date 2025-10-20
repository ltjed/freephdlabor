#!/bin/bash

# =============================================================================
# SLURM CONFIGURATION - Modify these for your cluster setup
# =============================================================================
#SBATCH --job-name=freephdlabor_run    # Job name (customize this)
#SBATCH --partition=gpu_h200              # SLURM partition (customize to your cluster)
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks-per-node=1               # Tasks per node
#SBATCH --cpus-per-task=8                 # CPUs per task
#SBATCH --gres=gpu:h200:1                 # GPU resources (customize GPU type)
#SBATCH --time=48:00:00                   # Time limit (HH:MM:SS)
#SBATCH --mem=64G                         # Memory allocation
#SBATCH --output=slurm_outputs/slurm_%j.out   # Standard output log
#SBATCH --error=slurm_outputs/slurm_%j.err    # Error log

# =============================================================================
# JOB INFORMATION - Prints debug info at job start
# =============================================================================
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"
echo "Working directory: $PWD"

# Forward any extra arguments passed to this script to the Python program
if [[ $# -gt 0 ]]; then
	echo "Extra args to Python: $*"
fi

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================
# Load conda module (if required by your cluster)
module load miniconda

# Activate the conda environment
echo "Activating conda environment..."
source ~/.bashrc
conda deactivate
conda activate freephdlabor  # Change to your conda environment name

# Verify Python environment
echo "Python version: $(python --version)"
echo "Python path: $(which python)"

# =============================================================================
# CUDA/GPU SETUP
# =============================================================================
# Set GPU visibility (helps with multi-GPU systems)
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

# Print GPU information for debugging
echo "GPU information:"
nvidia-smi

# =============================================================================
# RESEARCH TASK DEFINITION
# =============================================================================
# Define your research task here - describe what you want the system to do
RESEARCH_TASK="Complete a full research project on [YOUR RESEARCH TOPIC].

RESEARCH OBJECTIVES: (1) [Objective 1], (2) [Objective 2], (3) [Objective 3], ...

WORKFLOW AUTONOMY: You have full autonomy to iterate between agents if any stage reveals limitations. ..."

# =============================================================================
# PROJECT DIRECTORY SETUP
# =============================================================================
# Automatically determine project directory from script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
echo "Project directory: $SCRIPT_DIR"

# =============================================================================
# EXECUTION - Advanced wrapper to prevent process name conflicts
# =============================================================================
# Create a temporary wrapper to hide "python" from process command line
# This prevents AI-Scientist-v2's cleanup routine from accidentally killing the main process
echo "Starting multiagent system..."

# Get the active Python interpreter from conda environment
PYTHON_PATH=$(which python)

# Create temporary runner script (doesn't contain "python" in filename to avoid run_experiment_tool's cleanup routine from accidentally killing the main process)
RUNNER_SCRIPT=$(mktemp /tmp/multiagent_runner_XXXXXX)
cat > "$RUNNER_SCRIPT" << EOF
#!/bin/bash
exec $PYTHON_PATH launch_multiagent.py "\$@"
EOF
chmod +x "$RUNNER_SCRIPT"

# Ensure cleanup of temporary wrapper on exit
cleanup() { rm -f "$RUNNER_SCRIPT"; }
trap cleanup EXIT

# =============================================================================
# LAUNCH MULTIAGENT SYSTEM
# =============================================================================
echo "=== Multiagent system started at: $(date) ==="

# Execute via wrapper
"$RUNNER_SCRIPT" --task "$RESEARCH_TASK" "$@"
exit_code=$?

echo "=== Multiagent system completed with exit code: $exit_code at: $(date) ==="

# =============================================================================
# CLEANUP AND REPORTING
# =============================================================================
echo "Job completed at: $(date)"
echo "Final exit code: $exit_code"
exit $exit_code
