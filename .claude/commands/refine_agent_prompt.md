You are tasked with optimizing the multi-agent workflow by iteratively improving agent instructions to produce high-quality papers publishable at top AI conferences.

## Arguments provided: $ARGUMENTS

The arguments may or may not contain the following:
1. **agent_name** (optional): The specific agent to optimize (manager, ideation, experimentation, or writeup). If not specified, analyze all agents.
2. **results_dir** (optional): A specific results directory to analyze (e.g., "20250715_152703_adaptive_lr_cnn"). If not specified, run a new experiment first.

## Your workflow:

### Step 1: Determine Starting Point
- If a results_dir is provided, skip to Step 2 with that directory
- Otherwise, run the multi-agent system:
```bash
eval "$(conda shell.bash hook)" && conda activate freephdlabor && export $(cat .env | grep -v '^#' | xargs) && python launch_multiagent.py --model o4-mini-2025-04-16 2>&1 | tee logs/output_$(date +%Y%m%d_%H%M%S).log
```

### Step 2: Analyze Results
- Read the JSONL file at `results/{results_dir}/agent_llm_calls.jsonl`
- Focus on the specified agent_name if provided, otherwise analyze all agents
- Identify issues such as:
  - Agent coordination problems
  - Task execution efficiency issues  
  - Research quality and depth problems
  - Experimental design and execution issues
  - Inter-agent communication breakdowns

### Step 3: Improve Instructions
Based on identified issues, modify the corresponding instruction files:
- For specific agent: `freephdlabor/prompts/{agent_name}_instructions.py`
- For all agents: Review and update all instruction files as needed

**Important constraints**:
- Instructions are system prompts - keep them concise and actionable
- Focus on improving agent behavior and coordination
- Prioritize research quality and paper publishability
- Do NOT modify code structure, only instruction content

### Step 4: Commit Changes
Create a git commit with a clear message summarizing:
- Which agent(s) were optimized
- Key issues identified
- Main improvements made

### Step 5: Report Results
Provide a summary of:
1. Issues identified in the agent conversation flow
2. Specific changes made to instructions
3. Expected improvements for next iteration
4. Recommendations for further optimization

## Key Files
- **Agent Instructions**: `freephdlabor/prompts/{agent_name}_instructions.py`
- **Results**: `results/{YYYYMMDD_HHMMSS_idea_name}/agent_llm_calls.jsonl`
