You are tasked with analyzing agent instructions from the agent's perspective to evaluate whether they have sufficient information, context, and clarity to successfully perform requested tasks.

## Arguments provided: $ARGUMENTS

The arguments should contain:
1. **agent_name** (required): The specific agent to analyze (manager, ideation, experimentation, or writeup)
2. **results_dir** (optional): A specific results directory to analyze (e.g., "20250715_152703_adaptive_lr_cnn"). If not specified, run a new experiment first.

## Your workflow:

### Step 1: Determine Starting Point
- If a results_dir is provided, skip to Step 2 with that directory
- Otherwise, run the multi-agent system:
```bash
eval "$(conda shell.bash hook)" && conda activate freephdlabor && export $(cat .env | grep -v '^#' | xargs) && python launch_multiagent.py --model o4-mini-2025-04-16 2>&1 | tee logs/output_$(date +%Y%m%d_%H%M%S).log
```

### Step 2: Load Agent Instructions and Context
- Read the current instructions for the specified agent: `freephdlabor/prompts/{agent_name}_instructions.py`
- Review the agent's available tools in: `freephdlabor/agents/{agent_name}_agent.py`
- Understand the agent's role within the multi-agent system

### Step 3: Analyze Agent Task Reception Points
- Read the JSONL file at `results/{results_dir}/agent_llm_calls.jsonl`
- **Critical**: Find all occurrences when the specified agent **first receives a new task** from other agents
- Since agent_llm_calls.jsonl is organized in temporal order, identify when the agent is initially invoked by another agent
- For each task reception point, extract:
  - The exact task/instruction given to the agent
  - The context provided by the requesting agent
  - The agent's initial response and understanding
  - Any clarification requests or confusion indicators

### Step 4: Perspective Analysis for Each Task Reception
For each identified task reception point, put yourself in the agent's shoes and evaluate:

**Information Sufficiency at Task Reception:**
- Did the agent receive enough context to understand what was being asked?
- Was the task scope and boundaries clearly communicated?
- Did the agent have sufficient background information about the project/experiment?
- Were dependencies and prerequisites clearly stated?

**Contextual Clarity in Task Assignment:**
- Could the agent determine what constitutes successful task completion?
- Were quality standards and expectations explicitly communicated?
- Did the requesting agent provide enough domain-specific context?
- Was the urgency/priority of the task clear?

**Execution Readiness:**
- Could the agent immediately understand which tools to use?
- Were the procedural steps implied or explicitly stated?
- Did the agent have clear decision-making criteria for the task?
- Could the agent identify when to seek clarification vs. proceed autonomously?

### Step 5: Gap Identification from Real Interactions
Based on the actual task reception moments, identify specific gaps:
- **Information gaps**: What context was missing when tasks were assigned?
- **Communication gaps**: Where did inter-agent communication break down?
- **Instruction gaps**: What procedural guidance was unclear or missing?
- **Coordination gaps**: Where did agents misunderstand their roles or responsibilities?

### Step 6: Improvement Proposals Based on Real Data
For each identified gap from actual task receptions, propose specific improvements:
- **What to add to instructions**: Missing context patterns that repeatedly caused confusion
- **What to clarify**: Ambiguous communication patterns between agents
- **What to restructure**: Agent instruction improvements for better task comprehension
- **What examples to provide**: Concrete patterns for common task types observed

### Step 7: Report Findings
Provide a structured analysis including:

1. **Task Reception Analysis**: Summary of all moments when the agent received new tasks
2. **Contextual Sufficiency Assessment**: For each task reception, evaluate if the agent had enough information
3. **Communication Pattern Analysis**: How well other agents communicated tasks to this agent
4. **Critical Gaps Found**: Most important missing information or unclear instructions based on real interactions
5. **Specific Improvement Recommendations**: Concrete suggestions based on observed patterns
6. **Instruction Enhancement Proposals**: Specific changes to agent instructions to prevent identified issues

## Analysis Framework

When evaluating each task reception moment, ask these questions from the agent's perspective:
- "Did I receive enough information to start this task immediately?"
- "Was the success criteria for this task clearly communicated to me?"
- "Did I understand which other agents I need to coordinate with and how?"
- "Could I identify the right tools and sequence for this specific task?"
- "Was I given sufficient context about the broader project goals?"

## Output Format

Structure your analysis as:
- **Executive Summary**: Overview of how well the agent receives and understands tasks
- **Task Reception Timeline**: Chronological analysis of key task assignment moments
- **Gap Analysis**: Specific information/context deficiencies identified from real interactions
- **Communication Improvement Roadmap**: How other agents can better communicate with this agent
- **Instruction Enhancement Plan**: Specific changes to agent instructions based on findings

## Key Files
- **Agent Instructions**: `freephdlabor/prompts/{agent_name}_instructions.py`
- **Agent Implementation**: `freephdlabor/agents/{agent_name}_agent.py`
- **Related Tools**: `freephdlabor/toolkits/` (various tool implementations)