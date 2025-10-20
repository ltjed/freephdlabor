"""
Research-focused planning prompt templates for our agents.

These templates are optimized for research tasks and integrate with our
tool-centric multi-agent architecture.
"""

RESEARCH_PLANNING_TEMPLATES = {
    "initial_plan": """You are an expert research agent analyzing a research task to develop a systematic approach.

Your task is to create a comprehensive plan for the following research objective:
```
{{task}}
```

## 1. Research Context Analysis

### 1.1. Task Requirements
List the specific research deliverables and success criteria from the task description.

### 1.2. Available Resources
- **Tools**: Review your available tools for data gathering, analysis, and execution
- **Managed Agents**: Note any specialist agents you can delegate subtasks to
- **Workspace**: Consider what files/data may already exist in your workspace

### 1.3. Knowledge Gaps
Identify what information, data, or analysis you need to acquire to complete the task.

## 2. Research Strategy

Develop a step-by-step research plan that:
- Uses available tools efficiently
- Delegates appropriate subtasks to specialist agents
- Builds knowledge incrementally
- Produces verifiable results
- Meets all task requirements

Your available tools:
```python
{%- for tool in tools.values() %}
def {{ tool.name }}({% for arg_name, arg_info in tool.inputs.items() %}{{ arg_name }}: {{ arg_info.type }}{% if not loop.last %}, {% endif %}{% endfor %}) -> {{tool.output_type}}:
    \"\"\"{{ tool.description }}\"\"\"
{% endfor %}
```

{%- if managed_agents and managed_agents.values() | list %}
Your specialist team members:
```python
{%- for agent in managed_agents.values() %}
def {{ agent.name }}(task: str, additional_args: dict[str, Any]) -> str:
    \"\"\"{{ agent.description }}\"\"\"
{% endfor %}
```
{%- endif %}

Write your plan as numbered steps that can be executed systematically.
After your final step, write '<end_plan>' and stop.

**Remember**: Focus on the research methodology, not implementation details.""",

    "update_plan_pre_messages": """You are an expert research agent reviewing progress on a research task.

Original task:
```
{{task}}
```

Below you will see your previous progress and any obstacles encountered.
Use this information to assess what has been accomplished and what needs adjustment.""",

    "update_plan_post_messages": """Based on the above progress review, create an updated research plan:

## 1. Progress Assessment

### 1.1. Completed Objectives
List what research objectives have been successfully completed.

### 1.2. Partial Progress
Note any objectives that are partially complete and what remains.

### 1.3. Obstacles Encountered
Identify any technical, methodological, or resource constraints discovered.

### 1.4. New Information
List any new insights or requirements that have emerged.

## 2. Updated Research Strategy

Create a revised plan that:
- Builds on completed work
- Addresses identified obstacles
- Incorporates new requirements
- Optimizes remaining steps ({remaining_steps} steps remaining)

Available tools and agents remain the same as before.

Write your updated plan as numbered steps.
After your final step, write '<end_plan>' and stop."""
}


def get_research_planning_templates():
    """Get research-focused planning templates for our agents."""
    return RESEARCH_PLANNING_TEMPLATES