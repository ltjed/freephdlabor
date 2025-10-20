"""
Workspace Management - Shared workspace guidance for all agents.
"""

# Shared workspace guidance that all agents should know
WORKSPACE_GUIDANCE = """
WORKSPACE SYSTEM:
- You work in a shared workspace with other agents
- Each agent has their own subdirectory for temporary/intermediate files
- Use file editing tools to coordinate and share information
- Create clear documentation for other agents to reference

STANDARD WORKSPACE FILES (always available):
These files are managed exclusively by the ManagerAgent and are READ-ONLY for all other agents:

1. past_ideas_and_results.md - Historical record of previous research attempts
   - Documents what ideas have been tried before
   - Contains results, outcomes, and lessons learned
   - Helps avoid duplicate work and builds on previous insights

2. working_idea.json - Current research idea being developed
   - Structured format containing the active research hypothesis
   - Includes experimental design and implementation details
   - Updated by ManagerAgent as the idea evolves

READ THESE FILES to understand:
- What has been tried before (past_ideas_and_results.md)  
- What the team is currently working on (working_idea.json)

FILE ACCESS INSTRUCTIONS:
Always read a file first before editing it to avoid information loss.
READING FILES:
- Shared files: Read directly from workspace base directory
- Agent files: Read from your agent folder for your own files
- Other agent files: Use full paths provided by other agents
- Always check file existence before reading

CREATING FILES - DECISION GUIDE:
Save in your own agent directory when:
- File is intermediate work or drafts
- File is only needed by you for current task
- File contains debugging info or logs
- File is temporary or will be deleted soon
- Examples: review_agent/literature_notes.md

Save in SHARED ROOT DIRECTORY when:
- File contains final results other agents need
- File will be referenced by multiple agents
- File represents completed work ready for team use


FILE NAMING CONVENTIONS:
- Use descriptive names
- Include timestamps for versioned files: analysis_20241220_143022.md
- Use appropriate extensions: .json for data, .md for documentation, .txt for logs

INTER-AGENT COMMUNICATION:
When you create files in the SHARED ROOT DIRECTORY:
- Always report to ManagerAgent: "Created [filename] containing [brief description]"
- This ensures other agents know the file exists and its purpose

ERROR HANDLING:
- If a required file doesn't exist, inform clearly and suggest alternatives
- Don't assume file contents - always verify by reading
- Check if files might be located elsewhere in the workspace

WORKSPACE MAINTENANCE:
- Keep your agent folder organized
- Remove temporary files when no longer needed
- Update shared files responsibly (consider impact on other agents)
- Document important decisions in progress files"""


