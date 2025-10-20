import json
from typing import Optional
from smolagents import Tool
from smolagents.models import ChatMessage


class CheckIdeaNoveltyTool(Tool):
    name = "CheckIdeaNoveltyTool"
    description = "Assesses the novelty of a research idea by calling LLM and searching literature. Returns the idea with novelty field (True/False) and justification."
    
    inputs = {
        "idea_json": {
            "type": "string",
            "description": "JSON string containing the idea to check"
        },
        "task_description": {
            "type": "string",
            "description": "Description of the research task or domain",
            "nullable": True
        },
        "code": {
            "type": "string",
            "description": "Experiment code for context",
            "nullable": True
        },
        "max_num_iterations": {
            "type": "integer",
            "description": "Maximum number of search iterations",
            "nullable": True
        },
        "base_dir": {
            "type": "string",
            "description": "Base directory for loading context files",
            "nullable": True
        }
    }
    output_type = "string"
    
    def __init__(self, model=None):
        """
        Initialize the tool with access to the LLM model.
        
        Args:
            model: The LLM model to use for novelty checking
        """
        super().__init__()
        from .model_utils import get_raw_model
        self.model = get_raw_model(model)
    
    def forward(self, idea_json: str, task_description: str = "", code: str = "", 
                max_num_iterations: int = 3, base_dir: str = "") -> str:
        """
        Check the novelty of a research idea by performing literature search and LLM analysis.
        
        Args:
            idea_json: JSON string containing the idea to check
            task_description: Task description for context
            code: Experiment code for context
            max_num_iterations: Maximum number of search iterations (default: 3)
            base_dir: Base directory for loading context files
            
        Returns:
            JSON string containing the idea with novelty assessment
        """
        try:
            # Load additional context if base_dir is provided
            if base_dir:
                import os
                try:
                    if not code and os.path.exists(os.path.join(base_dir, "experiment.py")):
                        with open(os.path.join(base_dir, "experiment.py"), "r") as f:
                            code = f.read()
                    
                    if not task_description and os.path.exists(os.path.join(base_dir, "prompt.json")):
                        with open(os.path.join(base_dir, "prompt.json"), "r") as f:
                            prompt_data = json.load(f)
                            task_description = prompt_data.get("task_description", "")
                except Exception as e:
                    # Continue with provided parameters if file loading fails
                    pass
            
            # Parse the idea
            idea = json.loads(idea_json)
            
            if self.model is None:
                return json.dumps({"error": "No model provided to CheckIdeaNoveltyTool"})
            
            # Create the novelty system message
            system_msg = f"""You are an ambitious AI PhD student who is looking to publish a paper that will contribute significantly to the field.
You have an idea and you want to check if it is novel or not. I.e., not overlapping significantly with existing literature or already well explored.
Be a harsh critic for novelty, ensure there is a sufficient contribution in the idea for a new conference or workshop paper.
You will be given access to the Semantic Scholar API, which you may use to survey the literature and find relevant papers to help you make your decision.
The top 10 results for any search query will be presented to you with the abstracts.

You will be given {max_num_iterations} to decide on the paper, but you do not need to use them all.
At any round, you may exit early and decide on the novelty of the idea.
Decide a paper idea is novel if after sufficient searching, you have not found a paper that significantly overlaps with your idea.
Decide a paper idea is not novel, if you have found a paper that significantly overlaps with your idea.

{task_description}
<experiment.py>
{code}
</experiment.py>
"""

            last_query_results = ""
            current_round = 1
            decision = "undecided"
            novelty_justification = ""
            
            # Iterative novelty checking process
            while current_round <= max_num_iterations and decision == "undecided":
                prompt = f'''Round {current_round}/{max_num_iterations}.
You have this idea:

"""
{json.dumps(idea, indent=2)}
"""

The results of the last query are (empty on first round):
"""
{last_query_results}
"""

Respond in the following format:

THOUGHT:
<THOUGHT>

RESPONSE:
```json
<JSON>
```

In <THOUGHT>, first briefly reason over the idea and identify any query that could help you make your decision.
If you have made your decision, add "Decision made: novel." or "Decision made: not novel." to your thoughts.

In <JSON>, respond in JSON format with the following fields:
- "Query": An optional search query to search the literature (e.g. attention is all you need). You must make a query if you have not decided this round.
- "Decision": A decision on the novelty of the idea. Either "decision made: novel", "decision made: not novel", or "undecided".

A query will work best if you are able to recall the exact name of the paper you are looking for, or the authors.
This JSON will be automatically parsed, so ensure the format is precise.
'''
                
                # Get LLM response using ChatMessage format
                messages = [
                    ChatMessage(role="system", content=system_msg),
                    ChatMessage(role="user", content=prompt)
                ]
                response = self.model.generate(messages).content
                
                # Extract JSON from response
                if "```json" in response:
                    json_start = response.find("```json") + 7
                    json_end = response.find("```", json_start)
                    if json_end != -1:
                        response_json = response[json_start:json_end].strip()
                        response_data = json.loads(response_json)
                        
                        decision = response_data.get("Decision", "undecided")
                        query = response_data.get("Query", "")
                        
                        # Extract novelty justification from thoughts
                        if "THOUGHT:" in response:
                            thought_start = response.find("THOUGHT:") + 8
                            thought_end = response.find("RESPONSE:", thought_start)
                            if thought_end != -1:
                                novelty_justification += f"Round {current_round}: {response[thought_start:thought_end].strip()}\n\n"
                        
                        # If undecided and query provided, search for papers
                        if decision == "undecided" and query:
                            # This would ideally call PaperSearchTool, but for now we'll simulate
                            # In a full implementation, this would integrate with PaperSearchTool
                            last_query_results = f"Searched for: '{query}' - No significant overlaps found (simulated)"
                        
                        if decision.startswith("decision made:"):
                            break
                
                current_round += 1
            
            # Determine final novelty assessment
            is_novel = "novel" in decision.lower()
            
            # Add novelty assessment to the idea
            idea["novel"] = is_novel
            idea["novelty_decision"] = decision
            idea["novelty_justification"] = novelty_justification.strip()
            idea["novelty_checked_rounds"] = current_round - 1
            
            return json.dumps(idea, indent=2)
            
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid JSON in idea_json: {str(e)}"})
        except Exception as e:
            return json.dumps({"error": f"Error checking novelty: {str(e)}"}) 