import json
from typing import Optional, List, Dict
from smolagents import Tool, ChatMessage
from ..utils import extract_content_between_markers


class GenerateIdeaTool(Tool):
    name = "GenerateIdeaTool"
    description = "Generates research ideas by prompting LLM. Takes task description and optional seed ideas for context."
    inputs = {
        "task_description": {
            "type": "string", 
            "description": "Description of the research task, can contain details about the task, the domain, the dataset, the model, the evaluation, the benchmark, the implementation, the code, the results, the analysis, the conclusion, etc.",
            "nullable": True
        },
        "seed_ideas_json": {
            "type": "string", 
            "description": "JSON string containing seed ideas for context (optional)",
            "nullable": True
        }
    }
    output_type = "string"
    
    def __init__(self, model=None):
        """
        Initialize the tool with access to the LLM model.
        
        Args:
            model: The LLM model to use for generating ideas
        """
        super().__init__()
        from .model_utils import get_raw_model
        self.model = get_raw_model(model)
    
    def forward(self, task_description: str = "", seed_ideas_json: str = "") -> str:
        """
        Generate a complete research idea by calling the LLM.
        
        Args:
            task_description: Task description for context
            seed_ideas_json: JSON string containing existing seed ideas (optional)
            
        Returns:
            JSON string containing the generated research idea
        """
        try:
            # Parse seed ideas
            seed_ideas = json.loads(seed_ideas_json) if seed_ideas_json else []
            
            # Create the formatted string of previous ideas
            prev_ideas_string = "\n\n".join([json.dumps(idea) for idea in seed_ideas])
            
            # Create the idea generation prompt
            prompt = f"""{task_description}

Here are the ideas that you have already generated:

'''
{prev_ideas_string}
'''

Now, come up with an impactful and creative research idea based on the task description above.
Your new idea should not be more complex than those you have already generated.

Respond in the following format:

THOUGHT:
<THOUGHT>

NEW IDEA JSON:
```json
<JSON>
```

In <THOUGHT>, first thoroughly discuss your intuitions and motivations for your proposed idea. Detail your high-level plan, necessary design choices and ideal outcomes of the experiments. 
Be as specific as possible, including technical details if necessary.

In <JSON>, provide the new idea in JSON format with the following fields:
- "Name": A shortened descriptor of the idea. Lowercase, no spaces, underscores allowed.
- "Title": A title for the idea, will be used for the report writing.
- "Experiment": An outline of the implementation. E.g. which functions need to be added or modified, how results will be obtained, ...
- "Technical_Details": A precise and verbose technical description of the proposed improvement, using specific technical language and avoiding vague terms. BE SURE TO DEFINE BEFORE YOU USE NONSTANDARD TERMINOLOGY. WHENEVER POSSIBLE, USE MATHEMATICAL LANGUAGE TO AVOID AMBIGUITY.
- "Rationale": An extremely detailed explanation of why the proposed experiment can be expected to improve from the baseline model. Carefully explaining the logic behind every step of reasoning you make. Avoid making unjustified claims about improvement.
- "Implementation_Plan": A plan of steps to implement the experiment described above.

Be cautious and critical in your output.

This JSON will be automatically parsed, so ensure the format is precise. CRITICAL JSON FORMATTING RULES:
- USE PROPER JSON ESCAPING: Escape quotes (\"), backslashes (\\\\), and newlines (\\n)
- AVOID UNICODE MATHEMATICAL SYMBOLS: Instead of τ, π, θ use tau, pi, theta
- AVOID RAW UNICODE ESCAPES: Don't use \\u03c4 - use "tau" instead
- USE ASCII CHARACTERS ONLY in field values to prevent parsing errors
- Example: Instead of "Let τ = (s_0, a_0)" write "Let tau = (s_0, a_0)"
"""
            
            # Call the LLM model to generate the idea
            if self.model is None:
                return json.dumps({"error": "No model provided to GenerateIdeaTool"})
            
            # Generate response using the model with ChatMessage format
            messages = [ChatMessage(role="user", content=prompt)]
            response = self.model.generate(messages).content
            
            # Extract JSON content using the utility function
            json_content = extract_content_between_markers(response, "```json", "```")
            
            if json_content:
                try:
                    # Parse and validate the JSON
                    idea_dict = json.loads(json_content)
                    return json.dumps(idea_dict, indent=2)
                except json.JSONDecodeError as e:
                    return json.dumps({"error": f"Invalid JSON extracted: {str(e)}", "raw_content": json_content[:500]})
            else:
                # No JSON block found, try to parse entire response as JSON
                try:
                    idea_dict = json.loads(response)
                    return json.dumps(idea_dict, indent=2)
                except json.JSONDecodeError:
                    return json.dumps({
                        "error": "Could not extract valid JSON from LLM response",
                        "raw_response": response[:500]
                    })
            
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid JSON in seed_ideas_json: {str(e)}"})
        except Exception as e:
            return json.dumps({"error": f"Error generating idea: {str(e)}"}) 