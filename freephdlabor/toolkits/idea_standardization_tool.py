from smolagents import Tool
import json
import logging


class IdeaStandardizationTool(Tool):
    """
    Converts research ideas from various agent formats to the standardized AI-Scientist-v2 format 
    using intelligent LLM-based mapping. This ensures that detailed research proposals are properly 
    preserved and formatted for AI-Scientist-v2 consumption.
    """
    name: str = "IdeaStandardizationTool"
    description: str = (
        "Converts research ideas from agent formats to AI-Scientist-v2 format using intelligent "
        "LLM-based mapping. Handles varied schema formats and preserves research content that "
        "would otherwise be lost in format conversion. "
        "MANDATORY: Must be used BEFORE RunExperimentTool to prevent experiments defaulting to "
        "wrong models (e.g., DistilBERT instead of Pythia) or synthetic data instead of real datasets."
    )
    
    inputs = {
        "idea_json": {
            "type": "string",
            "description": "Research idea in JSON string format from any agent schema (e.g., ManagerAgent output)"
        }
    }
    output_type = "string"

    def __init__(self, model=None):
        """
        Initialize IdeaStandardizationTool with raw LiteLLMModel.
        
        Args:
            model: LiteLLMModel instance (will extract raw model if LoggingLiteLLMModel wrapper)
        """
        super().__init__()
        # CRITICAL: Use get_raw_model to avoid LoggingLiteLLMModel interface issues
        from .model_utils import get_raw_model
        self.model = get_raw_model(model)
        
        # CRITICAL FIX: Ensure API keys are properly configured for litellm
        self._configure_api_keys()
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
    
    def _configure_api_keys(self):
        """Configure API keys for litellm from environment variables."""
        import os
        import litellm
        
        # Set Google/Gemini API key if available
        google_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GOOGLEAI_API_KEY')
        if google_key:
            litellm.api_key = google_key
            
        # Set other API keys that might be needed
        if os.getenv('OPENAI_API_KEY'):
            litellm.openai_key = os.getenv('OPENAI_API_KEY')
        if os.getenv('ANTHROPIC_API_KEY'):
            litellm.anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    
    def forward(self, idea_json: str) -> str:
        """
        Convert research idea to AI-Scientist-v2 format using intelligent LLM mapping.
        
        Args:
            idea_json: JSON string containing research idea in agent format
            
        Returns:
            JSON string in AI-Scientist-v2 format ready for RunExperimentTool
        """
        try:
            # Parse input JSON with enhanced error handling
            if isinstance(idea_json, str):
                # Clean input string to handle line-number prefixes from see_file output
                cleaned_json = self._clean_json_string(idea_json)
                idea_data = json.loads(cleaned_json)
            else:
                idea_data = idea_json
            
            # Validate that we have a dictionary, not a list
            if isinstance(idea_data, list):
                if len(idea_data) > 0 and isinstance(idea_data[0], dict):
                    idea_data = idea_data[0]  # Take first item if it's a list of dicts
                else:
                    raise ValueError("Input appears to be a list but doesn't contain valid research idea dictionaries")
            
            if not isinstance(idea_data, dict):
                raise ValueError(f"Expected dictionary or JSON object, got {type(idea_data)}")
                
            self.logger.info(f"Converting research idea with keys: {list(idea_data.keys())}")
            
            # Try LLM-based intelligent conversion
            try:
                standardized = self._llm_based_conversion(idea_data)
                self.logger.info("LLM-based conversion successful")
                return json.dumps([standardized])
                
            except Exception as llm_error:
                self.logger.warning(f"LLM conversion failed: {llm_error}, using rule-based fallback")
                # Fallback to rule-based conversion
                standardized = self._rule_based_conversion(idea_data)
                self.logger.info("Rule-based fallback conversion completed")
                return json.dumps([standardized])
                
        except Exception as e:
            self.logger.error(f"Idea standardization failed: {e}")
            self.logger.error(f"Input type: {type(idea_json)}, Input preview: {str(idea_json)[:200]}...")
            return json.dumps([{
                "Name": "conversion_failed",
                "Title": "Research Idea Conversion Failed",
                "Short Hypothesis": "Unable to convert research idea format",
                "Abstract": f"Conversion error: {str(e)}",
                "Experiments": ["Manual intervention required"],
                "Risk Factors and Limitations": ["Format conversion failed", "Original idea may be malformed"],
                "Related Work": "This represents a failed format conversion attempt."
            }])
    
    def _clean_json_string(self, json_str: str) -> str:
        """Clean JSON string to handle line-number prefixes and other formatting issues."""
        import re
        
        # Remove line number prefixes (e.g., "1:", "2:", etc.) - common from see_file output
        # Only remove if they appear at the start of lines and look like line numbers
        lines = json_str.split('\n')
        cleaned_lines = []
        for line in lines:
            # Only remove if it's clearly a line number prefix (digits + colon at start of line)
            # and the line contains more content after the colon
            if re.match(r'^\s*\d+:', line) and len(line.split(':', 1)) > 1:
                # Remove only the first occurrence of digits followed by colon at line start
                cleaned_line = re.sub(r'^\s*\d+:', '', line, count=1)
            else:
                cleaned_line = line
            cleaned_lines.append(cleaned_line)
        
        return '\n'.join(cleaned_lines).strip()
    
    def _llm_based_conversion(self, idea_data):
        """Use LLM to intelligently convert research idea to AI-Scientist-v2 format."""
        
        # Create flexible conversion prompt
        prompt = self._create_conversion_prompt(idea_data)
        
        # Call raw LiteLLMModel (not LoggingLiteLLMModel)
        if not self.model:
            raise Exception("No LLM model available for conversion")
        
        # Use ChatMessage format like other tools  
        from smolagents.models import ChatMessage
        messages = [ChatMessage(role="user", content=prompt)]
        response = self.model(messages)
        
        # Extract content from response
        if hasattr(response, 'content'):
            response_text = response.content
        elif isinstance(response, str):
            response_text = response
        else:
            response_text = str(response)
        
        # Parse LLM response
        try:
            # Try to extract JSON from response - robust approach
            import re
            
            # First try: Look for ```json code blocks
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_content = json_match.group(1).strip()
                standardized = json.loads(json_content)
            else:
                # Second try: Look for standalone JSON objects
                json_match = re.search(r'(\{.*\})', response_text, re.DOTALL)
                if json_match:
                    json_content = json_match.group(1).strip()
                    standardized = json.loads(json_content)
                else:
                    # Third try: Parse entire response as JSON
                    standardized = json.loads(response_text.strip())
            
            # Validate required fields
            self._validate_ai_scientist_format(standardized)
            return standardized
            
        except (json.JSONDecodeError, KeyError) as e:
            raise Exception(f"LLM response parsing failed: {e}")
    
    def _create_conversion_prompt(self, idea_data):
        """Create flexible conversion prompt for LLM."""
        
        prompt = f"""Convert this research idea to the AI-Scientist-v2 format. Keep as much information as possible from the original but fit it into the correct fields.

Original Research Idea:
{json.dumps(idea_data, indent=2)}

Convert to this JSON structure:
- Name: Create a short identifier from the title (lowercase, underscores, under 50 chars)
- Title: Use the exact title from the original
- Short Hypothesis: Extract the main research question or hypothesis
- Abstract: Preserve the full technical description and methodology - include ALL important details
- Experiments: List all experimental steps described in the original (include as many as needed)
- Risk Factors and Limitations: Generate realistic limitations based on the research domain
- Related Work: Provide brief context about related research

Example of good conversion:
```json
{{
    "Name": "self_distillation_anchoring",
    "Title": "Self-Distillation Anchoring: Mitigating Catastrophic Forgetting in Small Language Models with a Functional Snapshot",
    "Short Hypothesis": "Self-distillation using an anchor dataset can significantly reduce catastrophic forgetting while maintaining task performance",
    "Abstract": "We propose Self-Distillation Anchoring (SDA), which creates a small anchor dataset capturing the pre-trained model's behavior and uses self-distillation loss during fine-tuning to preserve general capabilities...",
    "Experiments": [
        "Generate anchor dataset using pre-trained Pythia-410M on diverse prompts",
        "Fine-tune baseline model on Alpaca dataset with standard SFT",
        "Fine-tune SDA model on Alpaca with composite loss function",
        "Evaluate both models on HellaSwag, ARC-Challenge, and MMLU benchmarks",
        "Compare task performance using AlpacaEval"
    ],
    "Risk Factors and Limitations": [
        "Anchor dataset quality affects regularization effectiveness",
        "Lambda hyperparameter requires careful tuning",
        "Method effectiveness may vary across model architectures"
    ],
    "Related Work": "This work builds upon continual learning and knowledge distillation research in language models."
}}
```

Output ONLY the JSON, no explanations."""

        return prompt
    
    def _rule_based_conversion(self, idea_data):
        """Fallback rule-based conversion when LLM fails."""
        
        # Map common field names to AI-Scientist-v2 format
        title = self._find_field(idea_data, ["title", "Title", "research_title", "name"])
        
        # Enhanced abstract extraction - look for multiple sources of content
        abstract = self._find_field(idea_data, [
            "abstract", "Abstract", "description", "summary", "rationale", "Rationale", 
            "technical_details", "Technical_Details", "experiment", "Experiment"
        ])
        
        # Enhanced hypothesis extraction
        hypothesis = self._find_field(idea_data, [
            "research_question", "hypothesis", "short_hypothesis", 
            "research_hypothesis", "question", "rationale", "Rationale"
        ])
        
        # Generate Name from title
        if title:
            name_words = [w.lower() for w in title.split() if len(w) > 2][:4]
            name = "_".join(name_words).replace("-", "_")[:50]
        else:
            name = "research_investigation"
        
        # Extract experiments (flexible number)
        experiments = self._extract_experiments_rule_based(idea_data)
        
        # Generate risk factors (flexible number)
        risk_factors = self._generate_risk_factors(idea_data)
        
        return {
            "Name": name,
            "Title": title or "Research Investigation",
            "Short Hypothesis": hypothesis or abstract or "Research investigation hypothesis",
            "Abstract": abstract or "Research investigation abstract",
            "Experiments": experiments,
            "Risk Factors and Limitations": risk_factors,
            "Related Work": "This work builds upon standard machine learning research practices."
        }
    
    def _find_field(self, data, candidates):
        """Find first available field from candidates list."""
        for candidate in candidates:
            if candidate in data and data[candidate]:
                return data[candidate]
        return None
    
    def _extract_experiments_rule_based(self, idea_data):
        """Extract experiments using rule-based logic - flexible number."""
        experiments = []
        
        # Check common locations for experiment descriptions
        experiment_sources = [
            ("experimental_design", "procedure"),
            ("methodology", "experimental_design"),  
            ("experimental_setup", "procedure"),
            ("method", "details"),
            ("proposed_method", "details")
        ]
        
        for main_key, sub_key in experiment_sources:
            if main_key in idea_data:
                section = idea_data[main_key]
                if isinstance(section, dict):
                    if sub_key in section:
                        proc = section[sub_key]
                        if isinstance(proc, list):
                            experiments.extend(proc)
                        elif isinstance(proc, str):
                            experiments.append(proc)
                    
                    # Extract other experimental details
                    for key, value in section.items():
                        if key in ["baselines", "evaluation", "implementation", "approach"]:
                            if isinstance(value, str):
                                experiments.append(f"{key.title()}: {value}")
                            elif isinstance(value, list):
                                for item in value:
                                    if isinstance(item, str):
                                        experiments.append(f"{key.title()}: {item}")
        
        # Look for procedure/steps in top level
        for key in ["procedure", "steps", "experimental_steps", "methodology"]:
            if key in idea_data:
                value = idea_data[key]
                if isinstance(value, list):
                    experiments.extend([str(item) for item in value])
                elif isinstance(value, str):
                    experiments.append(value)
        
        # Fallback experiments
        if not experiments:
            title = self._find_field(idea_data, ["title", "Title"])
            if title:
                experiments.append(f"Implement and evaluate {title}")
                experiments.append("Compare against baseline methods")
                experiments.append("Analyze experimental results")
            else:
                experiments = [
                    "Conduct experimental investigation", 
                    "Evaluate proposed method",
                    "Compare results with baselines"
                ]
        
        return experiments  # Return all experiments, don't limit artificially
    
    def _generate_risk_factors(self, idea_data):
        """Generate realistic risk factors based on research content - flexible number."""
        risks = []
        
        # Always include basic research risks
        risks.append("Experimental results may vary based on hyperparameter choices and random seeds")
        
        # Analyze content for domain-specific risks
        content = " ".join([
            str(idea_data.get("title", "")),
            str(idea_data.get("abstract", "")),
            str(idea_data.get("research_question", ""))
        ]).lower()
        
        if "small model" in content or "small llm" in content or "<0.5b" in content:
            risks.append("Limited model capacity may constrain performance on complex tasks")
        
        if "catastrophic forgetting" in content:
            risks.append("Forgetting effects may be sensitive to task sequence and similarity")
            risks.append("Sequential learning may require careful task ordering")
            
        if "fine-tuning" in content or "finetuning" in content:
            risks.append("Fine-tuning performance may depend heavily on learning rate and duration")
        
        if "dataset" in content:
            risks.append("Results may be dataset-specific and require validation on additional datasets")
            
        if "computational" in content or "resource" in content:
            risks.append("Computational resource constraints may limit experimental scope")
            
        if "memory" in content or "efficiency" in content:
            risks.append("Memory efficiency gains may come at the cost of model performance")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_risks = []
        for risk in risks:
            if risk not in seen:
                seen.add(risk)
                unique_risks.append(risk)
        
        return unique_risks  # Return all relevant risks
    
    def _validate_ai_scientist_format(self, data):
        """Validate that converted data has required AI-Scientist-v2 fields."""
        required_fields = ["Name", "Title", "Short Hypothesis", "Abstract", "Experiments", "Risk Factors and Limitations"]
        
        for field in required_fields:
            if field not in data:
                raise KeyError(f"Missing required field: {field}")
            if not data[field]:
                raise ValueError(f"Empty required field: {field}")