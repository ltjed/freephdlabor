"""
Validation Strategies for the Smolagents Research System.

This module implements validation strategies to prevent hallucination and
ensure research quality as specified in the Full Smolagents Agentization Plan.
"""

import json
import re
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from abc import ABC, abstractmethod


class ValidationStrategy(ABC):
    """Abstract base class for validation strategies."""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.logger = logging.getLogger(__name__)
        if debug:
            self.logger.setLevel(logging.DEBUG)
    
    @abstractmethod
    def validate(self, output: Any, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate output against specific criteria."""
        pass


class OutputValidationStrategy(ValidationStrategy):
    """Validates that agent output matches assigned task requirements."""
    
    def validate(self, output: Any, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate output against task requirements.
        
        Checks:
        - Output format matches expectations
        - Required deliverables are present
        - Output scope aligns with task
        - Quality standards are met
        """
        validation_score = self._calculate_validation_score(output, task_context)
        
        result = {
            "strategy": "output_validation",
            "validation_score": validation_score,
            "missing_deliverables": self._identify_missing_deliverables(output, task_context),
            "format_compliance": self._check_format_compliance(output, task_context),
            "scope_alignment": self._check_scope_alignment(output, task_context),
            "timestamp": datetime.now().isoformat()
        }
        
        self.logger.debug(f"Output validation result: {result}")
        return result
    
    def _calculate_validation_score(self, output: Any, task_context: Dict[str, Any]) -> float:
        """Calculate overall validation score."""
        score = 0.0
        
        # Check if output is not empty
        if output and str(output).strip():
            score += 0.3
        
        # Check if output relates to the task
        task_keywords = self._extract_task_keywords(task_context)
        output_text = str(output).lower()
        
        matching_keywords = sum(1 for keyword in task_keywords if keyword in output_text)
        if task_keywords:
            score += 0.4 * (matching_keywords / len(task_keywords))
        
        # Check output length (reasonable length indicates substance)
        output_length = len(str(output))
        if output_length > 100:  # Minimum substantial output
            score += 0.2
        if output_length > 500:  # More detailed output
            score += 0.1
        
        return min(score, 1.0)
    
    def _identify_missing_deliverables(self, output: Any, task_context: Dict[str, Any]) -> List[str]:
        """Identify missing deliverables from the output."""
        missing = []
        
        # Check for common research deliverables
        output_text = str(output).lower()
        
        # Check for research components
        if "research" in task_context.get("task_type", ""):
            if "idea" not in output_text and "concept" not in output_text:
                missing.append("research_idea")
            if "methodology" not in output_text and "method" not in output_text:
                missing.append("methodology")
            if "literature" not in output_text and "related work" not in output_text:
                missing.append("literature_review")
        
        # Check for experiment components
        if "experiment" in task_context.get("task_type", ""):
            if "result" not in output_text:
                missing.append("experimental_results")
            if "analysis" not in output_text:
                missing.append("result_analysis")
        
        return missing
    
    def _check_format_compliance(self, output: Any, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """Check if output format complies with expectations."""
        compliance = {
            "has_structure": False,
            "has_details": False,
            "format_score": 0.0
        }
        
        output_text = str(output)
        
        # Check for structured content
        if len(output_text.split('\n')) > 3:  # Multiple lines suggest structure
            compliance["has_structure"] = True
            compliance["format_score"] += 0.5
        
        # Check for detailed content
        if len(output_text) > 200:  # Reasonable detail
            compliance["has_details"] = True
            compliance["format_score"] += 0.5
        
        return compliance
    
    def _check_scope_alignment(self, output: Any, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """Check if output scope aligns with task requirements."""
        alignment = {
            "scope_match": False,
            "focus_maintained": False,
            "alignment_score": 0.0
        }
        
        output_text = str(output).lower()
        task_description = task_context.get("description", "").lower()
        
        # Check for scope match
        if task_description:
            common_words = set(task_description.split()) & set(output_text.split())
            if len(common_words) > 3:  # Reasonable overlap
                alignment["scope_match"] = True
                alignment["alignment_score"] += 0.5
        
        # Check focus maintenance (not too broad or narrow)
        if 50 < len(output_text) < 2000:  # Reasonable length range
            alignment["focus_maintained"] = True
            alignment["alignment_score"] += 0.5
        
        return alignment
    
    def _extract_task_keywords(self, task_context: Dict[str, Any]) -> List[str]:
        """Extract keywords from task context."""
        keywords = []
        
        # Extract from task description
        description = task_context.get("description", "")
        if description:
            # Simple keyword extraction
            words = re.findall(r'\b\w+\b', description.lower())
            keywords.extend([w for w in words if len(w) > 3])
        
        # Extract from task type
        task_type = task_context.get("task_type", "")
        if task_type:
            keywords.extend(task_type.lower().split())
        
        return list(set(keywords))


class AuthenticityCheckingStrategy(ValidationStrategy):
    """Detects hallucinated or fabricated research content."""
    
    def validate(self, output: Any, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check for hallucinated or fabricated content.
        
        Checks:
        - Citations exist and are plausible
        - Experimental results are reasonable
        - Claims are supported by evidence
        - No fabricated data or results
        """
        authenticity_flags = []
        
        # Check citations
        if self._contains_suspicious_citations(output):
            authenticity_flags.append("suspicious_citations")
        
        # Check experimental results
        if self._contains_implausible_results(output):
            authenticity_flags.append("implausible_results")
        
        # Check evidence support
        if self._lacks_evidence_support(output):
            authenticity_flags.append("unsupported_claims")
        
        # Check for fabricated specifics
        if self._contains_fabricated_specifics(output):
            authenticity_flags.append("fabricated_specifics")
        
        result = {
            "strategy": "authenticity_checking",
            "authenticity_score": self._calculate_authenticity_score(authenticity_flags),
            "authenticity_flags": authenticity_flags,
            "fabrication_risk": self._assess_fabrication_risk(output),
            "timestamp": datetime.now().isoformat()
        }
        
        self.logger.debug(f"Authenticity check result: {result}")
        return result
    
    def _contains_suspicious_citations(self, output: Any) -> bool:
        """Check for suspicious or fabricated citations."""
        output_text = str(output)
        
        # Look for citation patterns
        citation_patterns = [
            r'\([A-Z][a-z]+\s+et\s+al\.,?\s+\d{4}\)',  # (Author et al., 2023)
            r'\[[^\]]+\d{4}[^\]]*\]',  # [Author 2023]
            r'doi:\s*10\.\d+',  # DOI patterns
            r'arxiv:\d{4}\.\d{4,5}',  # arXiv patterns
        ]
        
        citations_found = 0
        for pattern in citation_patterns:
            citations_found += len(re.findall(pattern, output_text))
        
        # Suspicious if too many citations without context
        if citations_found > 10:
            return True
        
        # Check for obviously fake citations
        fake_patterns = [
            r'Smith\s+et\s+al\.,?\s+\d{4}',  # Generic "Smith et al."
            r'Johnson\s+et\s+al\.,?\s+\d{4}',  # Generic "Johnson et al."
            r'Brown\s+et\s+al\.,?\s+\d{4}',  # Generic "Brown et al."
        ]
        
        for pattern in fake_patterns:
            if re.search(pattern, output_text):
                return True
        
        return False
    
    def _contains_implausible_results(self, output: Any) -> bool:
        """Check for implausible experimental results."""
        output_text = str(output).lower()
        
        # Look for perfect or impossible results
        perfect_patterns = [
            r'100%\s+accuracy',
            r'100%\s+precision',
            r'100%\s+recall',
            r'zero\s+error',
            r'perfect\s+performance',
            r'flawless\s+results',
        ]
        
        for pattern in perfect_patterns:
            if re.search(pattern, output_text):
                return True
        
        # Look for impossible numerical results
        if re.search(r'\d{3,}%', output_text):  # Percentages over 100%
            return True
        
        return False
    
    def _lacks_evidence_support(self, output: Any) -> bool:
        """Check if claims lack evidence support."""
        output_text = str(output).lower()
        
        # Look for unsupported claim indicators
        unsupported_patterns = [
            r'clearly\s+shows',
            r'obviously\s+demonstrates',
            r'definitely\s+proves',
            r'undoubtedly\s+indicates',
            r'without\s+question',
            r'absolutely\s+certain',
        ]
        
        unsupported_count = 0
        for pattern in unsupported_patterns:
            unsupported_count += len(re.findall(pattern, output_text))
        
        # Check for evidence indicators
        evidence_patterns = [
            r'according\s+to',
            r'based\s+on',
            r'results\s+show',
            r'data\s+indicates',
            r'study\s+found',
            r'research\s+demonstrates',
        ]
        
        evidence_count = 0
        for pattern in evidence_patterns:
            evidence_count += len(re.findall(pattern, output_text))
        
        # Lack of evidence if many unsupported claims with little evidence
        return unsupported_count > 2 and evidence_count < unsupported_count
    
    def _contains_fabricated_specifics(self, output: Any) -> bool:
        """Check for fabricated specific details."""
        output_text = str(output)
        
        # Look for suspiciously specific numbers without context
        specific_patterns = [
            r'\d+\.\d{4,}\s*%',  # Very precise percentages
            r'\d+\.\d{6,}',  # Very precise numbers
            r'exactly\s+\d+\.\d+',  # "Exactly" with decimals
        ]
        
        for pattern in specific_patterns:
            if re.search(pattern, output_text):
                return True
        
        return False
    
    def _calculate_authenticity_score(self, authenticity_flags: List[str]) -> float:
        """Calculate authenticity score based on flags."""
        if not authenticity_flags:
            return 1.0
        
        # Each flag reduces authenticity score
        score = 1.0 - (len(authenticity_flags) * 0.2)
        return max(score, 0.0)
    
    def _assess_fabrication_risk(self, output: Any) -> str:
        """Assess overall fabrication risk level."""
        output_text = str(output)
        
        risk_indicators = 0
        
        # Check various risk factors
        if len(re.findall(r'\d+\.\d{3,}', output_text)) > 3:
            risk_indicators += 1
        
        if len(re.findall(r'[A-Z][a-z]+\s+et\s+al', output_text)) > 5:
            risk_indicators += 1
        
        if len(re.findall(r'clearly|obviously|definitely|undoubtedly', output_text)) > 3:
            risk_indicators += 1
        
        if risk_indicators >= 3:
            return "high"
        elif risk_indicators >= 2:
            return "medium"
        elif risk_indicators >= 1:
            return "low"
        else:
            return "minimal"


class HallucinationDetectionStrategy(ValidationStrategy):
    """Specialized detection of common LLM hallucination patterns."""
    
    def __init__(self, debug: bool = False):
        super().__init__(debug)
        
        # Common hallucination patterns
        self.hallucination_patterns = [
            r"definitely\s+proves",
            r"clearly\s+demonstrates",
            r"without\s+doubt",
            r"absolutely\s+certain",
            r"it\s+is\s+well-known\s+that",
            r"obviously",
            r"undeniably",
            r"unquestionably",
            r"indisputable",
            r"beyond\s+question"
        ]
    
    def validate(self, output: Any, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect common hallucination patterns.
        
        Patterns:
        - Overconfident claims without evidence
        - Specific numbers or results without source
        - Non-existent papers or authors
        - Contradictory statements within output
        """
        hallucination_indicators = []
        
        # Pattern detection
        if self._contains_unsourced_specifics(output):
            hallucination_indicators.append("unsourced_specifics")
        
        if self._contains_overconfident_claims(output):
            hallucination_indicators.append("overconfident_claims")
        
        if self._contains_contradictions(output):
            hallucination_indicators.append("internal_contradictions")
        
        if self._contains_impossible_claims(output):
            hallucination_indicators.append("impossible_claims")
        
        result = {
            "strategy": "hallucination_detection",
            "hallucination_risk": self._calculate_hallucination_risk(hallucination_indicators),
            "hallucination_indicators": hallucination_indicators,
            "confidence_assessment": self._assess_confidence_levels(output),
            "timestamp": datetime.now().isoformat()
        }
        
        self.logger.debug(f"Hallucination detection result: {result}")
        return result
    
    def _contains_unsourced_specifics(self, output: Any) -> bool:
        """Check for specific claims without sources."""
        output_text = str(output)
        
        # Look for very specific numbers without attribution
        specific_patterns = [
            r'\d+\.\d{4,}%\s+improvement',
            r'\d+\.\d{4,}%\s+accuracy',
            r'increased\s+by\s+\d+\.\d{3,}%',
            r'reduced\s+by\s+\d+\.\d{3,}%',
        ]
        
        for pattern in specific_patterns:
            matches = re.findall(pattern, output_text)
            if matches:
                # Check if there's a source nearby
                for match in matches:
                    match_pos = output_text.find(match)
                    context = output_text[max(0, match_pos-100):match_pos+100]
                    if not re.search(r'according\s+to|based\s+on|from\s+|reported\s+in', context):
                        return True
        
        return False
    
    def _contains_overconfident_claims(self, output: Any) -> bool:
        """Check for overconfident claims using linguistic patterns."""
        output_text = str(output).lower()
        
        overconfident_count = 0
        for pattern in self.hallucination_patterns:
            matches = re.findall(pattern, output_text)
            overconfident_count += len(matches)
        
        # Too many overconfident claims suggest hallucination
        return overconfident_count > 3
    
    def _contains_contradictions(self, output: Any) -> bool:
        """Check for internal contradictions."""
        output_text = str(output).lower()
        
        # Simple contradiction detection
        sentences = output_text.split('.')
        
        # Look for contradictory statements
        for i, sentence1 in enumerate(sentences):
            for sentence2 in sentences[i+1:]:
                # Check for negation patterns
                if 'not' in sentence1 and 'not' not in sentence2:
                    # Look for similar content
                    words1 = set(sentence1.split())
                    words2 = set(sentence2.split())
                    overlap = words1 & words2
                    if len(overlap) > 3:  # Significant overlap
                        return True
        
        return False
    
    def _contains_impossible_claims(self, output: Any) -> bool:
        """Check for impossible or unrealistic claims."""
        output_text = str(output).lower()
        
        impossible_patterns = [
            r'100%\s+accuracy',
            r'zero\s+error\s+rate',
            r'infinite\s+improvement',
            r'perfect\s+solution',
            r'eliminates\s+all\s+problems',
            r'solves\s+everything',
        ]
        
        for pattern in impossible_patterns:
            if re.search(pattern, output_text):
                return True
        
        return False
    
    def _calculate_hallucination_risk(self, hallucination_indicators: List[str]) -> str:
        """Calculate hallucination risk level."""
        if not hallucination_indicators:
            return "minimal"
        elif len(hallucination_indicators) == 1:
            return "low"
        elif len(hallucination_indicators) == 2:
            return "medium"
        else:
            return "high"
    
    def _assess_confidence_levels(self, output: Any) -> Dict[str, Any]:
        """Assess confidence levels in the output."""
        output_text = str(output).lower()
        
        # Count confidence indicators
        high_confidence = sum(1 for pattern in self.hallucination_patterns 
                            if re.search(pattern, output_text))
        
        moderate_confidence = len(re.findall(r'likely|probably|appears|seems|suggests', output_text))
        
        low_confidence = len(re.findall(r'maybe|perhaps|possibly|might|could', output_text))
        
        return {
            "high_confidence_claims": high_confidence,
            "moderate_confidence_claims": moderate_confidence,
            "low_confidence_claims": low_confidence,
            "confidence_balance": "balanced" if moderate_confidence > high_confidence else "overconfident"
        }