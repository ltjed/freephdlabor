"""
Supervision Manager for the Smolagents Research System.

This module manages hierarchical supervision to prevent hallucination and ensure
quality as specified in the Full Smolagents Agentization Plan.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum

from .validation_strategies import (
    ValidationStrategy,
    OutputValidationStrategy,
    AuthenticityCheckingStrategy,
    HallucinationDetectionStrategy
)


class SupervisionLevel(Enum):
    """Enumeration of supervision levels."""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    EXHAUSTIVE = "exhaustive"


class AgentSupervisionManager:
    """
    Manages hierarchical supervision to prevent hallucination and ensure quality.
    Each agent is supervised by the agent one level above.
    """
    
    def __init__(self, supervision_level: str = "standard", debug: bool = False):
        """
        Initialize AgentSupervisionManager.
        
        Args:
            supervision_level: Level of supervision to apply
            debug: Enable debug logging
        """
        self.supervision_level = SupervisionLevel(supervision_level)
        self.debug = debug
        
        # Setup logging
        logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Define supervision hierarchy
        self.supervision_hierarchy = {
            "meta_research_agent": None,  # Currently unsupervised
            "enhanced_ideation_agent": "meta_research_agent",
            "enhanced_experiment_agent": "meta_research_agent",
            "publication_agent": "meta_research_agent",
            "evaluation_agent": "meta_research_agent",
            "ideation_agent": "meta_research_agent",  # Legacy compatibility
            "manager_agent": None,  # Currently unsupervised (legacy)
        }
        
        # Initialize supervision strategies
        self.supervision_strategies = self._initialize_strategies()
        
        # Track supervision history
        self.supervision_history = []
    
    def _initialize_strategies(self) -> Dict[str, ValidationStrategy]:
        """Initialize supervision strategies based on level."""
        strategies = {
            "output_validation": OutputValidationStrategy(debug=self.debug),
            "authenticity_checking": AuthenticityCheckingStrategy(debug=self.debug),
            "hallucination_detection": HallucinationDetectionStrategy(debug=self.debug)
        }
        
        return strategies
    
    def supervise_agent_output(self, agent_id: str, output: Any, 
                             task_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Supervise agent output using multiple validation strategies.
        
        Args:
            agent_id: ID of the agent producing output
            output: The output to validate
            task_context: Context of the assigned task
            
        Returns:
            validated_output: Validated and potentially corrected output
            supervision_report: Report of supervision actions taken
        """
        self.logger.info(f"Supervising output from agent: {agent_id}")
        
        supervisor_id = self.supervision_hierarchy.get(agent_id)
        
        if supervisor_id is None:
            # No supervision for top-level agent
            return output, {
                "supervision_status": "unsupervised",
                "agent_id": agent_id,
                "timestamp": datetime.now().isoformat()
            }
        
        # Apply supervision strategies based on level
        validation_results = {}
        strategies_to_apply = self._select_strategies_by_level()
        
        for strategy_name in strategies_to_apply:
            if strategy_name in self.supervision_strategies:
                strategy = self.supervision_strategies[strategy_name]
                result = strategy.validate(output, task_context)
                validation_results[strategy_name] = result
        
        # Generate supervision report
        supervision_report = {
            "agent_id": agent_id,
            "supervisor_id": supervisor_id,
            "supervision_level": self.supervision_level.value,
            "validation_results": validation_results,
            "supervision_actions": self._determine_actions(validation_results),
            "final_status": self._determine_final_status(validation_results),
            "timestamp": datetime.now().isoformat()
        }
        
        # Store supervision history
        self.supervision_history.append(supervision_report)
        
        # Apply corrections if needed
        if supervision_report["final_status"] != "approved":
            corrected_output = self._apply_corrections(output, supervision_report)
            self.logger.warning(f"Applied corrections to output from {agent_id}")
            return corrected_output, supervision_report
        
        self.logger.info(f"Output from {agent_id} approved without corrections")
        return output, supervision_report
    
    def _select_strategies_by_level(self) -> List[str]:
        """Select validation strategies based on supervision level."""
        if self.supervision_level == SupervisionLevel.BASIC:
            return ["output_validation"]
        elif self.supervision_level == SupervisionLevel.STANDARD:
            return ["output_validation", "hallucination_detection"]
        elif self.supervision_level == SupervisionLevel.COMPREHENSIVE:
            return ["output_validation", "authenticity_checking", "hallucination_detection"]
        elif self.supervision_level == SupervisionLevel.EXHAUSTIVE:
            return ["output_validation", "authenticity_checking", "hallucination_detection"]
        else:
            return ["output_validation"]
    
    def _determine_actions(self, validation_results: Dict[str, Any]) -> List[str]:
        """Determine supervision actions based on validation results."""
        actions = []
        
        # Check output validation
        if "output_validation" in validation_results:
            validation_score = validation_results["output_validation"]["validation_score"]
            if validation_score < 0.7:
                actions.append("request_output_improvement")
            if validation_results["output_validation"]["missing_deliverables"]:
                actions.append("request_missing_deliverables")
        
        # Check authenticity
        if "authenticity_checking" in validation_results:
            authenticity_score = validation_results["authenticity_checking"]["authenticity_score"]
            if authenticity_score < 0.8:
                actions.append("verify_authenticity")
            if validation_results["authenticity_checking"]["fabrication_risk"] in ["high", "medium"]:
                actions.append("flag_fabrication_risk")
        
        # Check hallucination
        if "hallucination_detection" in validation_results:
            hallucination_risk = validation_results["hallucination_detection"]["hallucination_risk"]
            if hallucination_risk in ["high", "medium"]:
                actions.append("flag_hallucination_risk")
        
        return actions
    
    def _determine_final_status(self, validation_results: Dict[str, Any]) -> str:
        """Determine final supervision status."""
        # Check critical failures
        for strategy_name, result in validation_results.items():
            if strategy_name == "output_validation":
                if result["validation_score"] < 0.5:
                    return "rejected"
            elif strategy_name == "authenticity_checking":
                if result["authenticity_score"] < 0.6:
                    return "rejected"
                if result["fabrication_risk"] == "high":
                    return "rejected"
            elif strategy_name == "hallucination_detection":
                if result["hallucination_risk"] == "high":
                    return "rejected"
        
        # Check for warnings
        warning_count = 0
        for strategy_name, result in validation_results.items():
            if strategy_name == "output_validation":
                if result["validation_score"] < 0.8:
                    warning_count += 1
            elif strategy_name == "authenticity_checking":
                if result["authenticity_score"] < 0.9:
                    warning_count += 1
            elif strategy_name == "hallucination_detection":
                if result["hallucination_risk"] in ["medium", "low"]:
                    warning_count += 1
        
        if warning_count >= 2:
            return "warning"
        elif warning_count >= 1:
            return "approved_with_warnings"
        else:
            return "approved"
    
    def _apply_corrections(self, output: Any, supervision_report: Dict[str, Any]) -> Any:
        """Apply corrections to output based on supervision report."""
        # This is a simplified correction system
        # In a full implementation, this would use an LLM to apply corrections
        
        actions = supervision_report["supervision_actions"]
        corrected_output = str(output)
        
        # Apply basic corrections
        if "flag_fabrication_risk" in actions:
            corrected_output += "\n\n[SUPERVISION NOTE: Please verify all cited sources and numerical claims]"
        
        if "flag_hallucination_risk" in actions:
            corrected_output += "\n\n[SUPERVISION NOTE: Please provide evidence for confident claims]"
        
        if "request_missing_deliverables" in actions:
            validation_result = supervision_report["validation_results"].get("output_validation", {})
            missing = validation_result.get("missing_deliverables", [])
            if missing:
                corrected_output += f"\n\n[SUPERVISION NOTE: Please provide: {', '.join(missing)}]"
        
        return corrected_output
    
    def get_supervision_statistics(self) -> Dict[str, Any]:
        """Get statistics about supervision activities."""
        if not self.supervision_history:
            return {
                "total_supervisions": 0,
                "approval_rate": 0.0,
                "rejection_rate": 0.0,
                "warning_rate": 0.0
            }
        
        total = len(self.supervision_history)
        status_counts = {}
        
        for record in self.supervision_history:
            status = record["final_status"]
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "total_supervisions": total,
            "approval_rate": status_counts.get("approved", 0) / total,
            "rejection_rate": status_counts.get("rejected", 0) / total,
            "warning_rate": (status_counts.get("warning", 0) + 
                           status_counts.get("approved_with_warnings", 0)) / total,
            "status_breakdown": status_counts,
            "supervision_level": self.supervision_level.value
        }
    
    def get_agent_supervision_history(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get supervision history for a specific agent."""
        return [record for record in self.supervision_history 
                if record["agent_id"] == agent_id]
    
    def clear_supervision_history(self):
        """Clear supervision history."""
        self.supervision_history = []
        self.logger.info("Supervision history cleared")
    
    def update_supervision_level(self, new_level: str):
        """Update supervision level."""
        self.supervision_level = SupervisionLevel(new_level)
        self.logger.info(f"Supervision level updated to: {new_level}")
    
    def add_agent_to_hierarchy(self, agent_id: str, supervisor_id: str = None):
        """Add an agent to the supervision hierarchy."""
        self.supervision_hierarchy[agent_id] = supervisor_id
        self.logger.info(f"Added agent {agent_id} to supervision hierarchy with supervisor {supervisor_id}")
    
    def remove_agent_from_hierarchy(self, agent_id: str):
        """Remove an agent from the supervision hierarchy."""
        if agent_id in self.supervision_hierarchy:
            del self.supervision_hierarchy[agent_id]
            self.logger.info(f"Removed agent {agent_id} from supervision hierarchy")