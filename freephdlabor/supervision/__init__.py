"""
Supervision module for the Smolagents Research System.

This module provides hierarchical supervision to prevent hallucination and
ensure research quality through multiple validation strategies.
"""

from .supervision_manager import AgentSupervisionManager, SupervisionLevel
from .validation_strategies import (
    ValidationStrategy,
    OutputValidationStrategy,
    AuthenticityCheckingStrategy,
    HallucinationDetectionStrategy
)

__all__ = [
    "AgentSupervisionManager",
    "SupervisionLevel",
    "ValidationStrategy",
    "OutputValidationStrategy",
    "AuthenticityCheckingStrategy",
    "HallucinationDetectionStrategy"
]