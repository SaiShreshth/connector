"""
Intent Classification Models

This module defines all dataclasses and structures used for representing
intent definitions, classification results, entities, and metrics.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from .enums import IntentCategory, ActionCode, IntentPriority, EntityType


# ------------------------------------------------------------------------------
# INTENT DEFINITIONS
# ------------------------------------------------------------------------------

@dataclass
class IntentDefinition:
    """Defines a single intent with all metadata and configuration."""

    category: IntentCategory
    action_code: ActionCode
    description: str
    example_phrases: List[str]
    required_entities: List[EntityType]
    optional_entities: List[EntityType]
    confidence_threshold: float = 0.7
    priority: IntentPriority = IntentPriority.MEDIUM

    def __post_init__(self):
        """Validation logic for the intent definition."""
        if not self.example_phrases:
            raise ValueError("Intent definition must include at least one example phrase.")

        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError("Confidence threshold must be between 0.0 and 1.0.")


# ------------------------------------------------------------------------------
# INTENT RESULT
# ------------------------------------------------------------------------------

@dataclass
class IntentResult:
    """Represents the output of the intent classification process."""

    intent_name: str
    action_code: ActionCode
    confidence_score: float
    extracted_entities: Dict[str, Any]
    processing_time_ms: float
    metadata: Optional[Dict[str, Any]] = None

    def is_confident(self, threshold: float = 0.7) -> bool:
        """Return True if the classification confidence meets or exceeds the threshold."""
        return self.confidence_score >= threshold

    def get_required_entities(self) -> List[str]:
        """Return the names of all extracted entities."""
        return list(self.extracted_entities.keys())


# ------------------------------------------------------------------------------
# ENTITY EXTRACTION RESULT
# ------------------------------------------------------------------------------

@dataclass
class EntityExtraction:
    """Represents the result of a single extracted entity from user input."""

    entity_type: EntityType
    value: Any
    confidence: float
    start_position: int
    end_position: int
    original_text: str

    def __post_init__(self):
        """Validation for entity extraction."""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Entity confidence must be between 0.0 and 1.0.")

        if self.start_position < 0 or self.end_position < self.start_position:
            raise ValueError("Invalid position range for entity extraction.")


# ------------------------------------------------------------------------------
# CLASSIFICATION REQUEST
# ------------------------------------------------------------------------------

@dataclass
class ClassificationRequest:
    """Input request for intent classification."""

    user_input: str
    context: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: Optional[str] = None

    def __post_init__(self):
        """Validate classification request data."""
        if not self.user_input or not self.user_input.strip():
            raise ValueError("User input cannot be empty.")


# ------------------------------------------------------------------------------
# CLASSIFICATION RESPONSE
# ------------------------------------------------------------------------------

@dataclass
class ClassificationResponse:
    """Output structure returned after intent classification."""

    primary_intent: IntentResult
    alternative_intents: List[IntentResult]
    entities: List[EntityExtraction]
    requires_clarification: bool
    clarification_message: Optional[str] = None
    processing_metadata: Optional[Dict[str, Any]] = None

    def get_best_intent(self, threshold: float = 0.7) -> Optional[IntentResult]:
        """Return the best intent that meets the confidence threshold."""
        if self.primary_intent.is_confident(threshold):
            return self.primary_intent

        for alt_intent in self.alternative_intents:
            if alt_intent.is_confident(threshold):
                return alt_intent

        return None

    def get_all_confident_intents(self, threshold: float = 0.7) -> List[IntentResult]:
        """Return all intents whose confidence meets or exceeds the threshold."""
        confident_intents = []

        if self.primary_intent.is_confident(threshold):
            confident_intents.append(self.primary_intent)

        for alt_intent in self.alternative_intents:
            if alt_intent.is_confident(threshold):
                confident_intents.append(alt_intent)

        return confident_intents


# ------------------------------------------------------------------------------
# INTENT METRICS
# ------------------------------------------------------------------------------

@dataclass
class IntentMetrics:
    """Performance metrics and monitoring statistics for intent classification."""

    total_requests: int
    successful_classifications: int
    average_confidence: float
    average_processing_time_ms: float
    most_common_intents: Dict[str, int]
    error_rate: float

    @property
    def success_rate(self) -> float:
        """Compute and return the success rate of classification."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_classifications / self.total_requests

# ------------------------------------------------------------------------------
# INTENT EXAMPLES
# ------------------------------------------------------------------------------

@dataclass
class IntentExample:
    """Represents a single example phrase for an intent, including optional entity annotations."""

    phrase: str
    entities: Optional[Dict[str, Any]] = None  # Mapping from entity name to value
    metadata: Optional[Dict[str, Any]] = None  # Optional metadata (e.g., source, timestamp)

    def __post_init__(self):
        """Validate the intent example data."""
        if not self.phrase or not self.phrase.strip():
            raise ValueError("Intent example phrase cannot be empty.")
        if self.entities is not None and not isinstance(self.entities, dict):
            raise TypeError("Entities must be a dictionary mapping entity names to values.")
