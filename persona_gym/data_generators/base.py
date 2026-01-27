"""
Base class for data generators.

All data generators must inherit from BaseDataGenerator and implement
the generate() method to produce DataGenerationOutput.
"""

from abc import ABC, abstractmethod
from typing import Optional

from persona_gym.schemas import DataGenerationOutput


class BaseDataGenerator(ABC):
    """Abstract base class for conversation data generators.

    All data generators must produce DataGenerationOutput, ensuring
    consistent input to the task generation stage regardless of the
    generation strategy used.

    Subclasses must implement:
        - generate(): Create conversation data and return DataGenerationOutput

    Attributes:
        topic: The conversation topic (e.g., "travel", "cooking", "therapy")
        output_dir: Directory to save generated files (optional)
    """

    def __init__(
        self,
        topic: str,
        output_dir: Optional[str] = None,
    ):
        """Initialize the generator.

        Args:
            topic: Conversation topic domain
            output_dir: Directory for output files (optional, generator may work in-memory)
        """
        self.topic = topic
        self.output_dir = output_dir

    @abstractmethod
    def generate(self) -> DataGenerationOutput:
        """Generate conversation data with preferences.

        Returns:
            DataGenerationOutput containing:
                - conversation: List of {role, content} messages
                - preferences: List of extracted PreferenceItems
                - metadata: Generation metadata

        Raises:
            GenerationError: If generation fails
        """
        pass

    @property
    def name(self) -> str:
        """Return the name of this generator."""
        return self.__class__.__name__


class GenerationError(Exception):
    """Raised when data generation fails."""

    pass
