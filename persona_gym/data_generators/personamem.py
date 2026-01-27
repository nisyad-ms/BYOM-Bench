"""
PersonaMem-based data generator.

This generator uses the PersonaMem pipeline to create synthetic conversations
with rich user preferences that evolve over time.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Optional

import yaml

from persona_gym.data_generators.base import BaseDataGenerator, GenerationError
from persona_gym.schemas import (
    DataGenerationMetadata,
    DataGenerationOutput,
    PreferenceItem,
)

logger = logging.getLogger(__name__)


class PersonaMemGenerator(BaseDataGenerator):
    """Generate conversation data using the PersonaMem pipeline.

    PersonaMem creates synthetic conversations with:
    - Rich persona histories that evolve over time
    - Explicit preference annotations (likes, dislikes, updates)
    - Multiple time periods (init, week, month, year)

    This is the default generator for PersonaGym evaluation.

    Attributes:
        topic: Conversation topic (e.g., "travel", "cooking", "therapy")
        persona_idx: Index of persona to use from persona file
        config_path: Path to config.yaml
        output_dir: Directory for output files
    """

    def __init__(
        self,
        topic: str,
        persona_idx: int = 0,
        config_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        debug_mode: bool = False,
        quick_mode: bool = False,
    ):
        """Initialize PersonaMem generator.

        Args:
            topic: Conversation topic domain
            persona_idx: Index of persona to use (default: 0)
            config_path: Path to config.yaml (default: package config)
            output_dir: Output directory (default: from config)
            debug_mode: Enable debug mode for faster generation
            quick_mode: Enable quick mode (fewer conversation turns)
        """
        super().__init__(topic=topic, output_dir=output_dir)
        self.persona_idx = persona_idx
        self.debug_mode = debug_mode
        self.quick_mode = quick_mode

        # Load config
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "config.yaml"
            )
        self.config_path = config_path
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from config.yaml."""
        with open(self.config_path, "r") as f:
            self.args = yaml.safe_load(f)

        # Override topic
        self.args["datasets"]["topics"] = [self.topic]

        # Override persona index
        self.args["inference"]["start_persona_idx"] = self.persona_idx

        # Set modes
        self.args["inference"]["debug_mode"] = self.debug_mode
        self.args["inference"]["quick_mode"] = self.quick_mode

        # Override output directory if specified
        if self.output_dir:
            self.args["inference"]["output_dir"] = self.output_dir

    def generate(self) -> DataGenerationOutput:
        """Generate conversation data using PersonaMem pipeline.

        Returns:
            DataGenerationOutput with conversation, preferences, and metadata

        Raises:
            GenerationError: If generation fails
        """
        # Import here to avoid circular imports and heavy loading at import time
        from persona_gym.data_generation import (
            AzureQueryLLM,
            extract_preferences_for_tod,
            postprocess_conversation_for_tod,
        )
        from persona_gym.personamemv1_core import utils
        from persona_gym.personamemv1_core.prepare_data import (
            prepare_data_on_other_topics,
            prepare_data_on_writing_topic,
            prepare_persona,
            prepare_topics,
        )

        logger.info(f"Starting PersonaMem generation for topic: {self.topic}")

        try:
            # Load personas
            with open(self.args["datasets"]["persona_file"], "r") as f:
                all_personas = f.readlines()

            # Initialize LLM
            llm = AzureQueryLLM(self.args)

            # Prepare persona
            (
                persona,
                expanded_persona,
                start_time,
                init_general_personal_history,
                general_personal_history_next_week,
                general_personal_history_next_month,
                general_personal_history_next_year,
            ) = prepare_persona(llm, self.persona_idx, all_personas, self.args)

            # Prepare topic resources
            all_topics = [self.topic]
            source_dir, all_source_files = prepare_topics(
                0, all_topics, self.topic, self.args
            )

            # Load source data
            source_data = (
                utils.load_one_source_data(source_dir, all_source_files, self.topic)
                if all_source_files
                else None
            )

            # Initialize data collector
            data_collector: dict[str, Any] = {
                "Original Persona": persona,
                "Expanded Persona": expanded_persona,
                "Topic": self.topic,
            }

            # Generate conversation data
            if self.topic in ["writing", "email", "coding"]:
                llm.create_a_thread(step="writing")
                prepare_data_on_writing_topic(
                    llm,
                    self.topic,
                    persona,
                    source_data,
                    "",  # output_file_path not needed for in-memory
                    self.args,
                    data_collector=data_collector,
                )
            else:
                llm.create_a_thread(step="conversation")
                prepare_data_on_other_topics(
                    llm,
                    expanded_persona,
                    source_data,
                    source_dir,
                    self.topic,
                    0,  # idx_topic
                    start_time,
                    "",  # output_file_path not needed for in-memory
                    init_general_personal_history,
                    general_personal_history_next_week,
                    general_personal_history_next_month,
                    general_personal_history_next_year,
                    self.args,
                    data_collector=data_collector,
                )

            # Extract preferences
            preferences = extract_preferences_for_tod(data_collector, self.topic)

            # Post-process conversation to clean format
            conversation = postprocess_conversation_for_tod(data_collector, self.topic)

            # Build metadata
            metadata = DataGenerationMetadata(
                topic=self.topic,
                persona_id=f"persona_{self.persona_idx}",
                timestamp=datetime.now().isoformat(),
                source_file="",  # Generated in memory
            )

            logger.info(
                f"Generation complete: {len(conversation)} turns, {len(preferences)} preferences"
            )

            return DataGenerationOutput(
                conversation=conversation,
                preferences=preferences,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"PersonaMem generation failed: {e}")
            raise GenerationError(f"PersonaMem generation failed: {e}") from e

    def generate_and_save(
        self,
        output_path: Optional[str] = None,
    ) -> tuple[DataGenerationOutput, str, str]:
        """Generate data and save to files.

        This method generates data and also saves to conversation.json
        and artifacts.json files for debugging/analysis.

        Args:
            output_path: Base path for output files (optional)

        Returns:
            Tuple of (DataGenerationOutput, conversation_path, artifacts_path)
        """
        from persona_gym.data_generation import (
            AzureQueryLLM,
            extract_preferences_for_tod,
            save_tod_ready_data,
        )
        from persona_gym.personamemv1_core import utils
        from persona_gym.personamemv1_core.prepare_data import (
            prepare_data_on_other_topics,
            prepare_data_on_writing_topic,
            prepare_persona,
            prepare_topics,
        )

        logger.info(f"Starting PersonaMem generation with file output for topic: {self.topic}")

        # Determine output path
        if output_path is None:
            mode_suffix = "_debug" if self.debug_mode else ("_quick" if self.quick_mode else "")
            output_dir = self.output_dir or self.args["inference"]["output_dir"]
            output_path = os.path.join(
                output_dir,
                self.topic,
                f"personamem_{self.topic}_persona{self.persona_idx}{mode_suffix}.json",
            )

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Load personas
        with open(self.args["datasets"]["persona_file"], "r") as f:
            all_personas = f.readlines()

        # Initialize LLM
        llm = AzureQueryLLM(self.args)

        # Prepare persona
        (
            persona,
            expanded_persona,
            start_time,
            init_general_personal_history,
            general_personal_history_next_week,
            general_personal_history_next_month,
            general_personal_history_next_year,
        ) = prepare_persona(llm, self.persona_idx, all_personas, self.args)

        # Prepare topic resources
        all_topics = [self.topic]
        source_dir, all_source_files = prepare_topics(
            0, all_topics, self.topic, self.args
        )

        # Load source data
        source_data = (
            utils.load_one_source_data(source_dir, all_source_files, self.topic)
            if all_source_files
            else None
        )

        # Initialize data collector
        data_collector: dict[str, Any] = {
            "Original Persona": persona,
            "Expanded Persona": expanded_persona,
            "Topic": self.topic,
        }

        # Generate conversation data
        if self.topic in ["writing", "email", "coding"]:
            llm.create_a_thread(step="writing")
            prepare_data_on_writing_topic(
                llm,
                self.topic,
                persona,
                source_data,
                output_path,
                self.args,
                data_collector=data_collector,
            )
        else:
            llm.create_a_thread(step="conversation")
            prepare_data_on_other_topics(
                llm,
                expanded_persona,
                source_data,
                source_dir,
                self.topic,
                0,
                start_time,
                output_path,
                init_general_personal_history,
                general_personal_history_next_week,
                general_personal_history_next_month,
                general_personal_history_next_year,
                self.args,
                data_collector=data_collector,
            )

        # Extract preferences and save files
        preferences = extract_preferences_for_tod(data_collector, self.topic)
        conversation_path, artifacts_path = save_tod_ready_data(
            data_collector, output_path, self.topic
        )

        # Build output
        metadata = DataGenerationMetadata(
            topic=self.topic,
            persona_id=f"persona_{self.persona_idx}",
            timestamp=datetime.now().isoformat(),
            source_file=conversation_path,
        )

        # Load conversation from saved file
        with open(conversation_path, "r") as f:
            conversation = json.load(f)

        output = DataGenerationOutput(
            conversation=conversation,
            preferences=preferences,
            metadata=metadata,
        )

        logger.info(f"Saved conversation to: {conversation_path}")
        logger.info(f"Saved artifacts to: {artifacts_path}")

        return output, conversation_path, artifacts_path

    @staticmethod
    def from_files(
        conversation_path: str,
        artifacts_path: Optional[str] = None,
    ) -> DataGenerationOutput:
        """Load DataGenerationOutput from previously generated files.

        This allows reusing previously generated data without regenerating.

        Args:
            conversation_path: Path to conversation.json
            artifacts_path: Path to artifacts.json (optional, for metadata)

        Returns:
            DataGenerationOutput loaded from files
        """
        # Load conversation
        with open(conversation_path, "r") as f:
            conversation = json.load(f)

        # Load metadata and preferences from artifacts if available
        preferences: list[PreferenceItem] = []
        metadata = DataGenerationMetadata(topic="", source_file=conversation_path)

        if artifacts_path and os.path.exists(artifacts_path):
            with open(artifacts_path, "r") as f:
                artifacts = json.load(f)

            # Extract metadata
            meta = artifacts.get("metadata", {})
            metadata = DataGenerationMetadata(
                topic=meta.get("topic", ""),
                persona_id="",
                timestamp=meta.get("timestamp", ""),
                source_file=conversation_path,
            )

            # Extract preferences
            prefs_data = artifacts.get("preferences", {}).get("extracted", [])
            for p in prefs_data:
                preferences.append(
                    PreferenceItem(
                        fact=p.get("fact", ""),
                        preference_type=p.get("type", p.get("preference_type", "current")),
                        source_date=p.get("source_date", ""),
                        topic=p.get("topic", ""),
                        old_value=p.get("old_value"),
                        reason_of_change=p.get("reason_of_change"),
                    )
                )

        return DataGenerationOutput(
            conversation=conversation,
            preferences=preferences,
            metadata=metadata,
        )
