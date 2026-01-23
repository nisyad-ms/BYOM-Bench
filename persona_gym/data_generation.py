"""
Data Generation Script for PersonaMem

This script generates conversation datasets using the PersonaMem pipeline.
It uses AzureOpenAI client with DefaultAzureCredential for authentication.
Make sure you're logged in via `az login` before running.

Environment variables required:
    AZURE_OPENAI_ENDPOINT: Your Azure OpenAI endpoint URL
    AZURE_OPENAI_DEPLOYMENT: Your Azure OpenAI deployment name  
    AZURE_OPENAI_API_VERSION: API version (e.g., "2024-02-15-preview")

Usage:
    az login  # Authenticate first
    python -m persona_gym.data_generation --topic therapy --verbose
"""

# =============================================================================
# PATH SETUP - Must come before ANY imports from this project
# =============================================================================
import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv

from persona_gym.personamem_core import utils
from persona_gym.personamem_core.prepare_data import (
    prepare_data_on_other_topics,
    prepare_data_on_writing_topic,
    prepare_persona,
    prepare_topics,
)
from persona_gym.personamem_core.schemas import (
    GeneratedConversation,
)

load_dotenv()

# ============================================================================
# LOGGING SETUP
# ============================================================================


def setup_logging(log_file='data_generation.log'):
    """Configure logging to both file and console with detailed formatting."""
    # Create logger
    logger = logging.getLogger('PersonaMem')
    logger.setLevel(logging.DEBUG)

    # Clear any existing handlers
    logger.handlers = []

    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(funcName)-25s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )

    # File handler - detailed logging
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    # Console handler - info and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Initialize logger
# Log to project root logs/ folder (one level up from package)
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
logger = setup_logging(os.path.join(LOG_DIR, 'data_generation.log'))


# ============================================================================
# TOD PREFERENCE EXTRACTION (for Task-Oriented Dialogue evaluation)
# ============================================================================

@dataclass
class PreferenceItem:
    """A user preference extracted from conversation history for TOD evaluation."""
    fact: str  # e.g., "prefers window seats"
    preference_type: str  # "current", "updated", "static"
    source_date: str  # When the preference was expressed
    topic: str = ""  # Topic category
    old_value: Optional[str] = None  # For updated preferences
    reason_of_change: Optional[str] = None  # For updated preferences

    def to_dict(self) -> dict:
        result = {
            "fact": self.fact,
            "type": self.preference_type,
            "source_date": self.source_date,
            "topic": self.topic,
        }
        if self.old_value:
            result["old_value"] = self.old_value
        if self.reason_of_change:
            result["reason_of_change"] = self.reason_of_change
        return result


def extract_preferences_for_tod(data: Dict[str, Any], topic: str = "") -> List[PreferenceItem]:
    """
    Extract structured preferences from PersonaMem conversation data for TOD evaluation.

    Parses personal history blocks to find:
    - Static facts: [Fact] Likes, [Fact] Dislikes
    - Updated preferences: [Old Fact] -> [Updated Fact] with [Reasons of Change]

    Also parses Topic-Specific Hobbies which contains preference events.

    Args:
        data: The conversation JSON data from sample_data_generation.py
        topic: The conversation topic

    Returns:
        List of PreferenceItem objects for TOD evaluation checklist
    """
    preferences = []

    # First, try to extract from Topic-Specific Hobbies (contains JSON with preference events)
    hobbies_data = data.get('Topic-Specific Hobbies', '')
    if hobbies_data and isinstance(hobbies_data, str):
        try:
            # Find JSON block in the hobbies string
            json_match = re.search(r'\{[\s\S]*\}', hobbies_data)
            if json_match:
                hobbies_json = json.loads(json_match.group())
                for timestamp, event in hobbies_json.items():
                    if not isinstance(event, dict):
                        continue

                    source_date = timestamp

                    if '[Fact] Likes' in event:
                        fact = event['[Fact] Likes']
                        preferences.append(PreferenceItem(
                            fact=f"likes {fact.lower()}" if not fact.lower().startswith('like') else fact.lower(),
                            preference_type="current",
                            source_date=source_date,
                            topic=topic,
                        ))

                    if '[Fact] Dislikes' in event:
                        fact = event['[Fact] Dislikes']
                        preferences.append(PreferenceItem(
                            fact=f"dislikes {fact.lower()}" if not fact.lower().startswith('dislike') else fact.lower(),
                            preference_type="current",
                            source_date=source_date,
                            topic=topic,
                        ))

                    if '[Updated Fact] Likes' in event:
                        new_fact = event['[Updated Fact] Likes']
                        old_fact = event.get('[Old Fact] Dislikes', event.get('[Fact] Dislikes', ''))
                        reason = event.get('[Reasons of Change]', '')
                        preferences.append(PreferenceItem(
                            fact=f"now likes {new_fact.lower()}" if not new_fact.lower().startswith('like') else new_fact.lower(),
                            preference_type="updated",
                            source_date=source_date,
                            topic=topic,
                            old_value=f"disliked {old_fact.lower()}" if old_fact else None,
                            reason_of_change=reason if reason else None,
                        ))

                    if '[Updated Fact] Dislikes' in event:
                        new_fact = event['[Updated Fact] Dislikes']
                        old_fact = event.get('[Old Fact] Likes', event.get('[Fact] Likes', ''))
                        reason = event.get('[Reasons of Change]', '')
                        preferences.append(PreferenceItem(
                            fact=f"now dislikes {new_fact.lower()}" if not new_fact.lower().startswith('dislike') else new_fact.lower(),
                            preference_type="updated",
                            source_date=source_date,
                            topic=topic,
                            old_value=f"liked {old_fact.lower()}" if old_fact else None,
                            reason_of_change=reason if reason else None,
                        ))
        except (json.JSONDecodeError, Exception) as e:
            logger.debug(f"Could not parse Topic-Specific Hobbies JSON: {e}")

    # Personal history keys to check
    history_keys = [
        'Init Contextual Personal History',
        'Contextual Personal History Next Week',
        'Contextual Personal History Next Month',
        'Contextual Personal History Next Year',
        'Init General Personal History',
        'General Personal History Next Week',
        'General Personal History Next Month',
        'General Personal History Next Year',
    ]

    for history_key in history_keys:
        history_block = data.get(history_key)
        if not history_block:
            continue

        # History block can be a string (JSON) or already parsed dict
        if isinstance(history_block, str):
            try:
                # Try to extract JSON from the string
                json_match = re.search(r'\{[\s\S]*\}', history_block)
                if json_match:
                    history_block = json.loads(json_match.group())
                else:
                    continue
            except (json.JSONDecodeError, Exception):
                continue

        if not isinstance(history_block, dict):
            continue

        # Iterate through timestamped events
        for timestamp, event in history_block.items():
            if not isinstance(event, dict):
                continue

            source_date = timestamp

            # Check for static likes
            if '[Fact] Likes' in event:
                fact = event['[Fact] Likes']
                preferences.append(PreferenceItem(
                    fact=f"likes {fact.lower()}" if not fact.lower().startswith('like') else fact.lower(),
                    preference_type="current",
                    source_date=source_date,
                    topic=topic,
                ))

            # Check for static dislikes
            if '[Fact] Dislikes' in event:
                fact = event['[Fact] Dislikes']
                preferences.append(PreferenceItem(
                    fact=f"dislikes {fact.lower()}" if not fact.lower().startswith('dislike') else fact.lower(),
                    preference_type="current",
                    source_date=source_date,
                    topic=topic,
                ))

            # Check for updated preferences (like -> dislike or dislike -> like)
            if '[Updated Fact] Likes' in event:
                new_fact = event['[Updated Fact] Likes']
                old_fact = event.get('[Old Fact] Dislikes', event.get('[Fact] Dislikes', ''))
                reason = event.get('[Reasons of Change]', '')

                preferences.append(PreferenceItem(
                    fact=f"now likes {new_fact.lower()}" if not new_fact.lower().startswith('like') else new_fact.lower(),
                    preference_type="updated",
                    source_date=source_date,
                    topic=topic,
                    old_value=f"disliked {old_fact.lower()}" if old_fact else None,
                    reason_of_change=reason if reason else None,
                ))

            if '[Updated Fact] Dislikes' in event:
                new_fact = event['[Updated Fact] Dislikes']
                old_fact = event.get('[Old Fact] Likes', event.get('[Fact] Likes', ''))
                reason = event.get('[Reasons of Change]', '')

                preferences.append(PreferenceItem(
                    fact=f"now dislikes {new_fact.lower()}" if not new_fact.lower().startswith('dislike') else new_fact.lower(),
                    preference_type="updated",
                    source_date=source_date,
                    topic=topic,
                    old_value=f"liked {old_fact.lower()}" if old_fact else None,
                    reason_of_change=reason if reason else None,
                ))

    logger.info(f"Extracted {len(preferences)} preferences for TOD evaluation")
    return preferences


def postprocess_conversation_for_tod(
    data: Dict[str, Any],
    topic: str = ""
) -> List[Dict[str, str]]:
    """
    Post-process PersonaMem conversation data into clean user/assistant format for TOD evaluation.

    Expects structured format: data dict with conversation keys containing GeneratedConversation objects.

    This function converts to clean format:
    [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]

    Args:
        data: The conversation data with GeneratedConversation objects
        topic: The conversation topic (for logging purposes)

    Returns:
        List of message dicts with "role" and "content" keys
    """
    messages = []
    side_note_count = 0

    # Keys that contain conversation data and their corresponding periods
    conversation_keys = [
        ('Init Conversation', 'INIT'),
        ('Conversation Next Week', 'WEEK'),
        ('Conversation Next Month', 'MONTH'),
        ('Conversation Next Year', 'YEAR'),
    ]

    for conv_key, period in conversation_keys:
        conversation = data.get(conv_key)
        if not conversation:
            continue

        # Must be GeneratedConversation - structured output format
        if isinstance(conversation, GeneratedConversation):
            structured_conv = conversation
        elif isinstance(conversation, dict) and 'turns' in conversation:
            # Already serialized from GeneratedConversation.model_dump()
            structured_conv = GeneratedConversation(**conversation)
        else:
            logger.warning(f"Unexpected conversation format for {conv_key}: {type(conversation)}. Expected GeneratedConversation.")
            continue

        # Process from structured format - roles are explicit, no guessing needed
        for turn in structured_conv.turns:
            messages.append({"role": turn.role, "content": turn.content})
            if turn.side_note is not None:
                side_note_count += 1

    logger.info(f"Post-processed {len(messages)} conversation turns, {side_note_count} side notes")
    return messages


def save_tod_ready_data(data: Dict[str, Any], output_path: str, topic: str) -> tuple[str, str]:
    """
    Save conversation data in two files:
    1. Main conversation file - clean messages format for TOD evaluation
    2. Artifacts file - all intermediate generation artifacts for debugging/analysis

    Args:
        data: The original conversation JSON data
        output_path: Path to the original output file
        topic: The conversation topic

    Returns:
        Tuple of (conversation_path, artifacts_path)
    """
    from datetime import datetime

    # Extract preferences
    preferences = extract_preferences_for_tod(data, topic)

    # Post-process conversation to clean messages format
    messages = postprocess_conversation_for_tod(data, topic=topic)

    # =========================================================================
    # 1. Save main conversation file (clean messages format)
    # =========================================================================
    conversation_path = output_path.replace('.json', '_conversation.json')
    with open(conversation_path, 'w', encoding='utf-8') as f:
        json.dump(messages, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved conversation ({len(messages)} turns) to {conversation_path}")

    # =========================================================================
    # 2. Save artifacts file (full generation journey)
    # =========================================================================

    # Parse Topic-Specific Hobbies to extract structured data
    hobbies_data = data.get('Topic-Specific Hobbies', '')
    hobbies_events = {}
    likes_list = []
    dislikes_list = []

    if hobbies_data and isinstance(hobbies_data, str):
        try:
            # Extract the hobbies list description
            lines = hobbies_data.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('[') and '] Likes' in line:
                    # Extract like items
                    item = line.split('] Likes')[1].strip()
                    if item:
                        likes_list.append(item)
                elif line.startswith('[') and '] Dislikes' in line:
                    # Extract dislike items
                    item = line.split('] Dislikes')[1].strip()
                    if item:
                        dislikes_list.append(item)

            # Extract JSON events
            json_match = re.search(r'\{[\s\S]*\}', hobbies_data)
            if json_match:
                hobbies_events = json.loads(json_match.group())
        except (json.JSONDecodeError, Exception) as e:
            logger.debug(f"Could not fully parse Topic-Specific Hobbies: {e}")

    # Helper function to convert GeneratedConversation to serializable format
    def serialize_conversation(conv):
        """Convert conversation to JSON-serializable format."""
        if isinstance(conv, GeneratedConversation):
            return conv.model_dump()
        return conv

    # Build artifacts structure
    artifacts = {
        "metadata": {
            "topic": topic,
            "timestamp": datetime.now().isoformat(),
            "original_output_file": output_path,
            "conversation_file": conversation_path,
            "num_conversation_turns": len(messages),
            "num_preferences": len(preferences),
        },
        "persona": {
            "original": data.get("Original Persona", ""),
            "expanded": data.get("Expanded Persona", ""),
        },
        "preferences": {
            "likes": likes_list,
            "dislikes": dislikes_list,
            "extracted": [p.to_dict() for p in preferences],
        },
        "events": hobbies_events,
        "history": {
            "general": {
                "init": data.get("Init General Personal History", {}),
                "next_week": data.get("General Personal History Next Week", {}),
                "next_month": data.get("General Personal History Next Month", {}),
                "next_year": data.get("General Personal History Next Year", {}),
            },
            "contextual": {
                "init": data.get("Init Contextual Personal History", {}),
                "next_week": data.get("Contextual Personal History Next Week", {}),
                "next_month": data.get("Contextual Personal History Next Month", {}),
                "next_year": data.get("Contextual Personal History Next Year", {}),
            },
        },
        "raw_conversations": {
            "init": serialize_conversation(data.get("Init Conversation", [])),
            "next_week": serialize_conversation(data.get("Conversation Next Week", [])),
            "next_month": serialize_conversation(data.get("Conversation Next Month", [])),
            "next_year": serialize_conversation(data.get("Conversation Next Year", [])),
        },
    }

    artifacts_path = output_path.replace('.json', '_artifacts.json')
    with open(artifacts_path, 'w', encoding='utf-8') as f:
        json.dump(artifacts, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved artifacts to {artifacts_path}")

    return conversation_path, artifacts_path


class AzureQueryLLM:
    """
    Azure OpenAI version of QueryLLM that mirrors the interface of the original QueryLLM class.
    Uses Chat Completions API instead of Assistants API (not available in Azure OpenAI).
    """

    def __init__(self, args):
        logger.info("=" * 60)
        logger.info("INITIALIZING AzureQueryLLM")
        logger.info("=" * 60)

        from azure.identity import DefaultAzureCredential, get_bearer_token_provider
        from openai import AzureOpenAI

        from persona_gym.personamem_core import prompts as prompts_module

        self.args = args
        self.prompts = prompts_module

        # Log Azure configuration
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION")

        logger.info(f"Azure Endpoint: {endpoint}")
        logger.info(f"Azure Deployment: {deployment}")
        logger.info(f"API Version: {api_version}")

        # Initialize Azure OpenAI client with DefaultAzureCredential
        logger.debug("Creating DefaultAzureCredential...")
        credential = DefaultAzureCredential()
        token_provider = get_bearer_token_provider(
            credential, "https://cognitiveservices.azure.com/.default"
        )
        logger.debug("DefaultAzureCredential created successfully")

        logger.debug("Creating AzureOpenAI client with token-based auth...")
        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider=token_provider,
            api_version=api_version,
        )
        self.deployment = deployment
        logger.info("AzureOpenAI client initialized successfully (using DefaultAzureCredential)")

        # Conversation histories for different threads (simulating Assistants API threads)
        self._conversation_histories = {
            'persona': [],
            'conversation': [],
            'reflect_conversation': [],
            'preparing_new_content': [],
            'new_content': [],
            'eval_new_content': [],
            'irrelevant': [],
        }
        logger.debug(f"Initialized conversation history threads: {list(self._conversation_histories.keys())}")

        # State variables (same as original QueryLLM)
        self.expanded_persona = ""
        self.general_personal_history = ""
        self.init_general_personal_history = ""
        self.first_expand_general_personal_history = ""
        self.second_expand_general_personal_history = ""
        self.third_expand_general_personal_history = ""
        self.init_personal_history = ""
        self.first_expand_personal_history = ""
        self.second_expand_personal_history = ""
        self.third_expand_personal_history = ""
        logger.debug("Initialized all state variables for persona and history tracking")
        logger.info("AzureQueryLLM initialization complete")

    def create_a_thread(self, step):
        """Create/reset conversation history for a step (mirrors original API)."""
        logger.info(f"Creating thread for step: '{step}'")
        if step == 'conversation':
            self._conversation_histories['persona'] = []
            self._conversation_histories['conversation'] = []
            self._conversation_histories['reflect_conversation'] = []
            logger.debug("Reset conversation histories: persona, conversation, reflect_conversation")
        elif step == 'writing':
            self._conversation_histories['preparing_new_content'] = []
            logger.debug("Reset conversation history: preparing_new_content")
        elif step == 'qa':
            self._conversation_histories['new_content'] = []
            self._conversation_histories['eval_new_content'] = []
            logger.debug("Reset conversation histories: new_content, eval_new_content")
        elif step == 'irrelevant':
            self._conversation_histories['irrelevant'] = []
            logger.debug("Reset conversation history: irrelevant")
        else:
            logger.error(f"Invalid step for create_a_thread: {step}")
            raise ValueError(f'Invalid step: {step}')
        logger.info(f"Thread created successfully for step: '{step}'")

    def delete_a_thread(self, step):
        """Clear conversation history for a step (mirrors original API)."""
        logger.debug(f"Deleting/resetting thread for step: '{step}'")
        self.create_a_thread(step)  # Same as resetting

    def _get_thread_key(self, step):
        """Map step to the appropriate conversation history key."""
        if step in ['source_data', 'init_conversation', 'first_expand_conversation',
                    'second_expand_conversation', 'third_expand_conversation']:
            return 'conversation'
        elif step == 'reflect_conversation':
            return 'reflect_conversation'
        elif step == 'prepare_new_content':
            return 'preparing_new_content'
        elif step == 'eval_new_content':
            return 'eval_new_content'
        elif step in ['random_question', 'random_question_follow_up', 'random_question_follow_up_response']:
            return 'irrelevant'
        else:
            return 'persona'

    def _call_chat_completion(self, messages, model=None):
        """Call Azure OpenAI Chat Completions API."""
        model_used = model or self.deployment
        logger.debug("Calling Azure OpenAI Chat Completions API")
        logger.debug(f"  Model: {model_used}")
        logger.debug(f"  Number of messages: {len(messages)}")
        logger.debug("  Max tokens: 10000")

        response = self.client.chat.completions.create(
            model=model_used,
            messages=messages,
            max_tokens=10000,
        )

        response_text = response.choices[0].message.content
        logger.debug(f"  Response length: {len(response_text)} characters")
        return response_text

    def query_llm_structured(self, prompt: str, response_schema, model=None, verbose: bool = False):
        """Query LLM with structured output using Pydantic schema.

        Uses OpenAI's response_format parameter to guarantee the response
        matches the specified Pydantic schema. This eliminates the need for
        regex parsing, json_repair, and legacy format conversion.

        Args:
            prompt: The prompt to send to the LLM
            response_schema: A Pydantic BaseModel class defining the expected output structure
            model: Optional model override (defaults to deployment)
            verbose: Whether to print the response

        Returns:
            An instance of the response_schema class with parsed data
        """
        model_used = model or self.deployment
        logger.debug("Calling Azure OpenAI Chat Completions API with structured output")
        logger.debug(f"  Model: {model_used}")
        logger.debug(f"  Schema: {response_schema.__name__}")

        response = self.client.beta.chat.completions.parse(
            model=model_used,
            messages=[{"role": "user", "content": prompt}],
            response_format=response_schema,
            max_tokens=10000,
        )

        parsed = response.choices[0].message.parsed

        if verbose:
            import utils
            print(f'{utils.Colors.OKGREEN}Structured Response:{utils.Colors.ENDC}')
            print(parsed.model_dump_json(indent=2))

        logger.info(f"Structured output received: {response_schema.__name__} with {len(parsed.turns) if hasattr(parsed, 'turns') else 'N/A'} turns")
        return parsed

    def query_llm(self, step='source_data', persona=None, topic=None, seed=None, data=None,
                  action=None, data_type=None, idx_topic=0, start_time=None, verbose=False):
        """
        Query the LLM - mirrors the exact interface of the original QueryLLM.query_llm().
        Uses the same prompts from prompts.py module.
        """
        logger.info("-" * 60)
        logger.info(f"QUERY LLM - Step: '{step}'")
        logger.info("-" * 60)
        logger.debug("  Parameters:")
        logger.debug(f"    topic={topic}, idx_topic={idx_topic}")
        logger.debug(f"    action={action}, data_type={data_type}")
        logger.debug(f"    start_time={start_time}, verbose={verbose}")
        if persona:
            logger.debug(f"    persona={persona[:100]}..." if len(str(persona)) > 100 else f"    persona={persona}")

        # Build prompt using the same logic as original QueryLLM
        logger.debug(f"Building prompt for step: '{step}'")
        if step == 'source_data':
            prompt = self.prompts.prompts_for_background_data(seed)
        elif step == 'elaborate_topic':
            prompt = self.prompts.prompts_for_elaborating_topic(topic)
        elif step == 'expand_persona':
            prompt = self.prompts.prompts_for_expanding_persona(persona, start_time)
        elif step == 'random_question':
            prompt = data + " Explain thoroughly in details. "
        elif step == 'random_question_follow_up':
            prompt = self.prompts.prompts_for_random_question_follow_up()
        elif step == 'random_question_follow_up_response':
            prompt = data + " Explain thoroughly in details. "
        elif step == 'translate_code':
            prompt = self.prompts.prompts_for_translating_code(data, persona)
        elif step == 'rewrite_email':
            prompt = self.prompts.prompts_for_rewriting_email(data, persona)
        elif step == 'rewrite_creative_writing':
            prompt = self.prompts.prompts_for_rewriting_creative_writing(data, persona)
        elif step == 'init_general_personal_history':
            prompt = self.prompts.prompts_for_init_general_personal_history(persona, start_time)
        elif step == 'first_expand_general_personal_history':
            prompt = self.prompts.prompts_for_expanding_personal_history(type='general', period='WEEK')
        elif step == 'second_expand_general_personal_history':
            prompt = self.prompts.prompts_for_expanding_personal_history(type='general', period='MONTH')
        elif step == 'third_expand_general_personal_history':
            prompt = self.prompts.prompts_for_expanding_personal_history(type='general', period='YEAR')
        elif step == 'init_contextual_personal_history':
            prompt = self.prompts.prompts_for_init_contextual_personal_history(topic, start_time, self.expanded_persona, self.general_personal_history)
        elif step == 'first_expand_contextual_personal_history':
            prompt = self.prompts.prompts_for_expanding_personal_history(topic=topic, type='contextual', period='WEEK')
        elif step == 'second_expand_contextual_personal_history':
            prompt = self.prompts.prompts_for_expanding_personal_history(topic=topic, type='contextual', period='MONTH')
        elif step == 'third_expand_contextual_personal_history':
            prompt = self.prompts.prompts_for_expanding_personal_history(topic=topic, type='contextual', period='YEAR')
        elif step == 'init_conversation':
            prompt = self.prompts.prompts_for_generating_conversations(topic, self.expanded_persona, curr_personal_history=self.init_personal_history, period='INIT')
        elif step == 'first_expand_conversation':
            prompt = self.prompts.prompts_for_generating_conversations(topic, self.expanded_persona, curr_personal_history=self.first_expand_personal_history, period='WEEK')
        elif step == 'second_expand_conversation':
            prompt = self.prompts.prompts_for_generating_conversations(topic, self.expanded_persona, curr_personal_history=self.second_expand_personal_history, period='MONTH')
        elif step == 'third_expand_conversation':
            prompt = self.prompts.prompts_for_generating_conversations(topic, self.expanded_persona, curr_personal_history=self.third_expand_personal_history, period='YEAR')
        elif step == 'reflect_init_conversation':
            prompt = self.prompts.prompts_for_reflecting_conversations(topic, data={'history_block': self.init_personal_history, 'conversation_block': data}, round=action, period='INIT')
        elif step == 'reflect_first_expand_conversation':
            prompt = self.prompts.prompts_for_reflecting_conversations(topic, data={'history_block': self.first_expand_personal_history, 'conversation_block': data}, round=action, period='WEEK')
        elif step == 'reflect_second_expand_conversation':
            prompt = self.prompts.prompts_for_reflecting_conversations(topic, data={'history_block': self.second_expand_personal_history, 'conversation_block': data}, round=action, period='MONTH')
        elif step == 'reflect_third_expand_conversation':
            prompt = self.prompts.prompts_for_reflecting_conversations(topic, data={'history_block': self.third_expand_personal_history, 'conversation_block': data}, round=action, period='YEAR')
        elif step == 'expand_conversation_section':
            prompt = self.prompts.prompts_for_expanding_conversation_section(topic, data)
        elif step == 'qa_helper':
            prompt = self.prompts.prompts_for_generating_qa(data, action)
        elif step == 'prepare_new_content':
            prompt = self.prompts.prompt_for_preparing_new_content(data, action, data_type)
        elif step == 'new_content':
            prompt = self.prompts.prompt_for_content_generation(data, action)
        elif step == 'eval_new_content':
            prompt = self.prompts.prompt_for_evaluating_content(data, action)
        elif step == 'find_stereotype':
            prompt = self.prompts.prompts_for_classifying_stereotypical_preferences(data)
        else:
            logger.error(f"Invalid step: {step}")
            raise ValueError(f'Invalid step: {step}')

        # Log the prompt (truncated for readability)
        prompt_preview = prompt[:500] + "..." if len(prompt) > 500 else prompt
        logger.debug(f"Generated prompt ({len(prompt)} chars): {prompt_preview}")

        # Determine if this is an independent call or multi-turn
        independent_steps = ['expand_persona', 'qa_helper', 'expand_conversation_section', 'translate_code',
                             'rewrite_email', 'rewrite_creative_writing', 'new_content', 'find_stereotype']

        if step in independent_steps:
            # Independent API call (single turn)
            logger.info("  Mode: Independent (single-turn) API call")
            model = self.deployment
            if step == 'expand_conversation_section':
                # Original uses gpt-4o-mini for this step
                model = os.getenv("AZURE_OPENAI_DEPLOYMENT_MINI", self.deployment)
                logger.debug(f"  Using mini model for expand_conversation_section: {model}")

            messages = [{"role": "user", "content": prompt}]
            logger.debug("  Sending request to Azure OpenAI...")
            response = self._call_chat_completion(messages, model=model)
            logger.info(f"  Response received ({len(response)} chars)")

            if verbose:
                print(f'{utils.Colors.OKGREEN}{step.capitalize()}:{utils.Colors.ENDC} {response}')
        else:
            # Multi-turn API call (using conversation history)
            thread_key = self._get_thread_key(step)
            logger.info(f"  Mode: Multi-turn API call (thread: '{thread_key}')")
            history = self._conversation_histories[thread_key]
            logger.debug(f"  Current history length: {len(history)} messages")

            # Add user message to history
            history.append({"role": "user", "content": prompt})

            # Build messages with system prompt
            system_message = "You are a helpful assistant that generates persona-oriented conversational data in an user specified topic."
            messages = [{"role": "system", "content": system_message}] + history
            logger.debug(f"  Total messages in request: {len(messages)}")

            logger.debug("  Sending request to Azure OpenAI...")
            response = self._call_chat_completion(messages)
            logger.info(f"  Response received ({len(response)} chars)")

            # Add assistant response to history
            history.append({"role": "assistant", "content": response})
            logger.debug(f"  Updated history length: {len(history)} messages")

            if verbose:
                if step == 'new_content':
                    print(f'{utils.Colors.OKGREEN}{action.capitalize()}:{utils.Colors.ENDC} {response}')
                else:
                    print(f'{utils.Colors.OKGREEN}{topic}{utils.Colors.ENDC}' if topic else '')
                    print(f'{utils.Colors.OKGREEN}{step.capitalize()}:{utils.Colors.ENDC} {response}')

        # Save state (same logic as original QueryLLM)
        logger.debug("Updating internal state variables...")
        if idx_topic == 0:
            if step == 'init_general_personal_history':
                self.general_personal_history = response
                logger.debug("  Updated: general_personal_history")
            elif step == 'first_expand_general_personal_history':
                self.general_personal_history += response
                logger.debug("  Appended to: general_personal_history")
            elif step == 'second_expand_general_personal_history':
                self.general_personal_history += response
                logger.debug("  Appended to: general_personal_history")
            elif step == 'third_expand_general_personal_history':
                self.general_personal_history += response
                logger.debug("  Appended to: general_personal_history")
            if step == 'expand_persona':
                self.expanded_persona = response
                logger.debug("  Updated: expanded_persona")

        if step == 'init_contextual_personal_history':
            self.init_personal_history += response
            logger.debug("  Appended to: init_personal_history")
        elif step == 'first_expand_contextual_personal_history':
            self.first_expand_personal_history += response
            logger.debug("  Appended to: first_expand_personal_history")
        elif step == 'second_expand_contextual_personal_history':
            self.second_expand_personal_history += response
            logger.debug("  Appended to: second_expand_personal_history")
        elif step == 'third_expand_contextual_personal_history':
            self.third_expand_personal_history += response
            logger.debug("  Appended to: third_expand_personal_history")

        logger.info(f"Query complete for step: '{step}'")
        return response


def generate_sample(args):
    """
    Generate a single sample using the exact PersonaMem pipeline.
    Calls the same functions from prepare_data.py.
    """
    logger.info("=" * 70)
    logger.info("STARTING SAMPLE GENERATION")
    logger.info("=" * 70)

    # Load all personas (same as prepare_data.py)
    logger.info("Step 1: Loading personas from file")
    persona_file = args['datasets']['persona_file']
    logger.debug(f"  Persona file: {persona_file}")

    with open(args['datasets']['persona_file'], 'r') as file:
        all_personas = file.readlines()
    logger.info(f"  Loaded {len(all_personas)} personas")

    idx_persona = int(args['inference']['start_persona_idx'])
    idx_sample = int(args['inference']['start_sample_idx'])
    logger.info(f"  Using persona index: {idx_persona}, sample index: {idx_sample}")

    # Initialize Azure LLM (drop-in replacement for QueryLLM)
    logger.info("Step 2: Initializing Azure LLM client")
    LLM = AzureQueryLLM(args)

    # Step 1: Prepare persona (calls prepare_data.prepare_persona)
    logger.info("=" * 70)
    logger.info("Step 3: PREPARING PERSONA (calling prepare_data.prepare_persona)")
    logger.info("=" * 70)
    logger.debug("  This step will:")
    logger.debug("    - Load or create a random persona")
    logger.debug("    - Expand the persona with demographic details")
    logger.debug("    - Generate initial general personal history")

    persona, expanded_persona, start_time, init_general_personal_history, \
        general_personal_history_next_week, general_personal_history_next_month, \
        general_personal_history_next_year = prepare_persona(LLM, idx_persona, all_personas, args)

    logger.info("  Persona preparation complete")
    logger.debug(f"  Original persona: {persona[:200]}..." if len(str(persona)) > 200 else f"  Original persona: {persona}")
    logger.debug(f"  Start time: {start_time}")
    logger.debug(f"  Has init_general_personal_history: {init_general_personal_history is not None}")
    logger.debug(f"  Has general_personal_history_next_week: {general_personal_history_next_week is not None}")
    logger.debug(f"  Has general_personal_history_next_month: {general_personal_history_next_month is not None}")
    logger.debug(f"  Has general_personal_history_next_year: {general_personal_history_next_year is not None}")

    # Get topic
    logger.info("Step 4: Determining topic")
    if args['datasets']['topics'] == ['all']:
        all_topics = utils.get_all_context_names()
        logger.debug(f"  Using all available topics: {all_topics}")
    else:
        all_topics = [ctx.strip() for ctx in args['datasets']['topics']]
        logger.debug(f"  Using specified topics: {all_topics}")

    curr_topic = all_topics[0]  # Use first topic for sample
    idx_topic = 0
    logger.info(f"  Selected topic: '{curr_topic}'")

    # Step 2: Prepare topics (calls prepare_data.prepare_topics)
    logger.info("=" * 70)
    logger.info("Step 5: PREPARING TOPICS (calling prepare_data.prepare_topics)")
    logger.info("=" * 70)
    logger.debug("  This step will:")
    logger.debug("    - Set up source data directory for the topic")
    logger.debug("    - Load available source files for context")

    source_dir, all_source_files = prepare_topics(idx_topic, all_topics, curr_topic, args)

    logger.info(f"  Source directory: {source_dir}")
    logger.info(f"  Number of source files: {len(all_source_files) if all_source_files else 0}")

    # Step 3: Set up output file path and in-memory data collector
    logger.info("Step 6: Setting up output file path and data collector")
    # Determine file suffix based on mode
    if args['inference'].get('debug_mode', False):
        mode_suffix = "_debug"
    elif args['inference'].get('quick_mode', False):
        mode_suffix = "_quick"
    else:
        mode_suffix = ""
    output_file_path = os.path.join(
        args['inference']['output_dir'],
        curr_topic,
        f'{args["inference"]["output_file_name"]}_{curr_topic}_persona{idx_persona}_sample{idx_sample}{mode_suffix}.json'
    )
    logger.info(f"  Output file path: {output_file_path}")

    # Initialize in-memory data collector instead of writing to intermediate file
    data_collector = {}
    logger.debug("  Initializing in-memory data collector...")
    data_collector['Original Persona'] = persona
    logger.debug("    Collected: Original Persona")
    data_collector['Expanded Persona'] = expanded_persona
    logger.debug("    Collected: Expanded Persona")
    data_collector['Topic'] = curr_topic
    logger.debug("    Collected: Topic")

    print(f'{utils.Colors.OKGREEN}Output file path: {output_file_path}{utils.Colors.ENDC}')

    # Load source data
    logger.info("Step 7: Loading source data")
    source_data = utils.load_one_source_data(source_dir, all_source_files, curr_topic) if all_source_files is not None else None
    if source_data:
        logger.info(f"  Loaded source data ({len(str(source_data))} chars)")
        logger.debug(f"  Source data preview: {str(source_data)[:300]}...")
    else:
        logger.info("  No source data available for this topic")

    # Step 4: Generate data (calls prepare_data functions)
    logger.info("=" * 70)
    logger.info("Step 8: GENERATING CONVERSATION DATA")
    logger.info("=" * 70)

    if curr_topic in ['writing', 'email', 'coding']:
        logger.info(f"  Topic '{curr_topic}' uses WRITING pipeline")
        logger.debug("  This pipeline will:")
        logger.debug("    - Extract writing preferences from persona")
        logger.debug("    - Rewrite source content with persona style")
        logger.debug("    - Convert to conversation format")

        LLM.create_a_thread(step='writing')
        logger.info("  Calling prepare_data_on_writing_topic()...")
        prepare_data_on_writing_topic(LLM, curr_topic, persona, source_data, output_file_path, args, data_collector=data_collector)
    else:
        logger.info(f"  Topic '{curr_topic}' uses CONVERSATION pipeline")
        logger.debug("  This pipeline will:")
        logger.debug("    - Generate general personal history events")
        logger.debug("    - Generate topic-specific contextual history")
        logger.debug("    - Expand histories over time (week, month, year)")
        logger.debug("    - Convert histories to multi-turn conversations")
        logger.debug("    - Reflect and expand conversation sections")

        LLM.create_a_thread(step='conversation')
        logger.info("  Calling prepare_data_on_other_topics()...")
        prepare_data_on_other_topics(
            LLM, expanded_persona, source_data, source_dir, curr_topic, idx_topic, start_time, output_file_path,
            init_general_personal_history, general_personal_history_next_week,
            general_personal_history_next_month, general_personal_history_next_year, args,
            data_collector=data_collector
        )

    logger.info("=" * 70)
    logger.info("SAMPLE GENERATION COMPLETE")
    logger.info("=" * 70)

    # Extract preferences for TOD evaluation
    logger.info("Step 9: Extracting preferences for TOD evaluation...")
    try:
        # Use data_collector directly instead of reading from intermediate file
        generated_data = data_collector

        tod_preferences = extract_preferences_for_tod(generated_data, curr_topic)

        # Store preferences in data_collector (no file write needed)
        data_collector['TOD_Preferences'] = [p.to_dict() for p in tod_preferences]
        logger.info(f"Extracted {len(tod_preferences)} TOD preferences")

        # Save clean conversation and artifacts files directly from data_collector
        logger.info("Step 10: Creating conversation and artifacts files...")
        conversation_path, artifacts_path = save_tod_ready_data(data_collector, output_file_path, curr_topic)
        print(f'{utils.Colors.OKBLUE}Conversation file: {conversation_path}{utils.Colors.ENDC}')
        print(f'{utils.Colors.OKBLUE}Artifacts file: {artifacts_path}{utils.Colors.ENDC}')

        # No intermediate file to remove - data was collected in memory

    except Exception as e:
        logger.warning(f"Could not extract TOD preferences: {e}")
        print(f'{utils.Colors.OKGREEN}Sample data generation complete!{utils.Colors.ENDC}')
        logger.info(f"WARNING: Could not save final files due to error: {e}")
        return None, None

    print(f'{utils.Colors.OKGREEN}Sample data generation complete!{utils.Colors.ENDC}')
    logger.info(f"SUCCESS! Generated: {conversation_path}, {artifacts_path}")
    return conversation_path, artifacts_path


def main():
    """Entry point - mirrors prepare_data.py argument handling."""
    logger.info("=" * 70)
    logger.info("PersonaMem Data Generation Script Started")
    logger.info("=" * 70)
    logger.info(f"Log file: {os.path.join(LOG_DIR, 'data_generation.log')}")

    # Validate Azure environment variables
    logger.info("Validating Azure environment variables...")
    required_vars = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT",
                     "AZURE_OPENAI_API_VERSION"]
    missing = [v for v in required_vars if not os.getenv(v)]

    for var in required_vars:
        value = os.getenv(var)
        if value:
            logger.debug(f"  {var}: {value}")
        else:
            logger.error(f"  {var}: MISSING!")

    if missing:
        logger.error(f"Missing environment variables: {', '.join(missing)}")
        print(f"{utils.Colors.FAIL}Error: Missing environment variables: {', '.join(missing)}{utils.Colors.ENDC}")
        print("Please create a .env file with these variables. See .env.template")
        sys.exit(1)

    logger.info("All required environment variables present")

    # Load config from package directory
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    logger.info(f"Loading configuration from {config_path}...")
    try:
        with open(config_path, 'r') as file:
            args = yaml.safe_load(file)
        logger.info("Configuration loaded successfully")
        logger.debug(f"  Config keys: {list(args.keys())}")
    except Exception as e:
        logger.error(f"Error reading config.yaml: {e}")
        print(f'Error reading config.yaml: {e}')
        sys.exit(1)

    # Parse command-line arguments (same as prepare_data.py)
    logger.info("Parsing command-line arguments...")
    parser = argparse.ArgumentParser(description='Generate 1 sample using PersonaMem pipeline with Azure OpenAI')
    parser.add_argument('--topic', type=str, default="therapy", help='Conversation topic')
    parser.add_argument('--output_dir', type=str, default='outputs/', help='Output directory')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='Print verbose output')
    parser.add_argument('--quick', dest='quick', action='store_true',
                        help='Quick test mode: init + week steps (~2-3 min instead of 15+)')
    parser.add_argument('--debug', dest='debug', action='store_true',
                        help='Debug mode: absolute minimum for testing core logic (~30 sec)')
    cmd_args = parser.parse_args()

    logger.info(f"  --topic: {cmd_args.topic}")
    logger.info(f"  --output_dir: {cmd_args.output_dir}")
    logger.info(f"  --verbose: {cmd_args.verbose}")
    logger.info(f"  --quick: {cmd_args.quick}")
    logger.info(f"  --debug: {cmd_args.debug}")

    # Override config with command-line args
    logger.info("Applying command-line overrides to configuration...")
    args['datasets']['topics'] = [cmd_args.topic]
    args['inference']['output_dir'] = cmd_args.output_dir
    args['inference']['verbose'] = cmd_args.verbose
    args['inference']['num_personas'] = 1
    args['inference']['num_samples_per_topic'] = 1
    args['inference']['start_persona_idx'] = 0
    args['inference']['start_sample_idx'] = 0
    args['inference']['output_file_name'] = 'sample_conversation'
    args['inference']['quick_mode'] = cmd_args.quick  # Add quick mode flag
    args['inference']['debug_mode'] = cmd_args.debug  # Add debug mode flag

    logger.debug("Final configuration:")
    logger.debug(f"  Topic: {args['datasets']['topics']}")
    logger.debug(f"  Output dir: {args['inference']['output_dir']}")
    logger.debug(f"  Verbose: {args['inference']['verbose']}")
    logger.debug(f"  Quick mode: {args['inference']['quick_mode']}")
    logger.debug(f"  Debug mode: {args['inference']['debug_mode']}")
    logger.debug(f"  Persona file: {args['datasets']['persona_file']}")

    print(f"\n{'='*60}")
    print("PersonaMem Sample Data Generation (Azure OpenAI)")
    print(f"{'='*60}")
    print(f"Topic: {cmd_args.topic}")
    if cmd_args.debug:
        print("Mode: DEBUG (minimal for testing core logic, ~30 sec)")
    elif cmd_args.quick:
        print("Mode: QUICK (init + week steps, ~2-3 min)")
    else:
        print("Mode: FULL (all steps, ~15+ min)")
    print(f"Output: {cmd_args.output_dir}")
    print(f"Log file: {os.path.join(LOG_DIR, 'data_generation.log')}")
    print(f"{'='*60}\n")

    logger.info("Starting sample generation...")
    try:
        output_path = generate_sample(args)
        logger.info(f"SUCCESS! Generated: {output_path}")
        print(f"\nGenerated: {output_path}")
    except Exception as e:
        logger.error(f"FAILED! Error: {e}", exc_info=True)
        print(f"\n{utils.Colors.FAIL}Error: {e}{utils.Colors.ENDC}")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("Script completed successfully")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
