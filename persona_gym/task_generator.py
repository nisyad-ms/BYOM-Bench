#!/usr/bin/env python3
"""
task_generator.py - Generate TOD Tasks from PersonaMem Conversation Data

This module generates Task-Oriented Dialogue (TOD) evaluation tasks from 
PersonaMem conversation data. Each task includes:
- A realistic task description (e.g., "Book a flight", "Find a restaurant")
- Tool schemas appropriate for the task
- Relevant user preferences extracted from conversation history
- Expected behaviors the agent should exhibit based on preferences

Usage:
    python -m persona_gym.task_generator --input <conversation.json> --output <tasks.jsonl>
"""

# =============================================================================
# PATH SETUP - Must come before ANY imports from this project
# =============================================================================
import argparse
import json
import logging
import os
import random
import re
import sys
import uuid
from typing import Any, Dict, List, Optional

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from openai import AzureOpenAI

from persona_gym.metric import PreferenceItem, TODTask
from persona_gym.personamem_core import utils

load_dotenv()

# Configure logging - log to project root logs/ folder
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, 'task_generator.log'))
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# Topic-specific Task Templates
# =============================================================================

TASK_TEMPLATES = {
    "therapy": [
        {
            "task": "Schedule a therapy session for next week",
            "tools": ["search_available_slots", "book_appointment", "send_reminder"],
            "context": "mental health appointment booking",
        },
        {
            "task": "Find a support group that matches my needs",
            "tools": ["search_support_groups", "get_group_details", "register_for_group"],
            "context": "support group discovery",
        },
        {
            "task": "Set up a self-care routine reminder system",
            "tools": ["create_reminder", "set_recurring_schedule", "customize_notification"],
            "context": "wellness routine planning",
        },
    ],
    "travel": [
        {
            "task": "Book a flight from {origin} to {destination} for {date}",
            "tools": ["search_flights", "get_flight_details", "book_flight", "select_seat"],
            "context": "flight booking",
        },
        {
            "task": "Find and book a hotel in {location} for {num_nights} nights",
            "tools": ["search_hotels", "get_hotel_details", "check_availability", "book_hotel"],
            "context": "hotel booking",
        },
        {
            "task": "Plan a day trip itinerary in {location}",
            "tools": ["search_attractions", "get_reviews", "create_itinerary", "book_activity"],
            "context": "trip planning",
        },
    ],
    "fitness": [
        {
            "task": "Create a workout plan for this week",
            "tools": ["get_exercise_library", "create_workout", "schedule_workout", "set_reminder"],
            "context": "fitness planning",
        },
        {
            "task": "Find a gym or fitness class near me",
            "tools": ["search_gyms", "search_classes", "get_reviews", "book_class"],
            "context": "gym/class discovery",
        },
        {
            "task": "Track and log today's workout",
            "tools": ["start_workout_log", "add_exercise", "record_metrics", "save_workout"],
            "context": "workout tracking",
        },
    ],
    "cooking": [
        {
            "task": "Find a recipe for dinner tonight",
            "tools": ["search_recipes", "get_recipe_details", "check_pantry", "create_shopping_list"],
            "context": "recipe discovery",
        },
        {
            "task": "Plan meals for the week",
            "tools": ["search_recipes", "create_meal_plan", "generate_shopping_list", "save_plan"],
            "context": "meal planning",
        },
        {
            "task": "Order groceries for a specific recipe",
            "tools": ["get_recipe_ingredients", "search_grocery_store", "add_to_cart", "checkout"],
            "context": "grocery ordering",
        },
    ],
    "general": [
        {
            "task": "Make a restaurant reservation for dinner",
            "tools": ["search_restaurants", "check_availability", "make_reservation", "send_confirmation"],
            "context": "restaurant booking",
        },
        {
            "task": "Schedule a meeting with a contact",
            "tools": ["check_calendar", "find_available_times", "send_invite", "confirm_meeting"],
            "context": "meeting scheduling",
        },
        {
            "task": "Find and purchase a gift",
            "tools": ["search_products", "get_product_details", "check_reviews", "add_to_cart", "checkout"],
            "context": "shopping",
        },
    ],
}

# Tool schemas with parameters
TOOL_SCHEMAS = {
    # Travel tools
    "search_flights": {
        "description": "Search for available flights",
        "params": ["origin", "destination", "date", "time_preference", "seat_preference", "class", "airlines_to_exclude"],
    },
    "get_flight_details": {
        "description": "Get detailed information about a specific flight",
        "params": ["flight_id"],
    },
    "book_flight": {
        "description": "Book a flight",
        "params": ["flight_id", "passenger_info", "seat_selection"],
    },
    "select_seat": {
        "description": "Select a seat on a flight",
        "params": ["flight_id", "seat_preference", "specific_seat"],
    },
    "search_hotels": {
        "description": "Search for hotels",
        "params": ["location", "check_in", "check_out", "guests", "amenities", "price_range"],
    },
    "get_hotel_details": {
        "description": "Get detailed hotel information",
        "params": ["hotel_id"],
    },
    "check_availability": {
        "description": "Check room availability",
        "params": ["hotel_id", "dates"],
    },
    "book_hotel": {
        "description": "Book a hotel room",
        "params": ["hotel_id", "room_type", "guest_info"],
    },
    # Restaurant tools
    "search_restaurants": {
        "description": "Search for restaurants",
        "params": ["location", "cuisine", "dietary_restrictions", "price_range", "ambiance"],
    },
    "make_reservation": {
        "description": "Make a restaurant reservation",
        "params": ["restaurant_id", "date", "time", "party_size"],
    },
    # Fitness tools
    "get_exercise_library": {
        "description": "Get list of exercises",
        "params": ["muscle_group", "equipment", "difficulty"],
    },
    "create_workout": {
        "description": "Create a workout routine",
        "params": ["exercises", "duration", "intensity"],
    },
    "search_gyms": {
        "description": "Search for gyms nearby",
        "params": ["location", "amenities", "hours"],
    },
    "search_classes": {
        "description": "Search for fitness classes",
        "params": ["type", "location", "time", "instructor"],
    },
    # Generic tools
    "search_products": {
        "description": "Search for products",
        "params": ["query", "category", "price_range", "brand"],
    },
    "add_to_cart": {
        "description": "Add item to shopping cart",
        "params": ["product_id", "quantity"],
    },
    "checkout": {
        "description": "Complete purchase",
        "params": ["payment_method", "shipping_address"],
    },
    "create_reminder": {
        "description": "Create a reminder",
        "params": ["title", "datetime", "recurrence"],
    },
    "check_calendar": {
        "description": "Check calendar availability",
        "params": ["date_range"],
    },
    "send_invite": {
        "description": "Send meeting invite",
        "params": ["attendees", "datetime", "subject", "location"],
    },
}


class AzureLLMClient:
    """Simple Azure OpenAI client for task generation."""

    def __init__(self):
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

        if not endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT environment variable required")

        credential = DefaultAzureCredential()
        token_provider = get_bearer_token_provider(
            credential, "https://cognitiveservices.azure.com/.default"
        )

        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider=token_provider,
            api_version="2024-12-01-preview"
        )
        self.deployment = deployment

    def query(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7) -> str:
        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content


def extract_preferences_from_data(data: Dict[str, Any], topic: str) -> List[PreferenceItem]:
    """Extract preferences from PersonaMem conversation data."""
    preferences = []

    # Check if preferences already exists in clean format (from TOD-ready file)
    prefs_data = data.get('preferences') or data.get('TOD_Preferences')
    if prefs_data:
        for p in prefs_data:
            preferences.append(PreferenceItem(
                fact=p.get('fact', ''),
                preference_type=p.get('type', 'current'),
                source_date=p.get('source_date', ''),
                old_value=p.get('old_value'),
                reason_of_change=p.get('reason_of_change'),
            ))
        return preferences

    # Otherwise, extract from personal history
    history_keys = [
        'Init Contextual Personal History',
        'Contextual Personal History Next Week',
        'Contextual Personal History Next Month',
        'Contextual Personal History Next Year',
    ]

    for history_key in history_keys:
        history_block = data.get(history_key)
        if not history_block:
            continue

        if isinstance(history_block, str):
            try:
                json_match = re.search(r'\{[\s\S]*\}', history_block)
                if json_match:
                    history_block = json.loads(json_match.group())
                else:
                    continue
            except (json.JSONDecodeError, Exception):
                continue

        if not isinstance(history_block, dict):
            continue

        for timestamp, event in history_block.items():
            if not isinstance(event, dict):
                continue

            if '[Fact] Likes' in event:
                preferences.append(PreferenceItem(
                    fact=f"likes {event['[Fact] Likes'].lower()}",
                    preference_type="current",
                    source_date=timestamp,
                ))

            if '[Fact] Dislikes' in event:
                preferences.append(PreferenceItem(
                    fact=f"dislikes {event['[Fact] Dislikes'].lower()}",
                    preference_type="current",
                    source_date=timestamp,
                ))

            if '[Updated Fact] Likes' in event:
                preferences.append(PreferenceItem(
                    fact=f"now likes {event['[Updated Fact] Likes'].lower()}",
                    preference_type="updated",
                    source_date=timestamp,
                    old_value=event.get('[Old Fact] Dislikes', ''),
                    reason_of_change=event.get('[Reasons of Change]', ''),
                ))

            if '[Updated Fact] Dislikes' in event:
                preferences.append(PreferenceItem(
                    fact=f"now dislikes {event['[Updated Fact] Dislikes'].lower()}",
                    preference_type="updated",
                    source_date=timestamp,
                    old_value=event.get('[Old Fact] Likes', ''),
                    reason_of_change=event.get('[Reasons of Change]', ''),
                ))

    return preferences


def generate_task_description(template: Dict, llm: AzureLLMClient, preferences: List[PreferenceItem]) -> str:
    """Generate a concrete task description from a template."""
    task = template["task"]

    # Fill in placeholders with realistic values
    placeholders = {
        "{origin}": random.choice(["New York", "Los Angeles", "Chicago", "San Francisco", "Boston"]),
        "{destination}": random.choice(["Miami", "Seattle", "Denver", "Austin", "Portland"]),
        "{date}": "next Friday",
        "{location}": random.choice(["downtown", "near the airport", "in the city center"]),
        "{num_nights}": str(random.randint(2, 5)),
    }

    for placeholder, value in placeholders.items():
        task = task.replace(placeholder, value)

    return task


def generate_expected_behaviors(preferences: List[PreferenceItem], task_context: str) -> List[str]:
    """Generate expected agent behaviors based on preferences."""
    behaviors = []

    for pref in preferences:
        fact_lower = pref.fact.lower()
        if "dislikes" in fact_lower:
            item = pref.fact.replace("dislikes ", "").replace("now dislikes ", "")
            behaviors.append(f"Should avoid recommending '{item}'")
        elif "likes" in fact_lower:
            item = pref.fact.replace("likes ", "").replace("now likes ", "")
            behaviors.append(f"Should prioritize or recommend options related to '{item}'")

        if pref.preference_type == "updated" and pref.old_value:
            behaviors.append(f"Should use LATEST preference (not old: {pref.old_value})")

    behaviors.append("Should proactively use preferences without requiring reminders")
    behaviors.append("Should complete the task efficiently")

    return behaviors


# =============================================================================
# LLM-Based Dynamic Task Generation
# =============================================================================

# Few-shot examples for diverse task generation (used as guidance, not constraints)
FEW_SHOT_TASK_EXAMPLES = """
## Example 1 (travel - booking)
Preferences: likes window seats, dislikes United Airlines, prefers morning flights
Task: "Book me a flight to Chicago for next Thursday."
Why it works: Specific booking task. Agent should proactively select morning flight, window seat, avoid United.

## Example 2 (travel - finding)
Preferences: likes hostels with social events, likes meeting travelers, dislikes luxury hotels
Task: "Find me a place to stay in Barcelona for 3 nights."
Why it works: Finding task. Agent should prioritize social hostels over hotels.

## Example 3 (travel - recommendation)
Preferences: likes culinary tourism, likes trying local foods, dislikes chain restaurants
Task: "Where should I eat dinner tonight near the city center?"
Why it works: Recommendation task. Agent should suggest local restaurants, avoid chains.

## Example 4 (cooking - finding)
Preferences: likes spicy food, dislikes dairy, prefers quick recipes
Task: "Find me a recipe for dinner tonight."
Why it works: Recipe finding. Agent should filter for dairy-free, potentially spicy, quick options.

## Example 5 (fitness - booking)
Preferences: likes yoga, dislikes crowded classes, prefers morning sessions
Task: "Book me a fitness class for Saturday."
Why it works: Booking task. Agent should look for yoga, morning time, smaller class sizes.

## Example 6 (therapy - scheduling)
Preferences: prefers video calls, likes evening appointments, dislikes group sessions
Task: "Schedule a session with my therapist for next week."
Why it works: Scheduling task. Agent should look for evening video call slots.
"""


def generate_task_with_llm(
    preferences: List[PreferenceItem],
    topic: str,
    llm: AzureLLMClient,
    previously_generated: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Use LLM to generate a task that tests the given preferences.

    Args:
        preferences: List of user preferences to test
        topic: The topic domain (travel, cooking, therapy, etc.)
        llm: Azure LLM client
        previously_generated: List of previously generated task descriptions (for variety)

    Returns a dict with task details.
    """

    # Format preferences for the prompt
    pref_descriptions = []
    for p in preferences:
        pref_str = f"- {p.fact}"
        if p.preference_type == "updated" and p.old_value:
            pref_str += f" (CHANGED from: {p.old_value})"
            if p.reason_of_change:
                pref_str += f" because: {p.reason_of_change}"
        pref_descriptions.append(pref_str)

    prefs_text = "\n".join(pref_descriptions)

    # Add variety instruction if we have previously generated tasks
    variety_instruction = ""
    if previously_generated and len(previously_generated) > 0:
        prev_tasks = "\n".join(f"  - {t}" for t in previously_generated)
        variety_instruction = f"""
**IMPORTANT - ENSURE VARIETY:**
You have already generated these tasks:
{prev_tasks}

Generate a DIFFERENT type of task. For example:
- If previous was "booking", try "finding" or "recommendation"
- If previous was about accommodations, try restaurants or activities
- Vary the action type AND the subject matter
"""

    prompt = f"""You are designing a task-oriented dialogue evaluation task for an AI assistant.

TOPIC: {topic}

USER'S KNOWN PREFERENCES (from past conversations):
{prefs_text}

## Few-Shot Examples (for reference - adapt to any topic):
{FEW_SHOT_TASK_EXAMPLES}

## Your Task
{variety_instruction}
Create a SPECIFIC, NARROW task for the "{topic}" domain where:
1. The task is a SINGLE, FOCUSED action (book ONE thing, find ONE thing, recommend ONE thing, etc.)
2. Only 2-3 of the user's preferences are directly relevant (not all of them!)
3. An assistant who REMEMBERS these preferences could proactively apply them
4. An assistant WITHOUT access to history would need to ask clarifying questions

**CRITICAL RULES:**
1. The task_description must be GENERIC and NOT mention any preferences
2. Create a SPECIFIC task type (booking, finding, recommendation, scheduling, etc.)
3. The task should test only a SUBSET of preferences, not all of them
4. Do NOT create broad "plan everything" or "help me with X" tasks

Output in this exact JSON format:
{{
    "task_type": "booking|finding|recommendation|scheduling|other",
    "task_description": "A SPECIFIC task request (1 sentence, no preferences mentioned)",
    "relevant_preference_subset": ["which preferences this tests"],
    "why_preferences_matter": "Brief explanation",
    "tools": ["tool_name_1", "tool_name_2"],
    "tool_schemas": {{
        "tool_name_1": {{
            "description": "What this tool does",
            "params": ["param1", "param2"]
        }}
    }},
    "expected_behaviors": [
        "Should do X because user likes Y",
        "Should avoid Z because user dislikes W"
    ]
}}

Generate a specific task for the "{topic}" domain:"""

    response = llm.query(prompt, max_tokens=1500, temperature=0.7)

    # Parse JSON from response
    try:
        # Find JSON block in response
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            result = json.loads(json_match.group())
            return result
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"Failed to parse LLM response as JSON: {e}")

    return None


def generate_tod_task(
    data: Dict[str, Any],
    topic: str,
    llm: Optional[AzureLLMClient] = None,
    task_idx: int = 0,
    previously_generated: Optional[List[str]] = None,
) -> TODTask:
    """Generate a single TOD task from conversation data using LLM.

    Args:
        data: Conversation data with preferences
        topic: Topic domain
        llm: Azure LLM client
        task_idx: Index for task ID
        previously_generated: List of previously generated task descriptions (for variety)
    """

    # Extract preferences
    preferences = extract_preferences_from_data(data, topic)

    # Select a mix of likes and dislikes for more interesting evaluation
    likes = [p for p in preferences if 'likes' in p.fact and 'dislikes' not in p.fact]
    dislikes = [p for p in preferences if 'dislikes' in p.fact]
    updated = [p for p in preferences if p.preference_type == "updated"]

    # Prioritize updated preferences (they're the most interesting to test)
    relevant_preferences = updated[:2] + likes[:3] + dislikes[:3]
    if len(relevant_preferences) == 0:
        relevant_preferences = preferences[:6]

    # Remove duplicates while preserving order
    seen = set()
    unique_prefs = []
    for p in relevant_preferences:
        if p.fact not in seen:
            seen.add(p.fact)
            unique_prefs.append(p)
    relevant_preferences = unique_prefs[:6]

    # Always use LLM to generate appropriate task
    if llm is None:
        llm = AzureLLMClient()

    llm_result = generate_task_with_llm(relevant_preferences, topic, llm, previously_generated)

    if llm_result:
        task_description = llm_result.get("task_description", f"Help with a {topic} task")

        # Build tool schemas from LLM response
        tool_schemas = llm_result.get("tool_schemas", {})

        # Build expected behaviors
        expected_behaviors = llm_result.get("expected_behaviors", [])
        expected_behaviors.append("Should proactively use preferences without requiring reminders")
        expected_behaviors.append("Should complete the task efficiently")
    else:
        # Fallback if LLM fails
        logger.warning("LLM task generation failed, using fallback")
        task_description = f"Help me with a {topic}-related task"
        tool_schemas = {"general_search": {"description": "Search for options", "params": ["query"]}}
        expected_behaviors = generate_expected_behaviors(relevant_preferences, topic)

    return TODTask(
        task_id=f"{topic}_{task_idx}_{uuid.uuid4().hex[:8]}",
        task_description=task_description,
        topic=topic,
        relevant_preferences=relevant_preferences,
        expected_behaviors=expected_behaviors,
        tool_schemas=tool_schemas,
    )


def generate_all_tasks(
    data: Dict[str, Any],
    topic: str,
    num_tasks: int = 3,
    llm: Optional[AzureLLMClient] = None,
) -> List[TODTask]:
    """Generate multiple TOD tasks from conversation data with variety."""
    tasks = []
    previously_generated: List[str] = []

    for i in range(num_tasks):
        try:
            task = generate_tod_task(
                data, topic, llm,
                task_idx=i,
                previously_generated=previously_generated if i > 0 else None
            )
            tasks.append(task)
            previously_generated.append(task.task_description)
            logger.info(f"Generated task {i+1}/{num_tasks}: {task.task_description[:50]}...")
        except Exception as e:
            logger.warning(f"Failed to generate task {i+1}: {e}")

    return tasks


def main():
    parser = argparse.ArgumentParser(
        description="Generate TOD tasks from PersonaMem conversation data",
    )
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Path to input artifacts JSON file (or legacy full data file)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Path to output JSONL file (default: <input>_tasks.jsonl)')
    parser.add_argument('--num-tasks', '-n', type=int, default=1,
                        help='Number of TOD tasks to generate (default: 1)')
    parser.add_argument('--use-llm', action='store_true',
                        help='Use LLM to enhance task descriptions')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate input
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    # Set output path
    if args.output is None:
        base, _ = os.path.splitext(args.input)
        # Remove _artifacts suffix if present for cleaner output naming
        base = base.replace('_artifacts', '')
        args.output = f"{base}_tasks.jsonl"

    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")

    # Load data
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Handle new artifacts format vs legacy format
    if 'metadata' in data and 'preferences' in data:
        # New artifacts format
        logger.info("Detected new artifacts format")
        topic = data.get('metadata', {}).get('topic', 'general')
        # Convert artifacts format to format expected by extract_preferences_from_data
        converted_data = {
            'preferences': data.get('preferences', {}).get('extracted', []),
            'topic': topic,
        }
        data = converted_data
    else:
        # Legacy format (TOD-ready or full PersonaMem output)
        topic = data.get('Topic', data.get('topic', 'general'))

    logger.info(f"Topic: {topic}")

    # Initialize LLM if requested
    llm = AzureLLMClient() if args.use_llm else None

    # Generate tasks
    logger.info(f"Generating {args.num_tasks} TOD tasks...")
    tasks = generate_all_tasks(data, topic, args.num_tasks, llm)

    # Save tasks
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        for task in tasks:
            f.write(json.dumps(task.to_dict(), ensure_ascii=False) + '\n')

    logger.info(f"Saved {len(tasks)} tasks to {args.output}")

    # Print summary
    print(f"\n{utils.Colors.OKGREEN}=== TOD Task Generation Summary ==={utils.Colors.ENDC}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Topic: {topic}")
    print(f"Tasks generated: {len(tasks)}")

    for i, task in enumerate(tasks, 1):
        print(f"\n{utils.Colors.OKBLUE}Task {i}:{utils.Colors.ENDC} {task.task_description}")
        print(f"  Preferences: {len(task.relevant_preferences)}")
        print(f"  Tools: {list(task.tool_schemas.keys())}")
        print(f"  Expected behaviors: {len(task.expected_behaviors)}")


if __name__ == "__main__":
    main()
