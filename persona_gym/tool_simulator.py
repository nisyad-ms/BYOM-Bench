"""
Tool simulator for generating realistic fake tool results.

This module provides a simulator that generates plausible tool outputs without
any knowledge of user preferences. The key design principle is that the simulator
acts as a neutral, realistic API - it returns a diverse set of results based on
the query parameters, allowing the agent's preference-aware filtering to be tested.

For example, a flight search will return a realistic mix of airlines, seat types,
and times. If the agent forgets to apply preference-based filtering, it may select
an option that violates user preferences, leading to a correction turn.
"""

import json
import uuid
from typing import Optional

from persona_gym.client import get_client
from persona_gym.schemas import ToolResponse, ToolResult, ToolResultItem


def __post_init__(self):
    if self.preferences_applied is None:
        self.preferences_applied = []


TOOL_SIMULATOR_SYSTEM_PROMPT = """You are a realistic API simulator that generates plausible tool/API responses.

Your role is to generate diverse, realistic results for tool calls. You have access to the user's
known preferences, but you should ONLY use them to filter results if the agent EXPLICITLY requests
it via parameters like "additional_requirements", "preferences", "filters", or "notes".

Guidelines:
1. Generate realistic, varied results (mix of different options, prices, times, etc.)
2. Respect explicit query parameters (e.g., if date is specified, use that date)
3. Include 3-5 diverse options when returning lists
4. Make results feel authentic with realistic details (names, prices, times, addresses)
5. CRITICAL: If the agent does NOT explicitly mention a preference in the tool call parameters,
   you MUST return a diverse mix that may or may not match the user's actual preferences.
   Do NOT assume or apply preferences that weren't explicitly requested.
6. ONLY filter based on preferences when they are EXPLICITLY stated in "additional_requirements",
   "preferences", "filters", "notes", or similar free-form fields in the tool call.
7. When preferences ARE explicitly requested, filter results to match those specific requests.

Your output should be valid JSON that represents what a real API would return."""

TOOL_SIMULATOR_USER_PROMPT = """Generate a realistic response for this tool call:

Tool: {tool_name}
Parameters: {params_json}

User's Known Preferences (for reference - ONLY apply if agent explicitly requested them):
{preferences_json}

IMPORTANT: Check if the agent's parameters explicitly mention any preferences (in "additional_requirements",
"preferences", "filters", "notes", or similar fields). 
- If YES: Filter results to match those EXPLICITLY REQUESTED preferences.
- If NO: Return a diverse mix of options that may or may not match user preferences.

Return a JSON object with:
1. "result": The structured data a real API would return (list of options, confirmation, etc.)
2. "result_text": A brief human-readable summary (1-2 sentences) of what was found/done
3. "preferences_applied": List of preference keywords that were EXPLICITLY requested by the agent
   and applied to filter results. Empty list if agent didn't request any preference filtering.

Example for a flight search WITHOUT explicit preferences:
{{
    "result": {{
        "flights": [
            {{"id": "UA123", "airline": "United", "departure": "8:30 AM", "arrival": "11:45 AM", "seat_available": "14C (aisle)", "price": 380}},
            {{"id": "DL456", "airline": "Delta", "departure": "9:15 AM", "arrival": "12:30 PM", "seat_available": "12A (window)", "price": 420}},
            {{"id": "SW789", "airline": "Southwest", "departure": "2:15 PM", "arrival": "5:30 PM", "seat_available": "22B (middle)", "price": 290}}
        ]
    }},
    "result_text": "Found 3 available flights from NYC to LA on Friday, ranging from $290-$420.",
    "preferences_applied": []
}}

Example for a flight search WITH explicit preferences (agent passed "additional_requirements": "window seat, no United"):
{{
    "result": {{
        "flights": [
            {{"id": "DL456", "airline": "Delta", "departure": "9:15 AM", "arrival": "12:30 PM", "seat_available": "12A (window)", "price": 420}},
            {{"id": "AA101", "airline": "American", "departure": "10:00 AM", "arrival": "1:15 PM", "seat_available": "8A (window)", "price": 395}}
        ]
    }},
    "result_text": "Found 2 flights with window seats (excluding United) from NYC to LA on Friday.",
    "preferences_applied": ["window seat", "no United"]
}}

Respond with only the JSON, no additional text."""


class ToolSimulator:
    """Generates realistic fake tool results using an LLM.

    The simulator is preference-aware but only filters results when the agent
    EXPLICITLY requests preferences in the tool call parameters. This allows
    proper evaluation of whether agents remember and apply user preferences.

    Design principles:
    - Preference-aware: Has access to user preferences for reference
    - Explicit filtering only: Only applies preferences when agent explicitly requests them
    - Diverse output: Returns mixed results when no preferences are explicitly requested
    - Tracks applied preferences: Reports which preferences were actually used to filter

    Example:
        simulator = ToolSimulator(preferences=["prefers window seats", "dislikes United"])

        # Agent forgot to mention preferences - gets diverse (possibly violating) results
        result = simulator.execute(
            tool_name="search_flights",
            params={"origin": "NYC", "destination": "LA", "date": "2024-03-15"}
        )
        # Result will include United flights and various seat types

        # Agent remembered to add requirements - gets filtered results
        result = simulator.execute(
            tool_name="search_flights",
            params={
                "origin": "NYC", 
                "destination": "LA", 
                "date": "2024-03-15",
                "additional_requirements": "window seat, no United Airlines, morning flight"
            }
        )
        # Result will respect these explicit requirements
    """

    def __init__(self, model: Optional[str] = None, preferences: Optional[list[str]] = None):
        """Initialize the tool simulator.

        Args:
            model: Azure OpenAI deployment name for result generation.
                   Defaults to AZURE_OPENAI_DEPLOYMENT env var or 'gpt-4o'.
            preferences: List of user preference strings. The simulator will only
                        apply these if the agent explicitly requests them in tool calls.
        """
        self.preferences = preferences or []
        self._client = get_client(deployment=model)

    def execute(self, tool_name: str, params: dict) -> ToolResult:
        """Execute a simulated tool call and return realistic results.

        Args:
            tool_name: Name of the tool being called (e.g., "search_flights").
            params: Parameters passed to the tool call.

        Returns:
            ToolResult containing structured data and human-readable summary.
        """
        # Format preferences for the prompt
        preferences_json = json.dumps(self.preferences, indent=2) if self.preferences else "[]"

        user_prompt = TOOL_SIMULATOR_USER_PROMPT.format(
            tool_name=tool_name,
            params_json=json.dumps(params, indent=2),
            preferences_json=preferences_json,
        )

        result_data = self._client.complete_json(
            prompt=user_prompt,
            system_prompt=TOOL_SIMULATOR_SYSTEM_PROMPT,
            temperature=0.8,  # Some variety in results
        )

        return ToolResult(
            tool_name=tool_name,
            result=result_data.get("result", {}),
            result_text=result_data.get("result_text", "Tool executed successfully."),
            preferences_applied=result_data.get("preferences_applied", []),
        )

    def call_tool(self, tool_name: str, params: dict) -> ToolResponse:
        """Execute a tool call and return results in ToolResponse format.

        This is the preferred interface for TOD evaluation, providing
        structured results compatible with the evaluation pipeline.

        Args:
            tool_name: Name of the tool being called.
            params: Parameters passed to the tool call.

        Returns:
            ToolResponse with list of result items.
        """
        # Use execute() to get raw results
        raw_result = self.execute(tool_name, params)

        # Convert to ToolResponse format
        result_items = []
        result_data = raw_result.result

        # Handle different result structures
        if isinstance(result_data, dict):
            # Check for list fields (flights, hotels, restaurants, etc.)
            for key, value in result_data.items():
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            result_id = item.get("id") or f"{tool_name[:2].upper()}{uuid.uuid4().hex[:6].upper()}"
                            result_items.append(ToolResultItem(
                                result_id=result_id,
                                data=item,
                            ))
                    break

            # If no list found, wrap the whole result as one item
            if not result_items:
                result_items.append(ToolResultItem(
                    result_id=f"{tool_name[:3].upper()}{uuid.uuid4().hex[:6].upper()}",
                    data=result_data,
                ))

        return ToolResponse(
            tool_name=tool_name,
            success=True,
            results=result_items,
            message=raw_result.result_text,
            preferences_applied=raw_result.preferences_applied,
        )


def create_simulator_from_task(task) -> ToolSimulator:
    """Create a ToolSimulator from a TODTask object.

    The simulator is preference-aware and will filter results only when
    the agent explicitly requests preferences in tool call parameters.

    Args:
        task: TODTask object containing relevant_preferences.

    Returns:
        ToolSimulator instance with user preferences loaded.
    """
    # Extract preference facts from the task
    preferences = [p.fact for p in task.relevant_preferences] if hasattr(task, 'relevant_preferences') else []
    return ToolSimulator(preferences=preferences)


# =============================================================================
# Standard Tool Schemas
# =============================================================================

# These are generic tool schemas that don't hint at preferences.
# Agents should use "additional_requirements" for preference-based filtering.

STANDARD_TOOL_SCHEMAS = {
    "travel": [
        {
            "type": "function",
            "function": {
                "name": "search_flights",
                "description": "Search for available flights between two cities.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "origin": {
                            "type": "string",
                            "description": "Departure city or airport code"
                        },
                        "destination": {
                            "type": "string",
                            "description": "Arrival city or airport code"
                        },
                        "date": {
                            "type": "string",
                            "description": "Travel date"
                        },
                        "additional_requirements": {
                            "type": "string",
                            "description": "Any additional requirements or preferences for the search"
                        }
                    },
                    "required": ["origin", "destination", "date"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "book_flight",
                "description": "Book a specific flight.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "flight_id": {
                            "type": "string",
                            "description": "The flight identifier to book"
                        },
                        "seat": {
                            "type": "string",
                            "description": "Seat selection (if available)"
                        }
                    },
                    "required": ["flight_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_hotels",
                "description": "Search for available hotels in a location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City or area to search"
                        },
                        "check_in": {
                            "type": "string",
                            "description": "Check-in date"
                        },
                        "check_out": {
                            "type": "string",
                            "description": "Check-out date"
                        },
                        "additional_requirements": {
                            "type": "string",
                            "description": "Any additional requirements or preferences"
                        }
                    },
                    "required": ["location", "check_in", "check_out"]
                }
            }
        }
    ],
    "dining": [
        {
            "type": "function",
            "function": {
                "name": "search_restaurants",
                "description": "Search for restaurants in an area.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City or neighborhood"
                        },
                        "cuisine": {
                            "type": "string",
                            "description": "Type of cuisine (optional)"
                        },
                        "date": {
                            "type": "string",
                            "description": "Date for reservation"
                        },
                        "party_size": {
                            "type": "integer",
                            "description": "Number of guests"
                        },
                        "additional_requirements": {
                            "type": "string",
                            "description": "Any dietary restrictions, ambiance preferences, etc."
                        }
                    },
                    "required": ["location", "date"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "make_reservation",
                "description": "Make a restaurant reservation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "restaurant_id": {
                            "type": "string",
                            "description": "Restaurant identifier"
                        },
                        "date": {
                            "type": "string",
                            "description": "Reservation date"
                        },
                        "time": {
                            "type": "string",
                            "description": "Reservation time"
                        },
                        "party_size": {
                            "type": "integer",
                            "description": "Number of guests"
                        },
                        "special_requests": {
                            "type": "string",
                            "description": "Any special requests"
                        }
                    },
                    "required": ["restaurant_id", "date", "time", "party_size"]
                }
            }
        }
    ],
    "shopping": [
        {
            "type": "function",
            "function": {
                "name": "search_products",
                "description": "Search for products.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "category": {
                            "type": "string",
                            "description": "Product category (optional)"
                        },
                        "additional_requirements": {
                            "type": "string",
                            "description": "Any specific requirements (size, color, brand preferences, etc.)"
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "add_to_cart",
                "description": "Add a product to shopping cart.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "product_id": {
                            "type": "string",
                            "description": "Product identifier"
                        },
                        "quantity": {
                            "type": "integer",
                            "description": "Quantity to add"
                        }
                    },
                    "required": ["product_id"]
                }
            }
        }
    ]
}


def get_tools_for_topic(topic: str) -> list[dict]:
    """Get standard tool schemas for a given task topic.

    Args:
        topic: The task topic (e.g., "travel", "dining", "shopping").

    Returns:
        List of tool schemas in OpenAI function format.
    """
    return STANDARD_TOOL_SCHEMAS.get(topic, [])
