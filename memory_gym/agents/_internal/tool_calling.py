"""Shared tool-calling loop for memory-backed agents.

Provides the Azure OpenAI tool-calling loop used by Foundry, Google, AWS,
and Foundry Local agents.  Each agent supplies its own `retrieve_fn` to
search its memory store; this module handles the LLM interaction and tool
dispatch.
"""

import json
from typing import Any, Callable

from memory_gym.client import CONFIG, PooledLLMClient, _check_content_filter, _llm_retry
from memory_gym.prompts import render_prompt

_AGENT_MAX_TOKENS: int = CONFIG["max_tokens"]["agent"]

SEARCH_MEMORIES_TOOL: dict[str, Any] = {
    "type": "function",
    "name": "search_user_memories",
    "description": "Search stored memories about the user's preferences, habits, and personal information.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query to find relevant memories about the user.",
            },
        },
        "required": ["query"],
    },
}


@_llm_retry
def _respond_with_tools(
    azure_client: Any,
    deployment: str,
    messages: list[dict[str, Any]],
    retrieve_fn: Callable[[str], list[dict]],
) -> tuple[str, list[dict]]:
    """Run the Azure OpenAI tool-calling loop.

    Args:
        azure_client: Raw AzureOpenAI client.
        deployment: Model deployment name.
        messages: Conversation messages (including system prompt).
        retrieve_fn: Callable(query) -> list of memory dicts.

    Returns:
        Tuple of (final_text, retrieved_memories) where retrieved_memories
        is a list of {"query": ..., "results": [...]} dicts.
    """
    all_retrieved: list[dict] = []

    response = azure_client.responses.create(
        model=deployment,
        input=messages,  # type: ignore[arg-type]
        tools=[SEARCH_MEMORIES_TOOL],  # type: ignore[list-item]
        max_output_tokens=_AGENT_MAX_TOKENS,
    )
    _check_content_filter(response)

    while True:
        tool_calls = [item for item in response.output if item.type == "function_call"]
        if not tool_calls:
            break

        tool_results: list[dict[str, Any]] = []
        for tool_call in tool_calls:
            if tool_call.name == "search_user_memories":
                args = json.loads(tool_call.arguments)
                query = args["query"]
                memories = retrieve_fn(query)
                all_retrieved.append({"query": query, "results": memories})
                output = json.dumps(memories, ensure_ascii=False)
            else:
                output = json.dumps({"error": f"Unknown tool: {tool_call.name}"})

            tool_results.append(
                {
                    "type": "function_call_output",
                    "call_id": tool_call.call_id,
                    "output": output,
                }
            )

        response = azure_client.responses.create(
            model=deployment,
            input=response.output + tool_results,  # type: ignore[arg-type,operator]
            tools=[SEARCH_MEMORIES_TOOL],  # type: ignore[list-item]
            max_output_tokens=_AGENT_MAX_TOKENS,
        )
        _check_content_filter(response)

    return response.output_text, all_retrieved


def respond_with_memory_search(
    llm_client: PooledLLMClient,
    prompt_name: str,
    conversation: list[dict[str, str]],
    retrieve_fn: Callable[[str], list[dict]],
) -> tuple[str, list[dict]]:
    """Build messages, acquire a pool slot, and run the tool-calling loop.

    Args:
        llm_client: Pooled LLM client for least-busy routing.
        prompt_name: Prompt template name (e.g. "agents/agent_system_memory").
        conversation: The dialogue so far.
        retrieve_fn: Memory search callable.

    Returns:
        Tuple of (response_text, retrieved_memories).
    """
    system_prompt = render_prompt(prompt_name)

    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    for msg in conversation:
        messages.append({"role": msg["role"], "content": msg["content"]})

    idx, llm = llm_client._acquire()
    try:
        return _respond_with_tools(llm._client, llm.deployment, messages, retrieve_fn)
    finally:
        llm_client._release(idx)
