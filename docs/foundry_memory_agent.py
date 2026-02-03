from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    MemorySearchOptions,
    MemorySearchTool,
    MemoryStoreDefaultDefinition,
    MemoryStoreDefaultOptions,
    PromptAgentDefinition,
    ResponsesAssistantMessageItemParam,
    ResponsesSystemMessageItemParam,
    ResponsesUserMessageItemParam,
)
from azure.identity import DefaultAzureCredential

# Initialize the client
client = AIProjectClient(
    endpoint="https://docimage-dev-foundry-wus.services.ai.azure.com/api/projects/docimage-dev-foundry-wus",
    credential=DefaultAzureCredential(),
)

# Create memory store
definition = MemoryStoreDefaultDefinition(
    chat_model="gpt-4.1-001",  # Your chat model deployment name
    embedding_model="text-embedding-3-small-001",  # Your embedding model deployment name
    options=MemoryStoreDefaultOptions(user_profile_enabled=True, chat_summary_enabled=True),
)

MEMORY_STORE_NAME = "my_memory_store_ny"

# Delete existing memory store if it exists (to ensure clean state)
existing_stores = [ms.name for ms in client.memory_stores.list()]
if MEMORY_STORE_NAME in existing_stores:
    client.memory_stores.delete(MEMORY_STORE_NAME)
    print(f"Deleted existing memory store: {MEMORY_STORE_NAME}")

# Create fresh memory store
memory_store = client.memory_stores.create(
    name=MEMORY_STORE_NAME,
    definition=definition,
    description="Memory store for interior design agent",
)
print(f"Created memory store: {memory_store.name}")


# Simple test conversation for memory - split into two threads
simple_test_thread = [
    # Thread 1
    [
        ResponsesSystemMessageItemParam(content="You are a helpful assistant that remembers user information."),
        ResponsesUserMessageItemParam(content="I have a blue bag and 3 bottles."),
        ResponsesAssistantMessageItemParam(content="Got it! You have a blue bag and 3 bottles. I'll remember that."),
    ],
    # Thread 2
    [
        ResponsesSystemMessageItemParam(content="You are a helpful assistant that remembers user information."),
        ResponsesUserMessageItemParam(content="I have brown pants."),
        ResponsesAssistantMessageItemParam(content="Noted! You have brown pants."),
        ResponsesUserMessageItemParam(content="I have an Xbox 3."),
        ResponsesAssistantMessageItemParam(content="Great! You have an Xbox 3."),
    ],
]

simple_query_thread = [ResponsesUserMessageItemParam(content="What items do I have?")]


scope = "user_123"

for i, thread in enumerate(simple_test_thread):
    update_poller = client.memory_stores.begin_update_memories(
        name=MEMORY_STORE_NAME,
        scope=scope,
        items=thread,  # Pass conversation items that you want to add to memory
        previous_update_id=update_poller.update_id
        if i > 0
        else None,  # Extend from previous update ID  # noqa: F821  # ty:ignore[unresolved-reference]
        update_delay=0,  # Trigger update immediately without waiting for inactivity
    )

    # Wait for the update operation to complete, but can also fire and forget
    update_result = update_poller.result()
    print(f"Updated with {len(update_result.memory_operations)} memory operations")
    for operation in update_result.memory_operations:
        print(
            f"  - Operation: {operation.kind}, Memory ID: {operation.memory_item.memory_id}, Memory Type: {operation.memory_item.kind}, Content: {operation.memory_item.content}"
        )

# Sanity check - search memories with sample query
result = client.memory_stores.search_memories(
    name=MEMORY_STORE_NAME, scope=scope, items=simple_query_thread, options=MemorySearchOptions(max_memories=5)
)

print("Sample Search Results:")
for memory in result.memories:
    print(f"  - Memory ID: {memory.memory_item.memory_id}, Content: {memory.memory_item.content}")

# Create an agent that uses the memory store
# Create memory search tool
tool = MemorySearchTool(
    memory_store_name=MEMORY_STORE_NAME,
    scope=scope,
    update_delay=300,  # Wait 5 minutes of inactivity before updating memories
)

# Create a prompt agent with memory search tool
agent = client.agents.create_version(
    agent_name="MyAgent",
    definition=PromptAgentDefinition(
        model="gpt-4.1-001",  # Use the correct deployment name
        instructions="You are a helpful assistant that answers general questions",
        tools=[tool],
    ),
)

# Now create a conversation with the agent
openai_client = client.get_openai_client()  # ty:ignore[unresolved-attribute]
conversation = openai_client.conversations.create()
print(f"Created conversation with ID: {conversation.id}")

# Create an agent response to initial user message
response = openai_client.responses.create(
    input="What color pants do I have?",
    conversation=conversation.id,
    extra_body={"agent": {"name": agent.name, "type": "agent_reference"}},
)

print(f"Response output: {response.output_text}")

# Continue the conversation - just pass the same conversation.id
response2 = openai_client.responses.create(
    input="What question did I ask you related to my pants?",
    conversation=conversation.id,
    extra_body={"agent": {"name": agent.name, "type": "agent_reference"}},
)

print(f"Response output: {response2.output_text}")
