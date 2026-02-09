from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    MemorySearchOptions,
    MemorySearchTool,
    MemoryStoreDefaultDefinition,
    MemoryStoreDefaultOptions,
    PromptAgentDefinition,
)
from azure.identity import DefaultAzureCredential

# Set scope to associate the memories with
# You can also use "{{$userId}}" to take the TID and OID of the request authentication header

MEMORY_STORE_NAME = "my_memory_store_ny"
SCOPE = "user"

# Initialize the client
client = AIProjectClient(
    # endpoint="https://docimage-dev-foundry-wus.services.ai.azure.com/api/projects/docimage-dev-foundry-wus",
    endpoint="https://customvision-dev-aoai.services.ai.azure.com/api/projects/customvision-dev-aoai-project",
    credential=DefaultAzureCredential()
)

# Create memory store
definition = MemoryStoreDefaultDefinition(
    chat_model="gpt-4.1-001",  # Your chat model deployment name
    embedding_model="text-embedding-3-small-001",  # Your embedding model deployment name
    options=MemoryStoreDefaultOptions(user_profile_enabled=True, chat_summary_enabled=True)
)

openai_client = client.get_openai_client()

# Create memory search tool
tool = MemorySearchTool(
    memory_store_name=MEMORY_STORE_NAME,
    scope=SCOPE,
    update_delay=5,  # Wait 1 second of inactivity before updating memories
    # In a real application, set this to a higher value like 300 (5 minutes, default)
    search_options=MemorySearchOptions(max_memories=5)
)

# Create a prompt agent with memory search tool
agent = client.agents.create_version(
    agent_name="MyAgent",
    definition=PromptAgentDefinition(
        model="gpt-4.1",
        instructions="You are a helpful assistant that answers general questions",
        tools=[tool],
    )
)


# Create a conversation with the agent with memory tool enabled
conversation = openai_client.conversations.create()
print(f"Created conversation (id: {conversation.id})")

# Create an agent response to initial user message
response = openai_client.responses.create(
    input="I prefer dark roast coffee",
    conversation=conversation.id,
    extra_body={"agent": {"name": agent.name, "type": "agent_reference"}},
)

print(f"Response output: {response.output_text}")
