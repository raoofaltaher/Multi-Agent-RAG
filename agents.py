# agents.py
from pydantic import BaseModel, Field
# Only import Agent from the top level
from pydantic_ai import Agent
from qdrant_client import QdrantClient

# Use direct imports
import config
from tools import vector_search_pydantic_tool, web_search_pydantic_tool, RagDeps

# --- Router Agent ---

class RoutingDecision(BaseModel):
    vector_search: bool = Field(..., description="True if the query is best answered using the internal knowledge base (e.g., about specific indexed documents like Llama 3 or Google Gemini API details).")
    web_search: bool = Field(..., description="True if the query requires up-to-date information, current events, or general knowledge outside the scope of the indexed documents.")

ROUTER_SYSTEM_PROMPT = f"""
You are an intelligent routing agent... [omitted for brevity] ...Metacognitive, answer directly)
"""

def create_router_agent() -> Agent:
    """Creates the PydanticAI Router Agent using plain model name string."""
    print("Creating Router Agent...")
    # *** FIX: Use plain model name string directly from config ***
    plain_model_name = config.ROUTER_MODEL_NAME
    print(f"  Configuring model: {plain_model_name}")
    agent = Agent(
        # Pass the plain model name string
        model=plain_model_name,
        # Pass the Google API key; pydantic-ai *should* associate it with the model
        api_key=config.GOOGLE_API_KEY,
        description="Router Agent: Decides whether vector search or web search is needed.",
        system_prompt=ROUTER_SYSTEM_PROMPT,
        result_type=RoutingDecision
    )
    print("Router Agent created.")
    return agent

# --- RAG Agent ---

RAG_SYSTEM_PROMPT = f"""
You are a helpful Retrieval-Augmented Generation assistant... [omitted for brevity] ...for querying the storage.
"""

def create_rag_agent() -> Agent:
    """Creates the PydanticAI RAG Agent using plain model name string."""
    print("Creating RAG Agent...")
    # *** FIX: Use plain model name string directly from config ***
    plain_model_name = config.RAG_MODEL_NAME
    print(f"  Configuring model: {plain_model_name}")
    agent = Agent(
        model=plain_model_name,
        api_key=config.GOOGLE_API_KEY,
        name="KnowledgeBaseAgent",
        description="RAG Agent: Answers questions using context from internal documents via VectorSearchKnowledgeBase tool.",
        system_prompt=RAG_SYSTEM_PROMPT,
        deps_type=RagDeps,
        tools=[vector_search_pydantic_tool],
        auto_execute_tools=True
    )
    print("RAG Agent created.")
    return agent

# --- Web Search Agent ---

WEB_SEARCH_SYSTEM_PROMPT = f"""
You are a helpful Web Search assistant... [omitted for brevity] ...yielded no relevant results.
"""

def create_web_search_agent() -> Agent:
    """Creates the PydanticAI Web Search Agent using plain model name string."""
    print("Creating Web Search Agent...")
    # *** FIX: Use plain model name string directly from config ***
    plain_model_name = config.WEB_SEARCH_MODEL_NAME
    print(f"  Configuring model: {plain_model_name}")
    agent = Agent(
        model=plain_model_name,
        api_key=config.GOOGLE_API_KEY,
        name="WebSearchAgent",
        description="Web Search Agent: Answers questions using context from web search results via WebSearchCurrentEvents tool.",
        system_prompt=WEB_SEARCH_SYSTEM_PROMPT,
        tools=[web_search_pydantic_tool],
        auto_execute_tools=True
    )
    print("Web Search Agent created.")
    return agent