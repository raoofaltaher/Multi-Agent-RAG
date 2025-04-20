# tools.py
import asyncio
from pydantic import BaseModel, Field
from pydantic_ai import RunContext, Tool
# Use models import style compatible with qdrant-client>=1.7
from qdrant_client import QdrantClient, models as qmodels # Use alias to avoid name clash
from qdrant_client.http import models as http_models # For PointStruct etc if needed
# *** FIX: Only import the synchronous DDGS class ***
from duckduckgo_search import DDGS
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import traceback # For error reporting

import data_pipeline
import vector_store
import config

# Define a dependency structure expected by the RAG agent/tool
@dataclass
class RagDeps:
    qdrant_client: QdrantClient # The tool needs access to the client

async def vector_search_tool(ctx: RunContext[RagDeps], query: str) -> str:
    """
    Performs semantic vector search for a query within the internal knowledge base (Qdrant).
    This tool is automatically called by the RAG Agent when needed.

    Args:
        query (str): The natural language query to search for.

    Returns:
        str: A string containing the formatted content of the most relevant documents found,
             or a message indicating no relevant documents were found or an error occurred.
    """
    print(f"--- Executing Vector Search Tool ---")
    print(f"   Query: '{query}'")
    if not ctx.deps or not ctx.deps.qdrant_client:
         print("!!! Error in vector_search_tool: Qdrant client dependency not found in context.")
         return "Error: Vector search cannot be performed due to missing client setup."

    client = ctx.deps.qdrant_client

    try:
        # 1. Get query embedding
        print(f"   Generating query embedding using: {config.EMBEDDING_MODEL_NAME}")
        query_embedding = data_pipeline.get_google_embeddings([query], task_type="retrieval_query")
        if not query_embedding:
             print("!!! Error: Failed to generate query embedding.")
             return "Error: Could not generate embedding for the search query."
        query_vector = query_embedding[0]
        print(f"   Query embedding generated (dimension: {len(query_vector)}).")


        # 2. Search Qdrant
        print(f"   Searching Qdrant collection '{config.COLLECTION_NAME}' (top_k={config.RAG_TOP_K})...")
        search_results: List[qmodels.ScoredPoint] = vector_store.search_vectors(
            client=client,
            query_embedding=query_vector,
            top_k=config.RAG_TOP_K
        )
        print(f"   Qdrant search returned {len(search_results)} results.")

        # 3. Format and return results
        if not search_results:
            print("   No relevant documents found in knowledge base.")
            return "No relevant documents found in the knowledge base for this query."

        context_pieces = []
        for i, doc in enumerate(search_results):
            if doc.payload and "content" in doc.payload:
                 piece = f"Document {i+1} (Score: {doc.score:.4f}):\n{doc.payload['content']}"
                 context_pieces.append(piece)
            else:
                 print(f"   Warning: Document {i+1} (ID: {doc.id}) has missing/invalid payload.")

        if not context_pieces:
             print("   Relevant documents found, but content could not be extracted from payload.")
             return "Found relevant document references, but could not retrieve their content."

        full_context = "\n\n---\n\n".join(context_pieces)
        print(f"--- Vector Search Tool: Returning context (length: {len(full_context)}) ---")
        return full_context

    except Exception as e:
        print(f"!!! Error occurred within vector_search_tool: {e}")
        traceback.print_exc()
        return f"An unexpected error occurred during vector search: {e}"


async def web_search_tool(ctx: RunContext[None], query: str) -> str:
    """
    Performs a web search for the given query using DuckDuckGo's synchronous API
    run inside an asynchronous executor thread.
    This tool is automatically called by the Web Search Agent when needed.

    Args:
        query (str): The search query string.

    Returns:
        str: A string containing formatted snippets from the web search results,
             or a message indicating no results were found or an error occurred.
    """
    print(f"--- Executing Web Search Tool ---")
    print(f"   Query: '{query}'")
    try:
        # *** FIX: Use synchronous DDGS in an executor thread ***
        loop = asyncio.get_running_loop()
        print(f"   Performing web search in executor (max_results={config.WEB_SEARCH_MAX_RESULTS})...")
        # Run the synchronous DDGS().text() function in the default executor
        results = await loop.run_in_executor(
            None, # Use default ThreadPoolExecutor
            lambda: DDGS(timeout=20).text(query, max_results=config.WEB_SEARCH_MAX_RESULTS)
        )
        print(f"   Web search returned {len(results)} results.")


        if not results:
            print("   No results found on the web for this query.")
            return "No relevant results found on the web for this query."

        # Format results: Include title and body if available
        context_pieces = []
        for i, doc in enumerate(results):
             title = doc.get('title', 'No Title')
             body = doc.get('body', 'No snippet available.')
             url = doc.get('href', 'No URL')
             piece = f"Result {i+1}: {title}\nURL: {url}\nSnippet: {body}"
             context_pieces.append(piece)

        full_context = "\n\n---\n\n".join(context_pieces)
        print(f"--- Web Search Tool: Returning context (length: {len(full_context)}) ---")
        return full_context

    except Exception as e:
        print(f"!!! Error occurred within web_search_tool: {e}")
        traceback.print_exc()
        return f"An unexpected error occurred during web search: {e}"

# --- Tool Instances for Agents ---
vector_search_pydantic_tool = Tool(vector_search_tool, name="VectorSearchKnowledgeBase")
web_search_pydantic_tool = Tool(web_search_tool, name="WebSearchCurrentEvents")