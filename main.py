# main.py
import asyncio
import sys
from pydantic_ai import Agent
try:
    from IPython import get_ipython
    from IPython.display import display, Markdown
    ipython_available = get_ipython() is not None
except ImportError:
    ipython_available = False
import traceback

import agents
import vector_store
import tools
import config

async def main():
    print("--- Initializing Multi-Agent RAG System ---")

    # Pre-check for API Key needed by agents
    if not config.GOOGLE_API_KEY:
         print("!!! FATAL ERROR: GOOGLE_API_KEY is not set in environment or .env file.")
         print("!!! Please ensure the key is correctly set and try again.")
         return # Exit if no API key

    # Initialize Qdrant Client
    qdrant_client = None
    try:
        print("Initializing Qdrant Client...")
        qdrant_client = vector_store.get_qdrant_client()
        vector_store.create_collection_if_not_exists(qdrant_client)
        print("Qdrant client initialized and collection verified.")
    except Exception as e:
        print(f"!!! Fatal Error: Could not connect to or verify Qdrant. Exiting. Error: {e}")
        traceback.print_exc()
        return

    # Create Agents (will fail if API key is invalid even if set)
    try:
        print("Creating Agents...")
        router_agent = agents.create_router_agent()
        rag_agent = agents.create_rag_agent()
        web_search_agent = agents.create_web_search_agent()
        print("All agents initialized.")
    except Exception as e:
        print(f"!!! Fatal Error: Could not create agents. Error: {e}")
        print("    (This often happens if the GOOGLE_API_KEY is invalid or lacks permissions)")
        traceback.print_exc()
        return

    print("\n--- System Ready ---")
    print("Enter query (or 'quit' / Ctrl+C to exit):")

    while True:
        try:
            query = input("> ")
            query = query.strip() # Remove leading/trailing whitespace
            if query.lower() == 'quit':
                break
            if not query: continue # Skip empty input

            print(f"\n--- Processing: '{query}' ---")

            # 1. Route the query
            print("-> Routing Query...")
            decision = agents.RoutingDecision(vector_search=False, web_search=False)
            try:
                router_agent.auto_execute_tools = False
                routing_result = await router_agent.run(query)
                if routing_result and isinstance(routing_result.data, agents.RoutingDecision):
                    decision = routing_result.data
                    if decision.vector_search and decision.web_search:
                        print("   Warning: Router suggested both; defaulting to vector search.")
                        decision.web_search = False
                else:
                    print(f"   Warning: Router gave unexpected output: {routing_result}. Defaulting to no search.")
                print(f"   Decision: Vector={decision.vector_search}, Web={decision.web_search}")
            except Exception as e:
                 print(f"!!! Error during routing: {e}. Defaulting to no search.")
                 traceback.print_exc()

            # 2. Execute based on route
            final_answer = None
            agent_used = "None"
            try:
                if decision.vector_search:
                    agent_used = "RAG Agent (Vector Search)"
                    print(f"-> Executing: {agent_used}")
                    rag_agent.auto_execute_tools = True
                    rag_deps = tools.RagDeps(qdrant_client=qdrant_client)
                    result = await rag_agent.run(query, deps=rag_deps)
                    final_answer = result.data if result else None
                elif decision.web_search:
                    agent_used = "Web Search Agent"
                    print(f"-> Executing: {agent_used}")
                    web_search_agent.auto_execute_tools = True
                    result = await web_search_agent.run(query)
                    final_answer = result.data if result else None
                else: # Direct Answer
                    agent_used = "Direct Answer Agent"
                    print(f"-> Executing: {agent_used}")
                    # Temporarily modify an agent for direct response
                    original_tools = web_search_agent.tools
                    original_prompt = web_search_agent.system_prompt
                    try:
                        web_search_agent.tools = []
                        web_search_agent.auto_execute_tools = False
                        web_search_agent.system_prompt = "As an AI assistant, answer concisely based on general knowledge. Do not mention searching or tools."
                        result = await web_search_agent.run(query)
                        final_answer = result.data if result else None
                    finally: # Ensure restoration
                        web_search_agent.tools = original_tools
                        web_search_agent.system_prompt = original_prompt
            except Exception as e:
                 print(f"!!! Error during execution by {agent_used}: {e}")
                 traceback.print_exc()
                 final_answer = f"Error processing request via {agent_used}. Please check logs."

            # 3. Display result
            print("\n--- Final Answer ---")
            if final_answer:
                answer_text = str(final_answer).strip()
                if ipython_available:
                    try: display(Markdown(answer_text))
                    except Exception: print(answer_text) # Fallback
                else: print(answer_text)
            else: print(f"<{agent_used} could not generate a response.>")

            print("\n" + "="*70 + "\n") # End of interaction separator

        except KeyboardInterrupt: break # Clean exit on Ctrl+C
        except EOFError: break # Clean exit on Ctrl+D
        except Exception as e: # Catch unexpected loop errors
            print(f"\n!!! Unhandled error in main loop: {e}")
            traceback.print_exc()
            print("   Trying to continue...")

    print("\n--- Multi-Agent RAG System Shutting Down ---")

if __name__ == "__main__":
    asyncio.run(main())