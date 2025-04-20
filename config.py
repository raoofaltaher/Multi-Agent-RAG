# config.py
import os
from dotenv import load_dotenv

# *** DEBUGGING STEP: Force override and check values ***
print("--- Loading .env file ---")
# Force overriding any system environment variables with the ones from .env
loaded = load_dotenv(override=True, verbose=True)
print(f".env file loaded: {loaded}") # verbose=True prints files searched/loaded

# Retrieve the key using os.getenv
google_api_key_from_env = os.getenv("GOOGLE_API_KEY")

# Explicitly print the loaded value (or lack thereof)
# Be cautious about printing the full key in shared logs, but necessary for debugging now.
# Mask part of it for slightly better security in logs.
if google_api_key_from_env:
    masked_key = google_api_key_from_env[:5] + '...' + google_api_key_from_env[-4:] if len(google_api_key_from_env) > 9 else google_api_key_from_env
    print(f"GOOGLE_API_KEY loaded: {masked_key} (Length: {len(google_api_key_from_env)})")
else:
    print("!!! GOOGLE_API_KEY not found by os.getenv() after load_dotenv() attempt.")
# Assign to the variable used by other modules
GOOGLE_API_KEY = google_api_key_from_env


# Check immediately after loading
if not GOOGLE_API_KEY:
    # This error will stop execution if the key wasn't loaded successfully
    raise ValueError("Critical Error: GOOGLE_API_KEY was not loaded from .env or environment.")

print("--- Configuration Constants ---")
# --- Model Names ---
ROUTER_MODEL_NAME = "gemini-1.5-pro-latest"
RAG_MODEL_NAME = "gemini-1.5-flash-latest"
WEB_SEARCH_MODEL_NAME = "gemini-1.5-flash-latest"
EMBEDDING_MODEL_NAME = "models/embedding-001"
print(f"Router Model: {ROUTER_MODEL_NAME}")
print(f"RAG Model: {RAG_MODEL_NAME}")
print(f"Web Search Model: {WEB_SEARCH_MODEL_NAME}")
print(f"Embedding Model: {EMBEDDING_MODEL_NAME}")

# --- Vector Store ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None) # Will be None if not set
COLLECTION_NAME = "agentic_rag_google"
VECTOR_SIZE = 768
METRIC = "COSINE"
print(f"Qdrant URL: {QDRANT_URL}")
print(f"Qdrant Collection: {COLLECTION_NAME}")

# --- Data Pipeline ---
CHUNK_TOKEN_SIZE = 150
CHUNK_OVERLAP = 10
JINA_READER_PREFIX_URL = "https://r.jina.ai/"

# --- Agent Settings ---
RAG_TOP_K = 3
WEB_SEARCH_MAX_RESULTS = 5
print(f"RAG Top K: {RAG_TOP_K}")
print(f"Web Max Results: {WEB_SEARCH_MAX_RESULTS}")
print("--- Config loading finished ---")