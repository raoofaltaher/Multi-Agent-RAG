# data_pipeline.py
import requests
import re
import google.generativeai as genai
from typing import Optional, List
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config

# Configure genai ONCE if key exists, otherwise functions will check
if config.GOOGLE_API_KEY:
    try:
        genai.configure(api_key=config.GOOGLE_API_KEY)
        print("Google Generative AI client configured for embeddings.")
    except Exception as e:
        print(f"Warning: Initial google.generativeai configuration failed: {e}")
else:
    print("Warning: GOOGLE_API_KEY not found in config. Embeddings will fail.")


def fetch_url_content(url: str) -> Optional[str]:
    prefix = config.JINA_READER_PREFIX_URL.rstrip('/') + '/'
    url_path = url.lstrip('/')
    full_url = prefix + url_path

    headers = {
        "Accept": "text/plain",
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    try:
        print(f"Fetching content from: {url} using Jina reader url: {full_url}...")
        response = requests.get(full_url, headers=headers, timeout=60)
        response.raise_for_status()
        print("Content fetched successfully.")
        content_type = response.headers.get('Content-Type', '').lower()
        # Relax check to accept more text-like types from Jina
        if 'text' in content_type:
            return response.text # Return raw text
        else:
            print(f"Warning: Received potentially non-text content type: {content_type}. Trying decode.")
            return response.content.decode('utf-8', errors='ignore')

    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {full_url}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during fetching {full_url}: {e}")
        return None

def clean_text(text: str) -> str:
    if not isinstance(text, str): return ""
    text = re.sub(r' [ \t]+', ' ', text)
    text = re.sub(r'\n[\n\t ]+', '\n\n', text)
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    return text.strip()

def split_text(text: str) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4",
        chunk_size=config.CHUNK_TOKEN_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    cleaned_text = clean_text(text)
    chunks = text_splitter.split_text(cleaned_text)
    print(f"Split text into {len(chunks)} chunks (approx. {config.CHUNK_TOKEN_SIZE} tokens each).")
    return chunks

def get_google_embeddings(texts: List[str], task_type="retrieval_document") -> List[List[float]]:
    if not texts:
        print("Warning: No texts provided to get_google_embeddings.")
        return []
    if not config.GOOGLE_API_KEY:
         raise ValueError("Google API Key not configured")

    # Ensure genai is configured *before* calling embed_content
    # This might be redundant if configured at module level, but safer
    try:
        genai.configure(api_key=config.GOOGLE_API_KEY)
    except Exception as e:
        raise ConnectionError(f"Failed to configure genai for embeddings: {e}") from e

    try:
        print(f"Generating embeddings for {len(texts)} text chunks using '{config.EMBEDDING_MODEL_NAME}' (task: {task_type})...")
        result = genai.embed_content(
            model=config.EMBEDDING_MODEL_NAME,
            content=texts,
            task_type=task_type
        )

        if 'embedding' not in result or not isinstance(result['embedding'], list):
            raise ValueError(f"Embedding API did not return expected structure. Result: {result}")

        embeddings = result['embedding']
        if len(embeddings) != len(texts):
             print(f"Warning: Mismatch counts! Input texts ({len(texts)}), received embeddings ({len(embeddings)}).")

        # Temporarily disable dimension check as it adds less value than fixing core functionality
        # if embeddings and len(embeddings[0]) != config.VECTOR_SIZE:
        #      print(f"Warning: Embedding dimension mismatch! Expected {config.VECTOR_SIZE}, got {len(embeddings[0])}.")

        print(f"Embeddings generated successfully ({len(embeddings)} vectors).")
        return embeddings

    except Exception as e:
        # Catch specific API errors if possible in the future (using error codes/reasons)
        print(f"!!! Error generating Google embeddings via genai.embed_content: {e}")
        # Consider how to handle this - retries? fallback? For now, just re-raise
        raise # Re-raise the exception for upstream handling