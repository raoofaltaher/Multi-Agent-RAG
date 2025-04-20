# vector_store.py
from qdrant_client import QdrantClient, models as qmodels
from qdrant_client.http import models as http_models
from qdrant_client.http.exceptions import UnexpectedResponse
from typing import List, Dict, Any, Optional
import traceback

import config

def get_qdrant_client() -> QdrantClient:
    client = None
    try:
        client = QdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY,
            timeout=60,
        )
        print(f"Qdrant client initialized for URL: {config.QDRANT_URL}")
        return client
    except Exception as e:
         print(f"!!! Error initializing Qdrant client: {e}")
         if isinstance(e, ConnectionRefusedError) or "Connection refused" in str(e):
              print("Hint: Is the Qdrant server running and accessible at the configured URL?")
         raise

def create_collection_if_not_exists(client: QdrantClient):
    collection_name = config.COLLECTION_NAME
    print(f"Checking Qdrant collection: '{collection_name}'")
    try:
        exists = client.collection_exists(collection_name=collection_name)
        if not exists:
            print(f"Collection '{collection_name}' not found. Creating...")
            vector_params = http_models.VectorParams(
                 size=config.VECTOR_SIZE,
                 distance=getattr(http_models.Distance, config.METRIC.upper(), http_models.Distance.COSINE)
            )
            client.create_collection(collection_name=collection_name, vectors_config=vector_params)
            print(f"Collection '{collection_name}' created (Size: {config.VECTOR_SIZE}, Dist: {config.METRIC}).")
        else:
            # Optionally verify parameters of existing collection
            try:
                 collection_info = client.get_collection(collection_name=collection_name)
                 existing_size = collection_info.vectors_config.params.size
                 existing_dist = str(collection_info.vectors_config.params.distance).upper() # Ensure comparison works
                 config_dist = config.METRIC.upper()

                 if existing_size != config.VECTOR_SIZE or existing_dist != config_dist:
                      print(f"!!! WARNING: Existing collection '{collection_name}' params mismatch config!")
                      print(f"    Config: size={config.VECTOR_SIZE}, distance={config_dist}")
                      print(f"    Actual: size={existing_size}, distance={existing_dist}")
                 else:
                      print(f"Collection '{collection_name}' exists and matches config parameters.")
            except Exception as info_e:
                print(f"Warning: Could not verify existing collection parameters: {info_e}")
                print(f"Collection '{collection_name}' exists.")

    except UnexpectedResponse as e:
        print(f"!!! Error during Qdrant collection operation (Status: {e.status_code}): {e.content}")
        raise
    except Exception as e:
        print(f"!!! Error during collection check/creation: {e}")
        traceback.print_exc()
        raise

def upload_embeddings(client: QdrantClient, embeddings: List[List[float]], payloads: List[Dict[str, Any]], ids: Optional[List[int]] = None):
    collection_name = config.COLLECTION_NAME
    if not embeddings or not payloads:
        print("Warning: No data provided to upload_embeddings.")
        return

    num_vectors = len(embeddings)
    print(f"Attempting to upload {num_vectors} vectors to '{collection_name}'...")

    if ids is None:
        print("   Generating sequential IDs...")
        try:
            # Note: exact=False is faster but may lead to slight gaps/overlaps if deletions occurred.
            # exact=True is safer for purely sequential but slower. Defaulting to safer.
            current_count = client.count(collection_name=collection_name, exact=True).count
            start_id = current_count
            print(f"   Current exact count is {current_count}. Starting IDs from {start_id}.")
        except Exception as count_e:
             print(f"Warning: Could not get exact collection count ({count_e}). Starting IDs from 0.")
             start_id = 0
        ids = list(range(start_id, start_id + num_vectors))
    elif len(ids) != num_vectors:
         raise ValueError(f"ID list length ({len(ids)}) != vector list length ({num_vectors}).")
    else:
         print(f"   Using provided IDs starting from {ids[0]}.")

    print(f"   Preparing {num_vectors} PointStructs...")
    points_to_upload = [
        http_models.PointStruct(id=idx, vector=vector, payload=payload)
        for idx, vector, payload in zip(ids, embeddings, payloads)
    ]

    if not points_to_upload:
        print("Error: No points were prepared for upload.")
        return

    print(f"   Executing upsert operation...")
    try:
        # Upsert can handle create or update
        client.upsert(collection_name=collection_name, points=points_to_upload, wait=True)
        print(f"Successfully upserted {len(points_to_upload)} points.")
    except UnexpectedResponse as e:
        print(f"!!! Error during Qdrant upsert (Status: {e.status_code}): {e.content}")
        raise
    except Exception as e:
        print(f"!!! Error uploading data to Qdrant: {e}")
        traceback.print_exc()
        raise

def search_vectors(client: QdrantClient, query_embedding: List[float], top_k: int = config.RAG_TOP_K) -> List[qmodels.ScoredPoint]:
    collection_name = config.COLLECTION_NAME
    if not query_embedding:
        print("Error: Cannot search with empty query embedding.")
        return []
    print(f"Searching '{collection_name}' (top_k={top_k})...")
    try:
        search_result = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            query_filter=None, # Add filters later if needed
            limit=top_k,
        )
        print(f"Search returned {len(search_result)} results.")
        return search_result
    except UnexpectedResponse as e:
         print(f"!!! Error during Qdrant search (Status: {e.status_code}): {e.content}")
         return []
    except Exception as e:
        print(f"!!! Error searching Qdrant '{collection_name}': {e}")
        traceback.print_exc()
        return []