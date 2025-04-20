# ingest_data.py
import sys
import asyncio
import os
import traceback

import config
import data_pipeline
import vector_store

async def ingest(url: str):
    print(f"--- Starting ingestion process for URL: {url} ---")

    # 1. Fetch Content
    print("\nStep 1: Fetching content...")
    content = data_pipeline.fetch_url_content(url)
    if not content:
        print("!!! Error: Failed to fetch content. Aborting ingestion.")
        return
    print(f"Content fetched (length: {len(content)} characters).")

    # 2. Split Text
    print("\nStep 2: Splitting text into chunks...")
    text_chunks = data_pipeline.split_text(content)
    if not text_chunks:
        print("!!! Error: No text chunks generated. Aborting ingestion.")
        return
    print(f"Text split into {len(text_chunks)} chunks.")

    # 3. Get Embeddings (Ensure API Key is valid before this step!)
    print("\nStep 3: Generating embeddings...")
    if not config.GOOGLE_API_KEY:
        print("!!! Error: GOOGLE_API_KEY is missing in config. Aborting embedding step.")
        return # Stop before calling API without key
    embeddings = []
    try:
        embeddings = data_pipeline.get_google_embeddings(text_chunks, task_type="retrieval_document")
        if not embeddings or len(embeddings) != len(text_chunks):
             print(f"!!! Error/Warning: Embedding generation mismatch or failure. Expected {len(text_chunks)}, got {len(embeddings)}. Aborting.")
             return
        print(f"Successfully generated {len(embeddings)} embeddings.")

    except Exception as e:
        print(f"!!! Error: Failed during embedding generation: {e}. Aborting ingestion.")
        # No need to print traceback here, get_google_embeddings already does/should
        return

    # 4. Prepare Payloads
    print("\nStep 4: Preparing payloads...")
    payloads = [{"url_source": url, "content": chunk} for chunk in text_chunks]
    # Let upload_embeddings generate IDs
    ids = None
    print(f"Prepared {len(payloads)} payloads. IDs will be generated sequentially.")

    # 5. Setup Vector Store and Upload
    print("\nStep 5: Connecting to Vector Store and Uploading...")
    client = None
    try:
        client = vector_store.get_qdrant_client()
        vector_store.create_collection_if_not_exists(client)
        vector_store.upload_embeddings(client, embeddings, payloads, ids) # Pass None for IDs
        print(f"\n--- Ingestion completed successfully for {url}. ---")

        try:
            count_result = client.count(collection_name=config.COLLECTION_NAME, exact=True)
            print(f"Collection '{config.COLLECTION_NAME}' now contains {count_result.count} vectors.")
        except Exception as count_e:
            print(f"Warning: Could not verify exact final count in collection: {count_e}")

    except Exception as e:
        print(f"!!! Error during vector store setup or upload: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    default_url = "https://ai.google.dev/gemini-api/docs/models/gemini" 
    url_to_ingest = sys.argv[1] if len(sys.argv) > 1 else default_url

    if not (url_to_ingest.startswith("http://") or url_to_ingest.startswith("https://")):
        print(f"Error: Invalid URL: '{url_to_ingest}'. Must start with http:// or https://.")
    else:
        try:
            print(f"Running ingestion for: {url_to_ingest}")
            asyncio.run(ingest(url_to_ingest))
        except Exception as e:
             print(f"!!! An unexpected error occurred running the ingest script: {e}")
             traceback.print_exc()