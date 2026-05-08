from ragflow_sdk import RAGFlow
import os
import time
import requests

DEFAULT_SIMILARITY_THRESHOLD = float(os.getenv("RAGFLOW_SIMILARITY_THRESHOLD", "0.15"))
DEFAULT_VECTOR_WEIGHT = float(os.getenv("RAGFLOW_VECTOR_WEIGHT", "0.25"))
DEFAULT_KEYWORD_WEIGHT = max(0.0, min(1.0, 1 - DEFAULT_VECTOR_WEIGHT))
DEFAULT_RERANK_MODEL = os.getenv("RAGFLOW_RERANK_MODEL", "").strip()
DEFAULT_CHAT_TOP_N = int(os.getenv("RAGFLOW_CHAT_TOP_N", "4"))
DEFAULT_CHAT_TOP_K = int(os.getenv("RAGFLOW_CHAT_TOP_K", "32"))

def create_ragflow_resources_multi_docs(page_documents, page_files, pdf_filename, api_key, base_url="http://localhost:8080", custom_dataset_name=None, custom_assistant_name=None):
    """
    Create RAGFlow knowledge base and chat assistant using multiple independent page documents.

    Parameters:
    - page_documents: List of page documents, each containing page, content, title
    - page_files: List of page file paths
    - pdf_filename: PDF filename
    - api_key: RAGFlow API key
    - base_url: RAGFlow base URL
    - custom_dataset_name: Custom knowledge base name (optional)
    - custom_assistant_name: Custom assistant name (optional)
    """
    print(f"Creating RAGFlow resources using {len(page_documents)} independent page documents")

    try:
        # Initialize RAGFlow client
        rag_object = RAGFlow(api_key=api_key, base_url=base_url)

        # Create dataset name
        if custom_dataset_name:
            dataset_name = custom_dataset_name
        else:
            base_name = os.path.splitext(os.path.basename(pdf_filename))[0]
            dataset_name = f"{base_name}_kb"

        print(f"Creating multi-document dataset: {dataset_name}")

        # Delete existing knowledge base and assistant if they exist
        try:
            existing_datasets = rag_object.list_datasets()
            dataset_ids_to_delete = []
            for ds in existing_datasets:
                if ds.name == dataset_name:
                    print(f"Found existing knowledge base '{dataset_name}', deleting...")
                    dataset_ids_to_delete.append(ds.id)

            if dataset_ids_to_delete:
                rag_object.delete_datasets(dataset_ids_to_delete)
                print("Deleted old knowledge base")

            if custom_assistant_name:
                assistant_name = custom_assistant_name
            else:
                base_name = os.path.splitext(os.path.basename(pdf_filename))[0]
                assistant_name = f"{base_name}_assistant"
            
            existing_chats = rag_object.list_chats()
            chat_ids_to_delete = []
            for chat in existing_chats:
                if chat.name == assistant_name:
                    print(f"Found existing assistant '{assistant_name}', deleting...")
                    chat_ids_to_delete.append(chat.id)

            if chat_ids_to_delete:
                rag_object.delete_chats(chat_ids_to_delete)
                print("Deleted old assistant")

        except Exception as cleanup_e:
            print(f"Error during resource cleanup: {str(cleanup_e)}")
            print("Proceeding to create new resources...")

        # Create dataset
        embedding_model = os.getenv("RAGFLOW_EMBEDDING_MODEL", "gemini-embedding-001@Gemini")

        dataset = rag_object.create_dataset(
            name=dataset_name,
            description="Maintenance cases indexed as one complete case per page",
            embedding_model=embedding_model,
            chunk_method="one"
        )
        print(f"Dataset '{dataset_name}' created successfully")
        print(f"Using Embedding model: {embedding_model}")

        print("Configuring page-as-case chunking strategy...")
        # Each extracted page already represents a complete maintenance case.
        # Use one-chunk-per-document so retrieval returns the full case directly.
        update_resp = requests.post(
            f"{base_url}/v1/kb/update",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "kb_id": dataset.id,
                "name": dataset_name,
                "description": "Maintenance cases indexed as one complete case per page",
                "permission": "me",
                "language": "English",
                "parser_id": "one",
                "pipeline_id": "",
                "embd_id": embedding_model,
                "parser_config": {
                    "chunk_token_num": 2048,      # One page ~= one case, keep it as a single chunk
                    "delimiter": "\n\n",
                    "enable_children": False,
                    "html4excel": False,
                    "layout_recognize": "Naive",
                    "auto_keywords": 0,
                    "auto_questions": 0,
                    "overlapped_percent": 0,
                    "raptor": {"use_raptor": False},
                    "graphrag": {"use_graphrag": False}
                }
            }
        )
        update_data = update_resp.json()
        if update_data.get("code") != 0:
            print(f"Warning: Failed to configure page-as-case chunking: {update_data.get('message')}")
        else:
            print("Page-as-case chunking configured: each uploaded page should stay as one complete case")

        # Prepare page documents for batch upload
        docs_to_upload = []
        metadata_by_doc_name = {}
        for page_doc, page_file in zip(page_documents, page_files):
            print(f"Preparing to upload page {page_doc['page']} document: {page_file}")
            encoded_text = page_doc['content'].encode('utf-8')
            display_name = os.path.basename(page_file)
            metadata_by_doc_name[display_name] = page_doc.get("metadata", {})
            docs_to_upload.append({
                # RAGFlow parses this as the document name; using the basename avoids
                # downstream "embedding extraction from file path" errors.
                "display_name": display_name,
                "blob": encoded_text
            })

        # Batch upload
        print(f"Uploading {len(docs_to_upload)} page documents...")
        dataset.upload_documents(docs_to_upload)
        print("All page documents uploaded successfully")

        # Parsing documents
        print("Starting document parsing...")
        docs = dataset.list_documents()
        doc_ids = [doc.id for doc in docs if hasattr(doc, 'id')]
        
        if doc_ids:
            dataset.async_parse_documents(doc_ids)
            print("Parsing initiated...")

            # Wait for completion
            all_done = False
            max_wait_time = 300
            start_time = time.time()

            while not all_done and (time.time() - start_time) < max_wait_time:
                all_done = True
                for doc_id in doc_ids:
                    docs_check = dataset.list_documents(id=doc_id)
                    if docs_check and len(docs_check) > 0:
                        doc_status = docs_check[0].run
                        print(f"Document {doc_id} status: {doc_status}")
                        if doc_status != "DONE":
                            all_done = False
                    else:
                        all_done = False

                if not all_done:
                    print("Parsing in progress, waiting 10 seconds...")
                    time.sleep(10)

            print("Document parsing completed!")

        # Write structured case metadata back to each RAGFlow document so the
        # dataset can support exact filtering on case identifiers and machine attributes.
        docs_with_metadata = dataset.list_documents()
        for doc in docs_with_metadata:
            metadata = metadata_by_doc_name.get(doc.name, {})
            if not metadata:
                continue
            print(f"Applying metadata to {doc.name}: {sorted(metadata.keys())}")
            doc.update({"meta_fields": metadata})

        # Create Chat Assistant
        if custom_assistant_name:
            assistant_name = custom_assistant_name
        else:
            base_name = os.path.splitext(os.path.basename(pdf_filename))[0]
            assistant_name = f"{base_name}_assistant"
        
        print(f"Creating chat assistant: {assistant_name}")

        assistant = rag_object.create_chat(
            name=assistant_name,
            dataset_ids=[dataset.id]
        )

        prompt_template = """
# Excavator Maintenance Expert

You are an experienced technical expert in excavator maintenance, dedicated to answering excavator repair questions.

## Intent Recognition (First Step — Always Do This)

Before answering, silently assess the current question:
- **Type A — Follow-up**: The question directly continues the previous conversation turn (e.g., asks for more detail, clarification, or next steps about the same fault/case).
- **Type B — New topic**: The question describes a different fault, a different machine, or is unrelated to any previous turn.

**If Type B**: Ignore all prior conversation history. Answer solely from the knowledge base below.
**If Type A**: You may reference the previous turn's context to provide continuity.

This classification is internal — do not mention it in your answer.

## Answer Requirements

### Answer Structure
1. **Problem Identification**: Confirm the fault type and equipment model.
2. **Case Reference**: Cite the case ID, model, and machine number from the matched case.
3. **Diagnostic Steps**: Troubleshooting from simple to complex.
4. **Maintenance Solution**: Specific steps and required parts.
5. **Prevention Suggestions**: How to avoid recurrence.

### Image Link Handling (Critical)
- Output image links exactly as they appear in the knowledge base — no modifications.
- Always use the format: `<img src="http://..." alt="..." width="300">`
- Include images when describing maintenance steps.

### Knowledge Base Boundary
- Answer only from the knowledge base below. Do not speculate or use external knowledge.
- If the knowledge base does not contain the answer, say so clearly.
- **When the retrieved knowledge base content does not match the conversation history topic, the knowledge base takes priority — answer based on it and disregard the history.**
- **Never merge two different faults or two different cases into one answer. If the current matched case differs from the previous conversation topic, answer only for the current matched case.**

Here is the knowledge base:
{knowledge}
The above is the knowledge base.
        """

        prompt_config = {
            "prompt": prompt_template.strip(),
            "show_quote": True,
            "top_n": DEFAULT_CHAT_TOP_N,
            "top_k": DEFAULT_CHAT_TOP_K,
            "similarity_threshold": DEFAULT_SIMILARITY_THRESHOLD,
            # Keep each user turn retrieval-focused. In this maintenance
            # dataset, consecutive short questions often describe a new fault
            # rather than a follow-up on the previous case.
            "refine_multiturn": False,
            # RAGFlow chat assistants persist this field as the vector-side
            # weight despite the SDK name.
            "keywords_similarity_weight": DEFAULT_VECTOR_WEIGHT,
        }
        if DEFAULT_RERANK_MODEL:
            prompt_config["rerank_model"] = DEFAULT_RERANK_MODEL

        assistant_update_resp = requests.put(
            f"{base_url}/api/v1/chats/{assistant.id}",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "name": assistant_name,
                "dataset_ids": [dataset.id],
                "prompt": prompt_config,
            },
            timeout=30,
        )
        assistant_update_data = assistant_update_resp.json()
        if assistant_update_data.get("code") != 0:
            raise RuntimeError(
                "Failed to update assistant configuration: "
                f"{assistant_update_data.get('message')}"
            )

        print(
            "Configured hybrid retrieval defaults: "
            f"keyword_weight={DEFAULT_KEYWORD_WEIGHT:.2f}, "
            f"vector_weight={1 - DEFAULT_KEYWORD_WEIGHT:.2f}, "
            f"similarity_threshold={DEFAULT_SIMILARITY_THRESHOLD:.2f}, "
            f"top_n={DEFAULT_CHAT_TOP_N}, "
            f"rerank_model={DEFAULT_RERANK_MODEL or 'disabled'}"
        )

        print(f"Chat assistant '{assistant_name}' created and configured successfully")

        return dataset, assistant

    except Exception as e:
        print(f"Error creating RAGFlow multi-document resources: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
