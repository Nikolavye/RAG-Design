#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
from dotenv import load_dotenv
from pdf_image_extractor import extract_images_from_pdf, copy_images_to_server

def main():
    # Load environment variables from .env file
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
    
    parser = argparse.ArgumentParser(description='Process PDF files, extract images, and create RAGFlow knowledge base')
    parser.add_argument('pdf_path', help='Path to the PDF file')
    parser.add_argument('--api_key', help='RAGFlow API key (optional, defaults to environment variable)')
    parser.add_argument('--image_dir', default='../data/images', help='Local directory for storing images')
    parser.add_argument('--skip_ragflow', action='store_true', help='Skip RAGFlow resource creation, only process images')
    
    args = parser.parse_args()
    
    # Ensure local image directory exists
    os.makedirs(args.image_dir, exist_ok=True)
    
    # Prioritize command line API key, then environment variable
    api_key = args.api_key or os.getenv('RAGFLOW_API_KEY')

    # Get PDF filename for MinIO bucket naming
    pdf_filename = os.path.basename(args.pdf_path)
    print(f"Processing PDF file: {pdf_filename}")
    
    try:
        # Step 1: Extract images and page documents
        print(f"Step 1: Processing PDF and extracting images...")
        
        # Special handling: if it's the excavator case, use a custom bucket name
        custom_bucket_name = None
        if "excavator" in pdf_filename.lower():
            custom_bucket_name = "ragflow-excavator-repair"
            print(f"Excavator case detected, using custom bucket name: {custom_bucket_name}")
            
        page_documents, extracted_images = extract_images_from_pdf(
            args.pdf_path, 
            pdf_filename=pdf_filename, 
            custom_bucket_name=custom_bucket_name
        )
        
        # Step 2: Verification of uploaded images
        print(f"Step 2: Verifying image uploads...")
        copy_images_to_server(extracted_images, args.image_dir)
        
        # Step 3: Save each page as an independent Markdown file
        print(f"Step 3: Generating independent Markdown files for each page...")
        page_files = []
        base_name = os.path.splitext(os.path.basename(args.pdf_path))[0]
        
        pages_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'pages')
        os.makedirs(pages_dir, exist_ok=True)
        for page_doc in page_documents:
            page_filename = os.path.join(pages_dir, f"{base_name}_page{page_doc['page']}.md")
            with open(page_filename, "w", encoding="utf-8") as f:
                f.write(page_doc['content'])
            page_files.append(page_filename)
            print(f"  - Saved: {page_filename}")

        print(f"Step 4: Generated {len(page_files)} page files")
        
        # Skip RAGFlow if requested
        if args.skip_ragflow:
            print("Skipped RAGFlow resource creation")
            print("\nProcessing complete!")
            print(f"- Images mapped to MinIO")
            print(f"- Page files saved: {', '.join(page_files)}")
            return
        
        # Check API key
        if not api_key:
            print("Error: No RAGFlow API key provided. Set RAGFLOW_API_KEY in .env or use --api_key")
            return
        
        print(f"Step 5: Creating RAGFlow knowledge base and assistant...")
        
        from ragflow_kb_manager import create_ragflow_resources_multi_docs
        
        custom_dataset_name = None
        custom_assistant_name = None
        
        if "excavator" in pdf_filename.lower():
            custom_dataset_name = "Excavator Repair Assistant (Rich Text Enhanced)"
            custom_assistant_name = "Excavator Repair Assistant (Rich Text Enhanced)"
            print(f"Using custom KB/Assistant name: {custom_assistant_name}")

        dataset, assistant = create_ragflow_resources_multi_docs(
            page_documents,
            page_files,
            args.pdf_path,
            api_key,
            base_url=os.getenv('RAGFLOW_BASE_URL', 'http://localhost:8080'),
            custom_dataset_name=custom_dataset_name,
            custom_assistant_name=custom_assistant_name
        )
        
        print(f"\nProcessing complete!")
        if dataset and assistant:
            print(f"- Knowledge Base ID: {dataset.id}")
            print(f"- Chat Assistant ID: {assistant.id}")
        else:
            print("- Failed to create RAGFlow resources")

        print("\nImages have been uploaded to MinIO and set to public access.")
        
    except Exception as e:
        print(f"An error occurred during processing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
