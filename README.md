# 🏭 RAGFlow Equipment Maintenance Multimodal Knowledge Base (MinIO Integrated)

A multimodal knowledge base solution based on **MinIO** object storage and **RAGFlow**, specifically designed for handling equipment maintenance manuals and operation guides containing large amounts of images.

## 📖 Project Background

In scenarios such as industrial maintenance and equipment operation manuals, documents are often rich in images (e.g., circuit diagrams, exploded views). Traditional text-only RAG often loses critical visual information, resulting in lackluster answers.

This project solves this problem through a **"Image-Link Decoupling"** architecture:
1. **Auto Extract**: Automatically extract images from PDFs and store them in MinIO.
2. **Replacement**: Replace original images with public access URLs from MinIO.
3. **Construction**: Build "Enhanced Markdown" containing these URLs and upload to RAGFlow.

The result: **When the LLM answers questions, it can directly invoke and display the original document images, achieving true "Image-Text Mixed QA".**

---

## 🏗️ System Architecture

### 🔄 Data Flow and Core Modules

This system acts as a **Middleware** between raw documents and the RAGFlow engine.

```mermaid
graph TD
    classDef file fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef process fill:#fff9c4,stroke:#fbc02d,stroke-width:2px;
    classDef db fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef ai fill:#f3e5f5,stroke:#4a148c,stroke-width:2px,color:black;

    subgraph "🚀 Core ETL Pipeline"
        direction TB
        RawPDF[("📄 Raw PDF Document\n(Incl. images/drawings)")]:::file
        
        subgraph "1. Extraction"
            Extractor["🔨 pdf_image_extractor.py\nextract_images_from_pdf()"]:::process
            ImgObj["🖼️ Image Objects"]:::file
            TextObj["📝 Raw Text Content"]:::file
        end
        
        subgraph "2. Storage & Linking"
            MinIOClient["☁️ minio_client.py\nupload_image_bytes()"]:::process
            MinIO[("️ MinIO Object Storage\n(Hosted images)")]:::db
            
            ImgLink["🔗 Public Image URL\n(http://minio:9000/...)"]:::file
        end
        
        subgraph "3. Reconstruction"
            Merger["🔄 Text Enhancement\n(Insert URLs into Markdown)"]:::process
            EnhancedDoc["📑 Enhanced Markdown\n(Mixed Text & Image URLs)")]:::file
        end

        subgraph "4. Ingestion"
            KBManager["🤖 ragflow_kb_manager.py\ncreate_ragflow_resources..."]:::process
            RAGFlow[("🧠 RAGFlow Engine\n(Vector Store + Assistant)")]:::ai
        end

        RawPDF --> Extractor
        Extractor --> ImgObj
        Extractor --> TextObj
        
        ImgObj --> MinIOClient
        MinIOClient --> MinIO
        MinIO --> ImgLink
        
        TextObj --> Merger
        ImgLink --> Merger
        Merger --> EnhancedDoc
        
        EnhancedDoc --> KBManager
        KBManager --> RAGFlow
    end
```

---

## 💻 Code Reading Guide

To understand how the system works, suggested reading order:

### 1. Infrastructure Layer: `minio_client.py`
- **Responsibility**: Interaction with MinIO object storage.
- **Key Design**: `create_bucket_if_not_exists` automatically sets a `public-read` policy. This ensures RAGFlow's web UI can render images directly via HTTP GET without complex signature handling.

### 2. Extraction Layer: `pdf_image_extractor.py`
- **Responsibility**: PDF parsing factory.
- **Key Logic**: Uses `PyMuPDF (fitz)` for dual-path extraction (text and images). Images are renamed with semantic identifiers (e.g., `manual_page3_img1.png`) for better traceability.

### 3. Orchestration Layer: `ragflow_pdf_processor.py`
- **Responsibility**: Main entry point (Controller).
- **Flow**: Initializes environment -> Calls extractor -> Uploads images -> Generates Markdown -> Calls KB Manager for ingestion.

### 4. Automation Layer: `ragflow_kb_manager.py`
- **Responsibility**: Automates RAGFlow resource creation using the Python SDK.
- **Key Strategy**: Uses a **Clean & Build** approach (recreates KB on each run) and "Naive" chunking to respect pre-split pages. Also injects a specific System Prompt to guide the LLM in rendering Markdown images.

---

## 🚀 Getting Started

### Step 1: Environment Setup
Ensure the following services are running:
1. **RAGFlow** Server
2. **MinIO** Server
3. **Python 3.8+**

### Step 2: Configuration
1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Mac/Linux
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure environment variables in `.env`:
   - `RAGFLOW_API_KEY`, `MINIO_ENDPOINT`, etc.

### Step 3: Run the Processor
Use the provided sample or your own PDF:
```bash
python ragflow_pdf_processor.py excavator_repair_case.pdf
```

---

## 💡 Engineering Design Insights

### Why MinIO instead of Base64?
- **Performance**: Base64 increases token consumption by ~33%, wasting the LLM's context window.
- **Standardization**: URLs are a universal standard that any Markdown renderer supports.
- **Decoupling**: Offloads storage pressure from the compute engine (RAGFlow).

### Why split Markdown by page?
- **Precision**: PDF physical pages often align with natural semantic boundaries. Treating each page as a document helps retrieval engines locate specific knowledge accurately.
- **Image Context**: Images are usually closely related to the text on the same page.

---

## ⚠️ Prompt Engineering Note

When designing the RAGFlow System Prompt, **must include the `{knowledge}` placeholder**.
- **What is it?** It's the slot where RAGFlow injects retrieved knowledge.
- **Without it?** The LLM won't receive the retrieved context and will suffer from hallucinations or generic answers.
- **Correct Usage**:
  ```markdown
  You are an assistant...
  
  Here is the knowledge base:
  {knowledge}    <--- Keep this!
  The above is the knowledge base.
  ```
