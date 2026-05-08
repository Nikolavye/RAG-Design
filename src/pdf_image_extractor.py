import fitz  # PyMuPDF
import os
import re
import uuid
import shutil
import sys
from PIL import Image
import io
from minio_client import get_minio_client, init_minio_bucket

def format_maintenance_page_text(page_text):
    """
    Normalize maintenance-case pages into paragraph-style Markdown blocks.

    Raw PDF extraction tends to split every visual line, which makes RAGFlow
    produce overly fragmented chunks. This formatter rebuilds a page into a few
    semantic sections so parent/child chunking can work at section granularity.
    """
    lines = [line.strip() for line in page_text.splitlines() if line.strip()]
    if not lines:
        return ""

    normalized_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line == "Troubleshooting" and i + 1 < len(lines) and lines[i + 1] == "Steps":
            normalized_lines.append("Troubleshooting Steps")
            i += 2
            continue
        if line == "Prevention /" and i + 1 < len(lines) and lines[i + 1] == "Suggestion":
            normalized_lines.append("Prevention / Suggestion")
            i += 2
            continue
        normalized_lines.append(line)
        i += 1

    section_headers = {
        "Symptom:",
        "Possible Causes",
        "Cause Analysis",
        "Troubleshooting Steps",
        "Maintenance Plan",
        "Effect Confirmation",
        "Prevention / Suggestion",
    }
    machine_headers = ["Model", "Machine No.", "Hours (h)", "Location", "Environment"]

    title = normalized_lines[0]
    cursor = 1
    blocks = [f"## {title}"]

    if normalized_lines[cursor:cursor + len(machine_headers)] == machine_headers:
        cursor += len(machine_headers)
        machine_values = []
        while cursor < len(normalized_lines) and normalized_lines[cursor] not in section_headers:
            machine_values.append(normalized_lines[cursor])
            cursor += 1

        machine_lines = []
        for idx, header in enumerate(machine_headers):
            if idx < len(machine_values) and machine_values[idx]:
                machine_lines.append(f"{header}: {machine_values[idx]}")

        if machine_lines:
            blocks.append("**Machine info:**\n" + "\n".join(machine_lines))

    current_header = None
    current_body = []
    while cursor < len(normalized_lines):
        line = normalized_lines[cursor]
        if line in section_headers:
            if current_header:
                body = " ".join(current_body).strip()
                if body:
                    blocks.append(f"**{current_header}**\n{body}")
            current_header = line
            current_body = []
        else:
            current_body.append(line)
        cursor += 1

    if current_header:
        body = " ".join(current_body).strip()
        if body:
            blocks.append(f"**{current_header}**\n{body}")

    return "\n\n".join(blocks)


def _collapse_whitespace(value):
    return re.sub(r"\s+", " ", value or "").strip()


def _normalize_model(value):
    return re.sub(r"[^A-Z0-9]", "", (value or "").upper())


def _normalize_machine_no(value):
    return re.sub(r"[^A-Z0-9]", "", (value or "").upper())


def _split_environment_and_temperature(value):
    cleaned = _collapse_whitespace(value.strip(" ,;")) if value else ""
    if not cleaned:
        return "", ""

    temp_match = re.search(r"\d+\s*-\s*\d+\s*°?\s*[CF]", cleaned, re.IGNORECASE)
    if temp_match:
        temperature = _collapse_whitespace(temp_match.group(0))
        environment = _collapse_whitespace(
            (cleaned[:temp_match.start()] + cleaned[temp_match.end():]).strip(" ,;/")
        )
        return environment, temperature

    return cleaned, ""


def _normalize_machine_metadata(machine_info):
    hours_raw = _collapse_whitespace(machine_info.get("Hours (h)", ""))
    location_raw = _collapse_whitespace(machine_info.get("Location", ""))
    environment_raw = _collapse_whitespace(machine_info.get("Environment", ""))

    normalized = {
        "hours": "",
        "location": "",
        "environment": "",
        "temperature_range": "",
    }

    numeric_hours = re.search(r"\b\d+(?:\.\d+)?\b", hours_raw)
    has_named_hours = bool(hours_raw and re.search(r"[A-Za-z]{2,}", hours_raw))

    if numeric_hours and not has_named_hours:
        normalized["hours"] = numeric_hours.group(0)
        normalized["location"] = location_raw
        environment, temperature = _split_environment_and_temperature(environment_raw)
        normalized["environment"] = environment
        normalized["temperature_range"] = temperature
        return normalized

    inferred_location_parts = []
    if hours_raw:
        inferred_location_parts.append(hours_raw.strip(" ,"))

    if environment_raw:
        if location_raw:
            inferred_location_parts.append(location_raw.strip(" ,"))
        normalized["location"] = _collapse_whitespace(", ".join(inferred_location_parts))
        normalized["environment"] = environment_raw
        return normalized

    if location_raw:
        environment, temperature = _split_environment_and_temperature(location_raw)
        normalized["location"] = _collapse_whitespace(", ".join(inferred_location_parts))
        if not normalized["location"]:
            normalized["location"] = location_raw
        normalized["environment"] = environment
        normalized["temperature_range"] = temperature
        return normalized

    normalized["location"] = _collapse_whitespace(", ".join(inferred_location_parts))
    return normalized


def extract_case_metadata(markdown_content, page_number=None, source_pdf=None):
    """
    Extract exact-match metadata from a page-level case document.

    These fields are designed for metadata filtering and for preserving
    structured identifiers such as case ID, model, and machine serial number.
    """
    lines = [line.rstrip() for line in markdown_content.splitlines()]
    if not lines:
        return {}

    title_line = ""
    for line in lines:
        if line.startswith("## "):
            title_line = line[3:].strip()
            break

    case_id = ""
    case_title = title_line
    title_match = re.match(r"^(?P<case_id>\d+(?:\.\d+)+)\.?\s*(?P<title>.+)$", title_line)
    if title_match:
        case_id = title_match.group("case_id")
        case_title = title_match.group("title").strip()

    case_title = re.sub(r"^\s*Case Study:\s*", "", case_title, flags=re.IGNORECASE)
    case_title = re.sub(r"\s+\(Page\s+\d+\)\s*$", "", case_title, flags=re.IGNORECASE)
    case_title = _collapse_whitespace(case_title)

    sections = {}
    current_section = None
    current_lines = []

    for line in lines[1:]:
        stripped = line.strip()
        section_match = re.match(r"^\*\*(.+?)\*\*$", stripped)
        if section_match:
            if current_section:
                sections[current_section] = "\n".join(current_lines).strip()
            current_section = section_match.group(1).strip().rstrip(":")
            current_lines = []
            continue

        if stripped.startswith("### "):
            if current_section:
                sections[current_section] = "\n".join(current_lines).strip()
            current_section = None
            current_lines = []
            continue

        if current_section and not stripped.startswith("<img "):
            current_lines.append(stripped)

    if current_section:
        sections[current_section] = "\n".join(current_lines).strip()

    machine_info = {}
    for line in sections.get("Machine info", "").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        machine_info[_collapse_whitespace(key)] = _collapse_whitespace(value)

    normalized_machine = _normalize_machine_metadata(machine_info)

    model = machine_info.get("Model", "")
    machine_no = machine_info.get("Machine No.", "")
    symptom = _collapse_whitespace(sections.get("Symptom", ""))
    maintenance_plan = _collapse_whitespace(sections.get("Maintenance Plan", ""))
    effect_confirmation = _collapse_whitespace(sections.get("Effect Confirmation", ""))
    prevention = _collapse_whitespace(sections.get("Prevention / Suggestion", ""))
    possible_causes = _collapse_whitespace(sections.get("Possible Causes", ""))

    metadata = {
        "case_type": "maintenance_case",
        "chunk_strategy": "page_as_case",
    }

    if page_number is not None:
        metadata["page_number"] = str(page_number)
    if source_pdf:
        metadata["source_pdf"] = os.path.basename(source_pdf)
    if case_id:
        metadata["case_id"] = case_id
    if case_title:
        metadata["case_title"] = case_title
        metadata["fault_name"] = case_title
    if model:
        metadata["model"] = model
        metadata["model_normalized"] = _normalize_model(model)
    if machine_no:
        metadata["machine_no"] = machine_no
        metadata["machine_no_normalized"] = _normalize_machine_no(machine_no)
    if symptom:
        metadata["symptom"] = symptom
    if normalized_machine["hours"]:
        metadata["hours"] = normalized_machine["hours"]
    if normalized_machine["location"]:
        metadata["location"] = normalized_machine["location"]
    if normalized_machine["environment"]:
        metadata["environment"] = normalized_machine["environment"]
    if normalized_machine["temperature_range"]:
        metadata["temperature_range"] = normalized_machine["temperature_range"]
    if maintenance_plan:
        metadata["maintenance_plan"] = maintenance_plan
    if effect_confirmation:
        metadata["effect_confirmation"] = effect_confirmation
    if prevention:
        metadata["prevention_suggestion"] = prevention
    if possible_causes:
        metadata["possible_causes"] = possible_causes

    return metadata


CASE_RETRIEVAL_HINTS = {
    "1.21.4": {
        "case_title_zh": "发动机冒蓝烟",
        "symptom_zh": "发动机冒蓝烟；冒蓝烟；蓝烟；排蓝烟；尾气蓝烟",
        "cause_zh": "喷油器故障；喷油器雾化不良；缸压不足；燃油泵问题；燃油质量差",
        "maintenance_plan_zh": "更换喷油器",
    },
    "1.21.5": {
        "case_title_zh": "怠速剧烈抖动",
        "symptom_zh": "怠速抖动；怠速振动；发动机抖动；怠速发抖",
        "cause_zh": "机脚垫损坏；喷油器问题；气缸失火；固定螺栓松动",
        "maintenance_plan_zh": "更换发动机机脚垫",
    },
    "1.21.6": {
        "case_title_zh": "冷却液高温报警",
        "symptom_zh": "水温高报警；高温报警；冷却液温度高；水温报警",
        "cause_zh": "传感器故障；搭铁不良；线束问题；散热器堵塞；风扇皮带打滑",
        "maintenance_plan_zh": "更换主电源总开关",
    },
    "1.21.7": {
        "case_title_zh": "长时间停放后偶发难启动",
        "symptom_zh": "停久了不好启动；长时间停放后无法启动；偶发启动困难；停放后难启动",
        "cause_zh": "油箱脏污；低压油路堵塞；熄火电磁阀故障；继电器问题；燃油泵故障",
        "maintenance_plan_zh": "清洗油箱",
    },
}


def build_multilingual_retrieval_hints(metadata):
    case_id = metadata.get("case_id", "")
    hint_block = CASE_RETRIEVAL_HINTS.get(case_id, {})
    if not hint_block:
        return "", {}

    hint_lines = []
    extra_metadata = {}

    case_title_zh = hint_block.get("case_title_zh", "").strip()
    if case_title_zh:
        hint_lines.append(f"中文故障名称: {case_title_zh}")
        extra_metadata["case_title_zh"] = case_title_zh

    symptom_zh = hint_block.get("symptom_zh", "").strip()
    if symptom_zh:
        hint_lines.append(f"中文症状别名: {symptom_zh}")
        extra_metadata["symptom_zh"] = symptom_zh

    cause_zh = hint_block.get("cause_zh", "").strip()
    if cause_zh:
        hint_lines.append(f"中文原因别名: {cause_zh}")
        extra_metadata["possible_causes_zh"] = cause_zh

    maintenance_plan_zh = hint_block.get("maintenance_plan_zh", "").strip()
    if maintenance_plan_zh:
        hint_lines.append(f"中文维修方案: {maintenance_plan_zh}")
        extra_metadata["maintenance_plan_zh"] = maintenance_plan_zh

    return "\n".join(hint_lines), extra_metadata

def extract_images_from_pdf(pdf_path, pdf_filename=None, custom_bucket_name=None):
    """
    Extract images and text from PDF, returning enhanced text with image location markers.

    Parameters:
    - pdf_path: Path to the PDF file
    - pdf_filename: PDF filename for creating MinIO bucket (optional)
    - custom_bucket_name: Custom bucket name (optional, prioritized)
    """
    print(f"Processing PDF: {pdf_path}")

    # Initialize MinIO client and bucket
    minio_client = get_minio_client()
    if not minio_client:
        raise ValueError("MinIO client initialization failed. Please check the MinIO configuration in the .env file.")

    # Initialize bucket
    if custom_bucket_name:
        bucket_name, base_url = init_minio_bucket(custom_bucket_name=custom_bucket_name)
    else:
        bucket_name, base_url = init_minio_bucket(pdf_filename or os.path.basename(pdf_path))
    if not bucket_name or not base_url:
        raise ValueError("MinIO bucket initialization failed.")

    print(f"Using MinIO bucket: {bucket_name}")
    print(f"Image base URL: {base_url}")
    
    doc = None
    extracted_text = []
    extracted_images = []
    page_count = 0
    
    try:
        # Open PDF and extract content
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        print(f"PDF opened successfully, total {page_count} pages")
        
        # Extract text and images from each page
        for page_idx in range(page_count):
            page = doc[page_idx]
            print(f"Processing page {page_idx+1}...")
            
            # Get page text
            text = page.get_text()
            extracted_text.append({"page": page_idx+1, "text": text})
            
            # Extract images
            image_list = page.get_images(full=True)
            print(f"Found {len(image_list)} images on page {page_idx+1}")
            
            # Process current page images
            for img_idx, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]

                    # Generate meaningful image name
                    pdf_base_name = os.path.splitext(os.path.basename(pdf_path))[0]
                    safe_name = ''.join(c for c in pdf_base_name if c.isalnum() or c in '_-').strip()
                    if not safe_name:
                        safe_name = 'document'

                    image_filename = f"{safe_name}_page{page_idx+1}_img{img_idx+1}_v1.png"
                    content_type = f"image/{base_image['ext']}"

                    # Upload directly to MinIO
                    image_url = minio_client.upload_image_bytes(
                        image_bytes=image_bytes,
                        object_key=image_filename,
                        content_type=content_type,
                        bucket_name=bucket_name
                    )

                    # Record image info
                    extracted_images.append({
                        "filename": image_filename,
                        "page": page_idx + 1,
                        "content_type": base_image["ext"],
                        "url": image_url
                    })
                    print(f"Successfully extracted and uploaded image: {image_filename}")
                except Exception as e:
                    print(f"Error extracting image: {str(e)}")
        
        doc.close()
        doc = None
        
        # Build enhanced text/markdown
        enhanced_text = []
        
        # Check if it's a maintenance case document
        is_maintenance_doc = any(
            "Equipment Name" in page_data["text"] or
            "Case Study" in page_data["text"] or
            (("Model" in page_data["text"]) and
             ("Fault Name" in page_data["text"] or "Symptom" in page_data["text"]))
            for page_data in extracted_text
        )
        
        if is_maintenance_doc:
            print("Detected maintenance case format, using multi-document strategy")
            page_documents = []
            
            for i, page_data in enumerate(extracted_text):
                page_num = page_data["page"]
                page_text = page_data["text"].strip()

                # Clean numeric prefixes if any
                import re
                page_text = re.sub(r'^\d+\s*\n', '', page_text, flags=re.MULTILINE)

                page_content = []
                page_content.append(format_maintenance_page_text(page_text))

                # Use standard Markdown image syntax so RAGFlow's Markdown
                # preview renders the image instead of escaping raw HTML.
                page_images = [img for img in extracted_images if img["page"] == page_num]
                if page_images:
                    page_content.append("\n### Related Images\n")
                    for img in page_images:
                        page_content.append(f"![Repair Image]({img['url']})")
                
                page_markdown = "\n".join(page_content)
                metadata = extract_case_metadata(
                    page_markdown,
                    page_number=page_num,
                    source_pdf=pdf_filename or os.path.basename(pdf_path),
                )
                multilingual_hints, extra_metadata = build_multilingual_retrieval_hints(metadata)
                if multilingual_hints:
                    page_markdown = "\n".join(
                        [
                            page_markdown,
                            "",
                            "**Multilingual Retrieval Hints**",
                            multilingual_hints,
                        ]
                    )
                    metadata.update(extra_metadata)

                page_documents.append({
                    "page": page_num,
                    "content": page_markdown,
                    "title": f"Repair Case Page {page_num}",
                    "metadata": metadata,
                })
            
            return page_documents, extracted_images
        else:
            print("Using general document format strategy")
            for page_data in extracted_text:
                page_num = page_data["page"]
                page_text = page_data["text"].strip()

                import re
                page_text = re.sub(r'^\d+\s*\n', '', page_text, flags=re.MULTILINE)

                # Add text
                paragraphs = page_text.split('\n\n')
                for para in paragraphs:
                    if para.strip():
                        enhanced_text.append(para.strip())

                # Find images for this page
                page_images = [img for img in extracted_images if img["page"] == page_num]

                if page_images:
                    for img in page_images:
                        enhanced_text.append(f"\n![Document Image]({img['url']})\n")
        
        return "\n".join(enhanced_text), extracted_images
    
    except Exception as e:
        print(f"Error processing PDF file: {str(e)}")
        
        if extracted_images:
            print("Attempting to build text using extracted images only...")
            enhanced_text = []
            
            for page_num in range(1, page_count + 1):
                page_images = [img for img in extracted_images if img["page"] == page_num]
                
                if page_images:
                    enhanced_text.append(f"## Page {page_num} Images\n")
                    for img in page_images:
                        enhanced_text.append(f"\n![Image]({img['url']})\n")
            
            if enhanced_text:
                return "\n".join(enhanced_text), extracted_images
        
        raise Exception(f"Unable to process file: {pdf_path}, reason: {str(e)}")
    finally:
        if doc:
            try:
                doc.close()
            except:
                pass

def copy_images_to_server(extracted_images, target_dir="./images"):
    """
    Log image upload results (MinIO integrated version)
    Since images are uploaded directly to MinIO, this function is now for logging/verification.

    Parameters:
    - extracted_images: List of extracted image information
    - target_dir: (Unused) Target directory for local storage
    """
    print(f"Images uploaded to MinIO ({len(extracted_images)} images):")

    for img_info in extracted_images:
        print(f"- {img_info['filename']}: {img_info['url']}")

    print("All images successfully uploaded to MinIO and set to public read access.")
