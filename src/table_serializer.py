#!/usr/bin/env python3
"""
Table Serializer for Maintenance PDF Documents
-------------------------------------------------
借鉴 IBM RAG Challenge 冠军方案的表格序列化思路:
  1. Docling 解析 PDF → 结构化文档（表格 + 文字块，保留页码和顺序）
     PyMuPDF 提取图片 → 保存为本地 PNG → 嵌入 Markdown
  2. 对每张表格：抓取 section header + 前后文字作为上下文，调用 LLM
     生成"自包含信息块"（每行/每个概念一个，完全独立）
  3. 输出 Markdown：信息块（可搜索）+ 原始表格 + 图片（保留原貌）
  4. 上传 .md 到 RAGFlow，使用 naive 解析器 + 大 chunk size

Usage:
  python table_serializer.py --pdf ../data/pdf/excavator_repair_case_en.pdf
  python table_serializer.py --pdf ../data/pdf/your.pdf --output ../data/out.md
"""

import os
import re
import json
import argparse
import fitz                      # PyMuPDF，用于提取图片
from pathlib import Path
from dotenv import load_dotenv

# ── Docling ───────────────────────────────────────────────────────────────────
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode

# ── Google Gemini ─────────────────────────────────────────────────────────────
from google import genai
from google.genai import types


# ─────────────────────────────────────────────────────────────────────────────
# Step 1a: 用 PyMuPDF 提取每页图片，保存为本地 PNG
# ─────────────────────────────────────────────────────────────────────────────

def extract_images_per_page(pdf_path: str, image_dir: Path) -> dict[int, list[str]]:
    """
    用 PyMuPDF 提取 PDF 每页的图片，保存为 PNG，返回 {page_num: [image_path, ...]}
    """
    image_dir.mkdir(parents=True, exist_ok=True)
    result: dict[int, list[str]] = {}

    doc = fitz.open(pdf_path)
    pdf_stem = Path(pdf_path).stem

    for page_idx in range(len(doc)):
        page_num = page_idx + 1
        page = doc[page_idx]
        image_list = page.get_images(full=True)
        result[page_num] = []

        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]
            ext = base_image["ext"]          # png / jpeg / ...

            # 过滤掉太小的图（通常是图标、线条）
            if len(img_bytes) < 5000:        # < 5KB 跳过
                continue

            filename = f"{pdf_stem}_page{page_num}_img{img_idx + 1}.{ext}"
            filepath = image_dir / filename
            with open(filepath, "wb") as f:
                f.write(img_bytes)

            result[page_num].append(str(filepath))

    doc.close()
    print(f"    → 图片已保存到 {image_dir}，"
          f"共 {sum(len(v) for v in result.values())} 张")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Step 1b: 用 Docling 解析文字和表格结构
# ─────────────────────────────────────────────────────────────────────────────

def parse_pdf(pdf_path: str) -> dict:
    """
    用 Docling 解析 PDF，按页返回结构化数据。
    每页结构：
    {
      "section_header": "1.21.4. Case Study: ...",   # 页面标题（如果有）
      "texts":  [{"text": ..., "label": ..., "order": int}, ...],
      "tables": [{"markdown": ..., "order": int}, ...],
      "pictures": [{"order": int}, ...],              # 记录图片位置（顺序）
    }
    """
    print(f"[1/3] Docling 解析 PDF: {pdf_path}")

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    doc = result.document

    pages: dict[int, dict] = {}

    for item, level in doc.iterate_items():
        label = str(getattr(item, "label", ""))
        if not label:
            continue

        page_num = None
        if hasattr(item, "prov") and item.prov:
            page_num = item.prov[0].page_no
        if page_num is None:
            continue

        if page_num not in pages:
            pages[page_num] = {
                "section_header": "",
                "texts": [],
                "tables": [],
                "pictures": [],
            }

        page = pages[page_num]
        order = len(page["texts"]) + len(page["tables"]) + len(page["pictures"])

        if label == "section_header":
            text = getattr(item, "text", "").strip()
            if text:
                # 优先使用包含 "Case Study" 的标题，否则用最后出现的
                if "case study" in text.lower() or not page["section_header"]:
                    page["section_header"] = text
                page["texts"].append({"text": text, "label": label, "order": order})

        elif label in ("text", "paragraph", "title", "list_item", "caption"):
            text = getattr(item, "text", "").strip()
            if text:
                page["texts"].append({"text": text, "label": label, "order": order})

        elif label == "table":
            md = item.export_to_markdown()
            if md.strip():
                page["tables"].append({"markdown": md, "order": order})

        elif label == "picture":
            # 记录图片在页面中的阅读顺序位置，图片数据由 PyMuPDF 提取
            page["pictures"].append({"order": order})

    n_tables = sum(len(p["tables"]) for p in pages.values())
    n_pics   = sum(len(p["pictures"]) for p in pages.values())
    print(f"    → {len(pages)} 页，{n_tables} 张表格，{n_pics} 张图片")
    return pages


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: 调用 Gemini 对每张表格生成自包含信息块
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a table serialization agent for technical maintenance manuals.

Your task: given a table and its surrounding context, produce a list of
SELF-CONTAINED information blocks — one block per row or logical concept.

Each block MUST:
- Be completely independent (no reference to "the table above" or "see header")
- Include ALL relevant context: case name, fault description, machine model,
  component name, steps, results — everything needed to understand it alone
- Be a complete sentence or short paragraph a technician can read and act on
- Be searchable when a technician types a natural language question

Output JSON only, no explanation:
{
  "blocks": [
    "Full self-contained statement about row/concept 1...",
    "Full self-contained statement about row/concept 2...",
    ...
  ]
}"""


def serialize_table(
    client: genai.Client,
    table_markdown: str,
    section_header: str,
    context_before: str,
    context_after: str,
    model: str = "gemini-2.5-pro"
) -> list[str]:
    """
    用 Gemini 把一张表格序列化为自包含信息块列表。
    section_header 是页面级标题（如 "1.21.4. Case Study: Engine Emitting Blue Smoke"），
    始终作为最高优先级的上下文传入。
    """
    user_parts = []

    # 页面标题是最重要的上下文，单独强调
    if section_header:
        user_parts.append(
            f'Case study / section this table belongs to:\n"""\n{section_header}\n"""'
        )

    # 表格前的文字上下文（如症状描述）
    if context_before.strip():
        user_parts.append(
            f'Text immediately before the table (symptom, description, etc.):\n"""\n{context_before.strip()}\n"""'
        )

    user_parts.append(
        f'Table content (Markdown):\n"""\n{table_markdown.strip()}\n"""'
    )

    if context_after.strip():
        user_parts.append(
            f'Text immediately after the table:\n"""\n{context_after.strip()}\n"""'
        )

    user_parts.append(
        "Produce self-contained information blocks. "
        "Always include the case name and fault type from the section header in each block."
    )

    try:
        response = client.models.generate_content(
            model=model,
            contents="\n\n".join(user_parts),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=2500,
                temperature=0,
                response_mime_type="application/json",
            ),
        )
        # Gemini 偶发返回 None，加保护
        if not response.text:
            print(f"    [WARN] Gemini 返回空响应，重试一次...")
            # 重试一次
            response = client.models.generate_content(
                model=model,
                contents="\n\n".join(user_parts),
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=2500,
                    temperature=0,
                    response_mime_type="application/json",
                ),
            )
        if not response.text:
            return []

        raw = response.text.strip()
        # 先尝试完整 JSON 解析
        try:
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                blocks = data.get("blocks", [])
                if isinstance(blocks, list) and blocks:
                    return [str(b).strip() for b in blocks if str(b).strip()]
        except json.JSONDecodeError:
            pass
        # 备选：提取所有长字符串
        string_blocks = re.findall(r'"([^"]{20,})"', raw)
        if string_blocks:
            return [b.strip() for b in string_blocks if b.strip()]
        return [raw[:500]] if raw else []
    except Exception as e:
        print(f"    [WARN] LLM 序列化失败: {e}")
        return []


def get_context(page_data: dict, table_order: int,
                max_chars: int = 400) -> tuple[str, str]:
    """
    返回该表格在同一页中紧邻的前后文字。
    - 图片不再阻断上文搜索（穿透图片继续往前找文字）
    - 遇到另一张表格才停止
    - 跳过 section_header（已单独传入 LLM）
    """
    all_elements = sorted(
        [{"type": "text",    **t} for t in page_data["texts"]
         if t["label"] != "section_header"] +
        [{"type": "table",   **t} for t in page_data["tables"]] +
        [{"type": "picture", **t} for t in page_data["pictures"]],
        key=lambda x: x["order"]
    )

    pos = next(
        (i for i, e in enumerate(all_elements)
         if e["type"] == "table" and e["order"] == table_order),
        None
    )
    if pos is None:
        return "", ""

    # 前文：穿透图片，遇到上一张表格才停
    before_texts = []
    for e in reversed(all_elements[:pos]):
        if e["type"] == "table":
            break
        if e["type"] == "text":
            before_texts.insert(0, e.get("text", ""))
        # picture: 跳过，继续往前找
    context_before = " ".join(before_texts)[-max_chars:]

    # 后文：穿透图片，遇到下一张表格才停，最多 3 段文字
    after_texts = []
    for e in all_elements[pos + 1:]:
        if e["type"] == "table":
            break
        if e["type"] == "text":
            after_texts.append(e.get("text", ""))
            if len(after_texts) >= 3:
                break
        # picture: 跳过，继续往后找
    context_after = " ".join(after_texts)[:max_chars]

    return context_before, context_after


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: 组装最终 Markdown
# ─────────────────────────────────────────────────────────────────────────────

def build_markdown(pages: dict, client: genai.Client, model: str,
                   page_images: dict[int, list[str]]) -> str:
    """
    每页输出一个 section，把同页所有表格整合在一起。
    - 第一张表格（机器信息）作为上下文传入 LLM
    - 第二张表格（诊断内容）是主体，LLM 生成信息块
    - 图片、症状文字、原始表格全部保留在同一 section
    """
    print(f"[2/3] LLM 表格序列化（模型: {model}）...")

    total_pages = len(pages)
    output_sections = []

    for page_num in sorted(pages.keys()):
        page        = pages[page_num]
        page_tables = sorted(page["tables"], key=lambda t: t["order"])
        section_hdr = page["section_header"]
        images      = page_images.get(page_num, [])
        # 症状文字：优先用页面中独立的症状描述
        symptom = next(
            (t["text"] for t in page["texts"]
             if t["label"] != "section_header"
             and len(t["text"].strip()) > 15
             and not t["text"].strip().isupper()),
            ""
        )
        # 如果没有独立症状文字，从 section_header 提取（去掉编号前缀）
        # 例："1.21.4. Case Study: Engine Emitting Blue Smoke" → "Engine Emitting Blue Smoke"
        if not symptom and section_hdr:
            match = re.search(r'Case Study:\s*(.+)', section_hdr, re.IGNORECASE)
            if match:
                symptom = match.group(1).strip()

        print(f"    Page {page_num}/{total_pages}（{len(page_tables)} 张表格）...", end=" ")

        # 无表格的页
        if not page_tables:
            text_body = "\n\n".join(
                t["text"] for t in page["texts"] if t["label"] != "section_header"
            )
            output_sections.append(
                f"---\n\n## {section_hdr or f'Page {page_num}'}  (Page {page_num})\n\n{text_body}\n"
            )
            print("（无表格，跳过 LLM）")
            continue

        # 区分机器信息表（第1张，通常只有1行数据）和诊断表（第2张）
        meta_table  = page_tables[0]                          # Model/Machine No./Hours...
        diag_tables = page_tables[1:] if len(page_tables) > 1 else []

        # 把机器信息表格转成一段上下文文字，传给 LLM
        meta_context = f"Machine info: {meta_table['markdown'].strip()}"

        all_blocks = []
        for diag_table in diag_tables:
            ctx_before, ctx_after = get_context(page, diag_table["order"])
            # 上文：优先用症状描述；补充机器信息
            combined_before = "\n".join(filter(None, [
                meta_context,
                symptom,
                ctx_before,
            ]))
            blocks = serialize_table(
                client=client,
                table_markdown=diag_table["markdown"],
                section_header=section_hdr,
                context_before=combined_before,
                context_after=ctx_after,
                model=model,
            )
            all_blocks.extend(blocks)

        print(f"{len(all_blocks)} 个信息块")

        # 组装单页 section（第一页不加前导 ---，避免被当成 YAML front matter）
        lines = (["---", ""] if output_sections else []) + [f"## {section_hdr}  (Page {page_num})", ""]

        # 症状描述
        if symptom:
            lines += [f"**Symptom:** {symptom}", ""]

        # 机器信息原始表格
        lines += ["**Machine info:**", "", meta_table["markdown"], ""]

        # 图片（放在机器信息和诊断内容之间）
        for i, img_path in enumerate(images):
            # 使用相对于输出 md 文件的路径
            rel_path = Path(img_path).name  # 只取文件名
            lines += [f"![Repair image {i+1}](images/{rel_path})", ""]

        # LLM 生成的信息块
        if all_blocks:
            lines += ["**Key information:**", ""]
            for i, block in enumerate(all_blocks, 1):
                lines.append(f"{i}. {block}")
            lines.append("")

        # 诊断原始表格
        for diag_table in diag_tables:
            lines += ["**Original table:**", "", diag_table["markdown"], ""]

        output_sections.append("\n".join(lines))

    return "\n".join(output_sections)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

    parser = argparse.ArgumentParser(
        description="将 PDF 表格序列化为可搜索的 Markdown，上传到 RAGFlow"
    )
    parser.add_argument("--pdf",    required=True, help="PDF 文件路径")
    parser.add_argument("--output", help="输出 Markdown 路径（默认同名 .md）")
    parser.add_argument("--model",  default="gemini-2.5-pro",
                        help="Gemini 模型（默认 gemini-2.5-pro）")
    parser.add_argument("--image-dir", default=None,
                        help="图片保存目录（默认 data/images/）")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"错误：PDF 不存在: {pdf_path}")
        return

    output_path = Path(args.output) if args.output else \
        pdf_path.parent.parent / (pdf_path.stem + "_serialized.md")

    image_dir = Path(args.image_dir) if args.image_dir else \
        pdf_path.parent.parent / "images"

    api_key = os.getenv("GOOGLE_AI_STUDIO_KEY")
    if not api_key:
        print("错误：.env 中未设置 GOOGLE_AI_STUDIO_KEY")
        return

    client = genai.Client(api_key=api_key)

    # 执行流水线
    print("=" * 60)
    page_images = extract_images_per_page(str(pdf_path), image_dir)
    pages       = parse_pdf(str(pdf_path))
    markdown    = build_markdown(pages, client, args.model, page_images)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")

    total_tables = sum(len(p["tables"]) for p in pages.values())
    total_images = sum(len(v) for v in page_images.values())
    print(f"\n[3/3] 完成！")
    print(f"    输入 PDF:    {pdf_path}")
    print(f"    输出 MD:     {output_path}  ({output_path.stat().st_size/1024:.1f} KB)")
    print(f"    表格数量:    {total_tables}")
    print(f"    图片数量:    {total_images}（保存在 {image_dir}）")
    print(f"\n下一步：")
    print(f"  1. 把 '{output_path.name}' 上传到 RAGFlow")
    print(f"  2. 设置 parser_id=naive，chunk_token_num=2048")


if __name__ == "__main__":
    main()
