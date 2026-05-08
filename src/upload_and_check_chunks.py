"""
上传序列化 Markdown 到 RAGFlow，并展示 chunking 结果。
用法：python3 src/upload_and_check_chunks.py
"""

import os
import sys
import time
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from dotenv import load_dotenv
load_dotenv()

from ragflow_sdk import RAGFlow

# ── 配置 ───────────────────────────────────────────────────────────────────────
API_KEY    = os.getenv("RAGFLOW_API_KEY")
BASE_URL   = os.getenv("RAGFLOW_BASE_URL", "http://localhost:8080")
EMBED_MODEL = os.getenv("RAGFLOW_EMBEDDING_MODEL", "gemini-embedding-001@Gemini")

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MD_FILE = os.path.join(PROJECT_ROOT, "data", "excavator_repair_case_en_serialized.md")

DATASET_NAME = "excavator_repair_kb_v3"

# chunk_method="one" 策略：
#   - 每个文档 = 1 个 available=True chunk（完整案例，~900 tokens）
#   - 完全跳过 Naive parser 的 element-level 提取（那会产生 24 个小碎片）
#   - 4 个案例文件 = 4 个向量，语义完整，检索精准
CHUNK_METHOD = "one"

PARSER_CONFIG = {
    "chunk_token_num": 2048,      # one 模式下不分割，设大值保险
    "delimiter": "\n!?。；！？",
    "html4excel": False,
    "layout_recognize": "Naive",
    "auto_keywords": 0,
    "auto_questions": 0,
    "overlapped_percent": 0,
    "raptor": {"use_raptor": False},
    "graphrag": {"use_graphrag": False},
}


def delete_existing_dataset(rag, name):
    try:
        existing = rag.list_datasets()
        ids = [ds.id for ds in existing if ds.name == name]
        if ids:
            rag.delete_datasets(ids)
            print(f"  已删除旧知识库: {name}")
    except Exception as e:
        print(f"  清理旧知识库失败（忽略）: {e}")


def wait_for_parsing(dataset, doc_ids, timeout=300):
    print("  等待解析完成...")
    start = time.time()
    while time.time() - start < timeout:
        all_done = True
        statuses = []
        for doc_id in doc_ids:
            docs = dataset.list_documents(id=doc_id)
            if docs:
                status = docs[0].run
                statuses.append(f"{doc_id[:8]}…={status}")
                if status not in ("DONE", "CANCEL"):
                    all_done = False
            else:
                all_done = False
        print(f"    状态: {', '.join(statuses)}")
        if all_done:
            return True
        time.sleep(8)
    print("  ⚠️  超时，解析可能仍在进行")
    return False


def list_chunks_via_api(dataset_id, doc_id):
    """通过 REST API 获取 chunk 列表"""
    url = f"{BASE_URL}/api/v1/datasets/{dataset_id}/documents/{doc_id}/chunks"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    params = {"page": 1, "page_size": 100}
    resp = requests.get(url, headers=headers, params=params)
    if resp.status_code == 200:
        data = resp.json()
        # 兼容不同版本的响应结构
        if "data" in data:
            chunks = data["data"].get("chunks", data["data"]) if isinstance(data["data"], dict) else data["data"]
        else:
            chunks = data.get("chunks", [])
        return chunks
    else:
        print(f"  获取 chunk 失败: {resp.status_code} {resp.text[:200]}")
        return []


def print_chunks(chunks):
    if not chunks:
        print("  ⚠️  未获取到 chunk，可能 API 路径不同")
        return

    # 分离父子 chunk
    parent_chunks = [c for c in chunks if not c.get("available_int", 1) == 0]

    print(f"\n{'='*70}")
    print(f"  共 {len(chunks)} 个 chunks")
    print(f"{'='*70}")

    for i, chunk in enumerate(chunks):
        content = chunk.get("content", chunk.get("content_with_weight", ""))
        chunk_id = chunk.get("id", chunk.get("chunk_id", "?"))[:12]
        available = chunk.get("available_int", chunk.get("available", 1))

        # available=1 是父chunk，available=0 是子chunk（在某些版本）
        # 也可能通过 content 长度区分
        chunk_type = "PARENT" if len(content) > 200 else "child "

        print(f"\n[{i+1:02d}] {chunk_type} | id={chunk_id}… | chars={len(content)}")
        print(f"  {'─'*60}")
        # 显示前 300 字符
        preview = content[:300].replace('\n', ' ↵ ')
        print(f"  {preview}")
        if len(content) > 300:
            print(f"  … (共 {len(content)} 字符)")


def split_md_into_cases(md_path):
    """
    按 \n---\n 把整个 MD 文件分割成独立的案例文件。
    每个案例一个 .md 文件，文件名带 case_N 后缀。
    """
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 按案例分割线切开（仅独行 --- 算分割线，表格中的 |---| 不受影响）
    import re
    sections = re.split(r'\n---\n', content)
    sections = [s.strip() for s in sections if s.strip()]

    print(f"  MD 文件切分为 {len(sections)} 个案例")
    cases = []
    for i, section in enumerate(sections, 1):
        # 提取标题作为文件名
        title_match = re.search(r'## .+?\(Page \d+\)', section)
        if title_match:
            title = title_match.group(0)
            safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')[:60]
        else:
            safe_title = f"case_{i}"
        filename = f"{safe_title}.md"
        cases.append({"filename": filename, "content": section, "case_num": i})

    return cases


def main():
    if not API_KEY:
        print("❌ 未找到 RAGFLOW_API_KEY，请检查 .env 文件")
        return

    if not os.path.exists(MD_FILE):
        print(f"❌ 找不到 MD 文件: {MD_FILE}")
        return

    print(f"✅ 连接 RAGFlow: {BASE_URL}")
    print(f"✅ MD 文件: {MD_FILE}")
    print(f"✅ Embedding: {EMBED_MODEL}")

    rag = RAGFlow(api_key=API_KEY, base_url=BASE_URL)

    # 0. 切分 MD 文件为独立案例
    print(f"\n[0/5] 切分 MD 文件为独立案例...")
    cases = split_md_into_cases(MD_FILE)
    for c in cases:
        print(f"  案例 {c['case_num']}: {c['filename']} ({len(c['content'])} chars)")

    # 1. 删除旧知识库
    print(f"\n[1/5] 清理旧知识库...")
    delete_existing_dataset(rag, DATASET_NAME)

    # 2. 创建知识库
    print(f"\n[2/5] 创建知识库: {DATASET_NAME}")
    dataset = rag.create_dataset(
        name=DATASET_NAME,
        description="挖掘机维修案例 - 表格序列化版本，整案例单向量",
        embedding_model=EMBED_MODEL,
        chunk_method=CHUNK_METHOD
    )
    print(f"  知识库 ID: {dataset.id}")

    # 3. 配置父子分块（通过 legacy endpoint）
    print(f"\n[3/5] 配置 chunk_method={CHUNK_METHOD} 参数...")
    resp = requests.post(
        f"{BASE_URL}/v1/kb/update",
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={
            "kb_id": dataset.id,
            "name": DATASET_NAME,
            "description": "挖掘机维修案例 - 表格序列化版本，整案例单向量",  # 必填字段
            "permission": "me",
            "language": "English",
            "parser_id": CHUNK_METHOD,
            "embd_id": EMBED_MODEL,
            "parser_config": PARSER_CONFIG,
        }
    )
    result = resp.json()
    if result.get("code") == 0:
        print("  ✅ 父子分块配置成功")
    else:
        print(f"  ⚠️  配置警告: {result.get('message', result)}")

    # 4. 上传 4 个独立案例文件
    print(f"\n[4/5] 上传 {len(cases)} 个案例文件...")
    docs_to_upload = [
        {"display_name": c["filename"], "blob": c["content"].encode("utf-8")}
        for c in cases
    ]
    dataset.upload_documents(docs_to_upload)
    print(f"  ✅ 已上传 {len(cases)} 个案例文件")

    # 等一下让文件出现在列表中
    time.sleep(3)

    # 5. 解析并等待完成
    print(f"\n[5/5] 开始解析文档...")
    time.sleep(3)
    docs = dataset.list_documents()
    if not docs:
        print("  ❌ 文档列表为空，上传可能失败")
        return

    doc_ids = [doc.id for doc in docs]
    print(f"  找到 {len(doc_ids)} 个文档: {[d[:8]+'…' for d in doc_ids]}")

    dataset.async_parse_documents(doc_ids)
    done = wait_for_parsing(dataset, doc_ids)

    if not done:
        print("  继续尝试获取 chunks（可能部分完成）...")

    # 6. 展示每个文档的 Chunk 结果
    print(f"\n{'='*70}")
    print("  📊 CHUNK 结果（每个文件 = 一个案例）")
    print(f"{'='*70}")

    total_chunks = 0

    for doc_id in doc_ids:
        docs_info = dataset.list_documents(id=doc_id)
        doc_name = docs_info[0].name if docs_info else doc_id
        chunks = list_chunks_via_api(dataset.id, doc_id)

        print(f"\n📄 {doc_name}")
        print(f"   chunks 数: {len(chunks)}")

        for i, c in enumerate(chunks, 1):
            content = c.get("content", "")
            approx_tokens = len(content) // 4
            print(f"   [{i}] {len(content)} chars (~{approx_tokens} tokens)")
            print(f"       {content[:200].replace(chr(10), ' ')[:200]}")
            if len(content) > 200:
                print(f"       ...(共 {len(content)} 字符)")

        total_chunks += len(chunks)

    print(f"\n{'='*70}")
    print(f"  ✅ 完成！")
    print(f"  知识库 ID: {dataset.id}")
    print(f"  文档数: {len(doc_ids)} 个案例文件")
    print(f"  总 chunks: {total_chunks} 个（理想值: {len(doc_ids)} 个，每案例一个向量）")
    print(f"  RAGFlow UI: {BASE_URL}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
