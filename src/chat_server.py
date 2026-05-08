#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Excavator Maintenance Chat Server
FastAPI server with app-managed conversation memory on top of RAGFlow.
"""

import os
import time
import json
import re
import asyncio
import logging
import httpx
from pathlib import Path
from uuid import uuid4
from datetime import datetime, timezone
from contextlib import nullcontext
from logging.handlers import RotatingFileHandler
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types
from mem0 import Memory
try:
    from langfuse import Langfuse, propagate_attributes
except ImportError:
    Langfuse = None
    propagate_attributes = None

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

app = FastAPI(title="Excavator Maintenance Assistant", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

RAGFLOW_API_KEY = os.getenv("RAGFLOW_API_KEY")
RAGFLOW_BASE_URL = os.getenv("RAGFLOW_BASE_URL", "http://localhost:8080")
DEFAULT_SIMILARITY_THRESHOLD = float(os.getenv("RAGFLOW_SIMILARITY_THRESHOLD", "0.15"))
DEFAULT_VECTOR_WEIGHT = float(os.getenv("RAGFLOW_VECTOR_WEIGHT", "0.25"))
DEFAULT_KEYWORD_WEIGHT = max(0.0, min(1.0, 1 - DEFAULT_VECTOR_WEIGHT))
DEFAULT_RERANK_MODEL = os.getenv("RAGFLOW_RERANK_MODEL", "").strip()
DEFAULT_RETRIEVAL_TOP_K = int(os.getenv("RAGFLOW_RETRIEVAL_TOP_K", "32"))
DEFAULT_RETRIEVAL_PAGE_SIZE = int(os.getenv("RAGFLOW_RETRIEVAL_PAGE_SIZE", "4"))
HISTORY_ROUTER_MODEL = os.getenv("HISTORY_ROUTER_MODEL", "gemini-2.5-flash-lite")
HISTORY_MEMORY_WINDOW = int(os.getenv("HISTORY_MEMORY_WINDOW", "6"))
GOOGLE_AI_STUDIO_KEY = os.getenv("GOOGLE_AI_STUDIO_KEY", "").strip()
MEM0_ENABLED = os.getenv("MEM0_ENABLED", "true").strip().lower() not in {"0", "false", "no"}
MEM0_COLLECTION_NAME = os.getenv("MEM0_COLLECTION_NAME", "excavator_chat_memory")
MEM0_AGENT_ID = os.getenv("MEM0_AGENT_ID", "excavator-maintenance-assistant")
MEM0_SEARCH_LIMIT = int(os.getenv("MEM0_SEARCH_LIMIT", "3"))
MEM0_MAX_INJECTED = int(os.getenv("MEM0_MAX_INJECTED", "2"))
MEM0_DISTANCE_THRESHOLD = float(os.getenv("MEM0_DISTANCE_THRESHOLD", "0.28"))
MEM0_CHROMA_PATH = os.getenv(
    "MEM0_CHROMA_PATH",
    str((Path(__file__).resolve().parent.parent / "data" / "mem0_chroma").resolve()),
)
MEM0_HISTORY_DB_PATH = os.getenv(
    "MEM0_HISTORY_DB_PATH",
    str((Path(__file__).resolve().parent.parent / "data" / "mem0_history.db").resolve()),
)
OBSERVABILITY_LOG_PATH = os.getenv(
    "OBSERVABILITY_LOG_PATH",
    str((Path(__file__).resolve().parent.parent / "data" / "logs" / "chat_observability.jsonl").resolve()),
)
OBSERVABILITY_LOG_LEVEL = os.getenv("OBSERVABILITY_LOG_LEVEL", "INFO").strip().upper()
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com").strip()
LANGFUSE_TRACING_ENVIRONMENT = os.getenv("LANGFUSE_TRACING_ENVIRONMENT", "development").strip()
MISSING_DETAIL_PATTERNS = (
    "not available in the provided knowledge base",
    "not available in the knowledge base",
    "only contains a relevant maintenance hint",
    "知识库中未提供",
    "没有提供",
    "缺乏详细",
)
FOLLOW_UP_MARKERS = (
    "怎么修", "怎么处理", "怎么办", "为什么", "原因", "步骤", "方案", "图片",
    "哪个", "哪一个", "第几个", "继续", "刚才", "上一个", "上一条", "那", "这个", "它",
    "how to fix", "why", "which", "that", "this", "more detail", "next step",
)
REFERENCE_MARKERS = (
    "之前", "刚才", "上一个", "上一条", "那个案例", "第一个问题", "前面那个",
    "earlier", "previous", "that case", "the first issue", "the earlier one",
)

# ── Find the chat assistant ID at startup ──────────────────────────────────────
CHAT_ID: str | None = None
DATASET_ID: str | None = None
GEMINI_CLIENT: genai.Client | None = None
MEM0_CLIENT: Memory | None = None
LANGFUSE_CLIENT: Langfuse | None = None
LANGFUSE_AUTH_OK: bool | None = None
OBSERVABILITY_LOGGER: logging.Logger | None = None
LOCAL_SESSIONS: dict[str, dict] = {}


def _json_default(value):
    if isinstance(value, (set, tuple)):
        return list(value)
    if isinstance(value, (datetime, Path)):
        return str(value)
    return str(value)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_observability_logger() -> logging.Logger:
    global OBSERVABILITY_LOGGER
    if OBSERVABILITY_LOGGER is not None:
        return OBSERVABILITY_LOGGER

    logger = logging.getLogger("excavator_chat_observability")
    logger.setLevel(getattr(logging, OBSERVABILITY_LOG_LEVEL, logging.INFO))
    logger.propagate = False
    if not logger.handlers:
        log_path = Path(OBSERVABILITY_LOG_PATH)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=5 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(file_handler)
    OBSERVABILITY_LOGGER = logger
    return logger


def _emit_observability_log(event: str, **payload) -> None:
    logger = _get_observability_logger()
    record = {
        "ts": _utc_now_iso(),
        "event": event,
        **payload,
    }
    logger.info(json.dumps(record, ensure_ascii=False, default=_json_default))


def _tail_observability_logs(limit: int = 50, session_id: str | None = None, request_id: str | None = None, issue_stage: str | None = None) -> list[dict]:
    log_path = Path(OBSERVABILITY_LOG_PATH)
    if not log_path.exists():
        return []

    rows: list[dict] = []
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if session_id and payload.get("session_id") != session_id:
                continue
            if request_id and payload.get("request_id") != request_id:
                continue
            if issue_stage:
                diagnosis = payload.get("diagnosis") or {}
                if diagnosis.get("stage") != issue_stage:
                    continue
            rows.append(payload)
    return rows[-max(1, min(limit, 200)):]


def _get_langfuse_client() -> Langfuse | None:
    global LANGFUSE_CLIENT
    if Langfuse is None:
        return None
    if not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY:
        return None
    if LANGFUSE_CLIENT is None:
        LANGFUSE_CLIENT = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            base_url=LANGFUSE_BASE_URL,
            environment=LANGFUSE_TRACING_ENVIRONMENT,
        )
    return LANGFUSE_CLIENT


def _check_langfuse_auth() -> bool:
    global LANGFUSE_AUTH_OK
    client = _get_langfuse_client()
    if client is None:
        return False
    if LANGFUSE_AUTH_OK is None:
        try:
            LANGFUSE_AUTH_OK = bool(client.auth_check())
        except Exception:
            LANGFUSE_AUTH_OK = False
    return LANGFUSE_AUTH_OK


def _langfuse_attributes_context(session_id: str, request_id: str):
    if propagate_attributes is None or _get_langfuse_client() is None:
        return nullcontext()
    return propagate_attributes(
        user_id=session_id,
        session_id=session_id,
        metadata={"request_id": request_id, "app": "excavator_chat_server"},
        tags=["excavator-chat", "ragflow", "mem0"],
        trace_name="excavator_chat_turn",
    )


def _safe_langfuse_update(observation, **kwargs) -> None:
    if observation is None:
        return
    try:
        observation.update(**kwargs)
    except Exception:
        pass


def _preview_text(value: str, limit: int = 320) -> str:
    return _collapse_whitespace(value)[:limit]

async def get_chat_id() -> str:
    global CHAT_ID
    if CHAT_ID:
        return CHAT_ID
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{RAGFLOW_BASE_URL}/api/v1/chats",
            headers={"Authorization": f"Bearer {RAGFLOW_API_KEY}"},
            timeout=10,
        )
        data = resp.json()
        chats = data.get("data", [])
        for c in chats:
            if "Excavator" in c.get("name", ""):
                CHAT_ID = c["id"]
                return CHAT_ID
        raise HTTPException(status_code=503, detail="Excavator chat assistant not found in RAGFlow")


async def get_dataset_id() -> str:
    global DATASET_ID
    if DATASET_ID:
        return DATASET_ID
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{RAGFLOW_BASE_URL}/api/v1/datasets",
            headers={"Authorization": f"Bearer {RAGFLOW_API_KEY}"},
            timeout=10,
        )
        data = resp.json()
        datasets = data.get("data", [])
        for dataset in datasets:
            if "Excavator" in dataset.get("name", ""):
                DATASET_ID = dataset["id"]
                return DATASET_ID
        raise HTTPException(status_code=503, detail="Excavator dataset not found in RAGFlow")


async def get_chat_details() -> dict:
    chat_id = await get_chat_id()
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{RAGFLOW_BASE_URL}/api/v1/chats",
            headers={"Authorization": f"Bearer {RAGFLOW_API_KEY}"},
            timeout=10,
        )
    data = resp.json()
    chats = data.get("data", [])
    for chat in chats:
        if chat.get("id") == chat_id:
            return chat
    raise HTTPException(status_code=503, detail="Excavator chat assistant details not found in RAGFlow")


def _get_gemini_client() -> genai.Client | None:
    global GEMINI_CLIENT
    if not GOOGLE_AI_STUDIO_KEY:
        return None
    if GEMINI_CLIENT is None:
        GEMINI_CLIENT = genai.Client(api_key=GOOGLE_AI_STUDIO_KEY)
    return GEMINI_CLIENT


def _get_mem0_client() -> Memory | None:
    global MEM0_CLIENT
    if not MEM0_ENABLED or not GOOGLE_AI_STUDIO_KEY:
        return None
    if MEM0_CLIENT is not None:
        return MEM0_CLIENT

    os.environ.setdefault("GOOGLE_API_KEY", GOOGLE_AI_STUDIO_KEY)
    Path(MEM0_CHROMA_PATH).mkdir(parents=True, exist_ok=True)
    Path(MEM0_HISTORY_DB_PATH).parent.mkdir(parents=True, exist_ok=True)

    config = {
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": MEM0_COLLECTION_NAME,
                "path": MEM0_CHROMA_PATH,
            },
        },
        "llm": {
            "provider": "gemini",
            "config": {
                "model": "gemini-2.5-flash-lite",
                "api_key": GOOGLE_AI_STUDIO_KEY,
            },
        },
        "embedder": {
            "provider": "gemini",
            "config": {
                "model": "models/gemini-embedding-001",
                "api_key": GOOGLE_AI_STUDIO_KEY,
                "embedding_dims": 768,
            },
        },
        "history_db_path": MEM0_HISTORY_DB_PATH,
    }

    try:
        MEM0_CLIENT = Memory.from_config(config)
    except Exception:
        MEM0_CLIENT = None
    return MEM0_CLIENT


async def _get_local_session(session_id: str) -> dict:
    session = LOCAL_SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


def _new_session_record(name: str, greeting: str) -> dict:
    now = time.time()
    session_id = uuid4().hex
    return {
        "id": session_id,
        "name": name,
        "create_date": now,
        "update_date": now,
        "messages": [{"role": "assistant", "content": greeting, "created_at": now}],
    }


def _collapse_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _normalize_model(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (value or "").upper())


def _normalize_machine_no(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (value or "").upper())


def _extract_automatic_filters(question: str) -> dict[str, str]:
    filters: dict[str, str] = {}
    if not question:
        return filters

    case_match = re.search(r"\b\d+(?:\.\d+){2,}\b", question)
    if case_match:
        filters["case_id"] = case_match.group(0)

    machine_match = re.search(r"\bFTC[0-9A-Z]+\b", question.upper())
    if machine_match:
        filters["machine_no_normalized"] = _normalize_machine_no(machine_match.group(0))

    model_match = re.search(r"\bFR-?\d+[A-Z]?\b", question.upper())
    if model_match:
        filters["model_normalized"] = _normalize_model(model_match.group(0))

    return filters


def _normalize_explicit_filters(raw_filters: dict | None) -> dict[str, str]:
    if not raw_filters:
        return {}

    normalized: dict[str, str] = {}
    for key, value in raw_filters.items():
        if value is None:
            continue

        cleaned = _collapse_whitespace(str(value))
        if not cleaned:
            continue

        field = key.lower()
        if field in {"case_id", "case_title", "fault_name", "symptom", "location", "environment", "source_pdf", "page_number"}:
            normalized[field] = cleaned
        elif field in {"model", "model_normalized"}:
            normalized["model_normalized"] = _normalize_model(cleaned)
        elif field in {"machine_no", "machine_number", "machine_no_normalized"}:
            normalized["machine_no_normalized"] = _normalize_machine_no(cleaned)

    return normalized


def build_metadata_condition(question: str, metadata_filters: dict | None = None) -> tuple[dict | None, dict[str, str]]:
    applied_filters = _extract_automatic_filters(question)
    applied_filters.update(_normalize_explicit_filters(metadata_filters))
    if not applied_filters:
        return None, {}

    return {
        "logic": "and",
        "conditions": [
            {
                "name": name,
                "comparison_operator": "is",
                "value": value,
            }
            for name, value in applied_filters.items()
        ],
    }, applied_filters


async def list_case_documents() -> list[dict]:
    dataset_id = await get_dataset_id()
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{RAGFLOW_BASE_URL}/api/v1/datasets/{dataset_id}/documents",
            headers={"Authorization": f"Bearer {RAGFLOW_API_KEY}"},
            params={"page": 1, "page_size": 100},
            timeout=20,
        )
    data = resp.json()
    if data.get("code") != 0:
        raise HTTPException(status_code=500, detail=data.get("message"))
    return data.get("data", {}).get("docs", [])


def _document_matches_filters(document: dict, applied_filters: dict[str, str]) -> bool:
    meta_fields = document.get("meta_fields", {}) or {}
    for field, expected in applied_filters.items():
        actual = _collapse_whitespace(str(meta_fields.get(field, "")))
        if actual != expected:
            return False
    return True


async def get_matching_documents(applied_filters: dict[str, str]) -> list[dict]:
    documents = await list_case_documents()
    if not applied_filters:
        return documents
    return [document for document in documents if _document_matches_filters(document, applied_filters)]


async def retrieve_case_chunks(question: str, document_ids: list[str] | None = None, page_size: int = DEFAULT_RETRIEVAL_PAGE_SIZE) -> list[dict]:
    dataset_id = await get_dataset_id()
    document_name_map = {
        document.get("id"): document.get("name", "")
        for document in await list_case_documents()
    }

    payload = {
        "question": question,
        "dataset_ids": [dataset_id],
        "document_ids": document_ids or [],
        "page": 1,
        "page_size": max(1, min(page_size, 10)),
        "similarity_threshold": 0.0 if document_ids else DEFAULT_SIMILARITY_THRESHOLD,
        "vector_similarity_weight": DEFAULT_VECTOR_WEIGHT,
        "top_k": DEFAULT_RETRIEVAL_TOP_K,
        "keyword": True,
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{RAGFLOW_BASE_URL}/api/v1/retrieval",
            headers={"Authorization": f"Bearer {RAGFLOW_API_KEY}"},
            json=payload,
            timeout=30,
        )

    data = resp.json()
    if data.get("code") != 0:
        raise HTTPException(status_code=500, detail=data.get("message"))
    chunks = data.get("data", {}).get("chunks", [])
    for chunk in chunks:
        if not chunk.get("document_name") and chunk.get("document_id") in document_name_map:
            chunk["document_name"] = document_name_map[chunk["document_id"]]
    return chunks


def _build_prefiltered_context(chunks: list[dict]) -> str:
    context_blocks = []
    for index, chunk in enumerate(chunks, start=1):
        doc_name = chunk.get("document_name", f"case_{index}")
        content = chunk.get("content", "").strip()
        if content:
            context_blocks.append(f"[Filtered Case {index}: {doc_name}]\n{content}")
    return "\n\n".join(context_blocks)


def _source_preview_from_chunks(chunks: list[dict]) -> list[dict]:
    return [
        {
            "chunk": chunk.get("content", "")[:200],
            "doc": (
                chunk.get("document_name")
                or chunk.get("doc_name")
                or chunk.get("document_id", "")
            ),
        }
        for chunk in chunks[:3]
    ]


def _extract_doc_ids(chunks: list[dict]) -> list[str]:
    doc_ids: list[str] = []
    for chunk in chunks:
        document_id = chunk.get("document_id")
        if document_id and document_id not in doc_ids:
            doc_ids.append(document_id)
    return doc_ids


def _mem0_scope(session_id: str) -> dict[str, str]:
    return {
        "user_id": session_id,
        "agent_id": MEM0_AGENT_ID,
        "run_id": session_id,
    }


def _memory_distance(memory: dict) -> float:
    try:
        return float(memory.get("score"))
    except (TypeError, ValueError):
        return 999.0


def _compact_memory_candidates(memories: list[dict]) -> list[dict]:
    compact = []
    for memory in memories:
        metadata = memory.get("metadata") or {}
        compact.append(
            {
                "id": memory.get("id"),
                "score": _memory_distance(memory),
                "case_id": metadata.get("case_id"),
                "case_title": metadata.get("case_title"),
                "source_doc": metadata.get("source_doc"),
                "symptom": metadata.get("symptom"),
                "memory": memory.get("memory", ""),
            }
        )
    return compact


def _select_memory_candidates(memory_hits: list[dict], indexes: list[int]) -> list[dict]:
    selected: list[dict] = []
    for index in indexes:
        if 0 <= index < len(memory_hits):
            selected.append(memory_hits[index])
        if len(selected) >= MEM0_MAX_INJECTED:
            break
    return selected


def _format_memory_candidates_for_prompt(memory_hits: list[dict]) -> str:
    if not memory_hits:
        return "[none]"
    lines = []
    for index, memory in enumerate(memory_hits):
        metadata = memory.get("metadata") or {}
        lines.append(
            f"[{index}] distance={_memory_distance(memory):.3f} | "
            f"case_id={metadata.get('case_id', '')} | "
            f"source_doc={metadata.get('source_doc', '')} | "
            f"summary={memory.get('memory', '')}"
        )
    return "\n".join(lines)


def _format_selected_memories(selected_memories: list[dict]) -> str:
    if not selected_memories:
        return ""
    blocks = []
    for index, memory in enumerate(selected_memories, start=1):
        metadata = memory.get("metadata") or {}
        lines = [
            f"[Memory {index}]",
            f"Case ID: {metadata.get('case_id', '')}",
            f"Case Title: {metadata.get('case_title', '')}",
            f"Source Document: {metadata.get('source_doc', '')}",
            f"Symptom: {metadata.get('symptom', '')}",
            f"Stored Memory Summary: {memory.get('memory', '')}",
        ]
        blocks.append("\n".join(line for line in lines if line.rstrip(": ")))
    return "\n\n".join(blocks)


def _search_mem0_memories(session_id: str, question: str) -> list[dict]:
    memory_client = _get_mem0_client()
    if memory_client is None:
        return []
    try:
        result = memory_client.search(
            question,
            limit=MEM0_SEARCH_LIMIT,
            rerank=False,
            **_mem0_scope(session_id),
        )
    except Exception:
        return []
    return (result or {}).get("results", [])


def _build_contextual_retrieval_query(session: dict, question: str, context_mode: str, selected_memories: list[dict]) -> str:
    if context_mode == "recent_history":
        previous_user, previous_assistant = _last_turn_pair(session)
        parts = [part for part in [previous_user, previous_assistant[:180] if previous_assistant else "", question] if part]
        return "\n".join(parts)
    if context_mode == "episodic_memory" and selected_memories:
        memory = selected_memories[0]
        metadata = memory.get("metadata") or {}
        parts = [
            metadata.get("case_title", ""),
            metadata.get("symptom", ""),
            memory.get("memory", "")[:240],
            question,
        ]
        return "\n".join(part for part in parts if part)
    return question


def _has_reference_marker(question: str) -> bool:
    normalized = _collapse_whitespace(question).lower()
    return any(marker in normalized for marker in REFERENCE_MARKERS)


def _normalize_router_decision(
    session: dict,
    question: str,
    router: dict,
    current_turn_chunks: list[dict],
    memory_hits: list[dict],
    selected_memories: list[dict],
) -> tuple[dict, list[dict]]:
    _, _, previous_source_doc = _last_turn_context(session)
    current_doc = (current_turn_chunks[0].get("document_name") if current_turn_chunks else "") or ""
    top_memory = memory_hits[0] if memory_hits else None
    top_memory_doc = ((top_memory or {}).get("metadata") or {}).get("source_doc", "")

    if router.get("mode") == "recent_history":
        if top_memory and top_memory_doc and previous_source_doc and top_memory_doc != previous_source_doc:
            if _has_reference_marker(question) or (current_doc and current_doc == top_memory_doc and current_doc != previous_source_doc):
                return (
                    {
                        **router,
                        "mode": "episodic_memory",
                        "reason": "memory_reference_override",
                        "memory_indexes": [0],
                    },
                    [top_memory],
                )
        return router, selected_memories

    if router.get("mode") == "independent":
        if _has_follow_up_marker(question) and previous_source_doc and (not current_doc or current_doc == previous_source_doc):
            return {**router, "mode": "recent_history", "reason": "follow_up_marker_override"}, []
        if top_memory and top_memory_doc and current_doc and top_memory_doc == current_doc and top_memory_doc != previous_source_doc and _has_reference_marker(question):
            return (
                {
                    **router,
                    "mode": "episodic_memory",
                    "reason": "independent_promoted_to_memory_reference",
                    "memory_indexes": [0],
                },
                [top_memory],
            )
        return router, selected_memories

    if not selected_memories:
        return {**router, "mode": "independent", "reason": "episodic_without_selected_memory"}, []

    memory_doc = ((selected_memories[0].get("metadata") or {}).get("source_doc")) or ""
    if current_doc and memory_doc and current_doc != memory_doc and not _has_reference_marker(question):
        return {**router, "mode": "independent", "reason": "episodic_doc_mismatch"}, []
    if _memory_distance(selected_memories[0]) > MEM0_DISTANCE_THRESHOLD and not _has_reference_marker(question):
        return {**router, "mode": "independent", "reason": "episodic_distance_too_weak"}, []
    return router, selected_memories


def _recent_turn_messages(session: dict, max_messages: int = HISTORY_MEMORY_WINDOW) -> list[dict]:
    messages = session.get("messages", [])
    filtered = [m for m in messages if m.get("role") in {"user", "assistant"}]
    if len(filtered) <= 1:
        return []
    # Drop the initial greeting for routing/generation context.
    if filtered and filtered[0].get("role") == "assistant":
        filtered = filtered[1:]
    return filtered[-max_messages:]


def _last_turn_context(session: dict) -> tuple[str, str, str]:
    messages = _recent_turn_messages(session, max_messages=4)
    previous_user = ""
    previous_assistant = ""
    previous_source_doc = ""
    for message in reversed(messages):
        if message.get("role") == "assistant" and not previous_assistant:
            previous_assistant = message.get("content", "")
            previous_source_doc = message.get("source_doc", "")
        elif message.get("role") == "user" and not previous_user:
            previous_user = message.get("content", "")
        if previous_user and previous_assistant:
            break
    return previous_user, previous_assistant, previous_source_doc


def _last_turn_pair(session: dict) -> tuple[str, str]:
    previous_user, previous_assistant, _ = _last_turn_context(session)
    return previous_user, previous_assistant


def _has_follow_up_marker(question: str) -> bool:
    normalized = _collapse_whitespace(question).lower()
    return any(marker in normalized for marker in FOLLOW_UP_MARKERS)


def _history_router_fallback(question: str, memory_hits: list[dict]) -> tuple[str, list[int], str]:
    if _has_follow_up_marker(question):
        return "recent_history", [], "follow_up_marker"
    if memory_hits and _memory_distance(memory_hits[0]) <= MEM0_DISTANCE_THRESHOLD:
        return "episodic_memory", [0], "memory_distance_match"
    return "independent", [], "fallback_independent"


async def _classify_context_mode(
    session: dict,
    question: str,
    retrieval_hint: str,
    memory_hits: list[dict],
) -> dict:
    previous_user, previous_assistant = _last_turn_pair(session)
    if not previous_user and not previous_assistant and not memory_hits:
        return {"mode": "independent", "reason": "no_previous_turn_or_memory", "memory_indexes": []}

    client = _get_gemini_client()
    if client is None:
        mode, indexes, reason = _history_router_fallback(question, memory_hits)
        return {"mode": mode, "reason": reason, "memory_indexes": indexes}

    def _call_router():
        prompt = (
            "Decide how to assemble context for the current user question.\n"
            "Return strict JSON with keys mode, reason, memory_indexes.\n"
            'mode must be one of "recent_history", "episodic_memory", "independent".\n'
            "Choose recent_history only when the current question clearly depends on the immediately previous turn's case, answer, or omitted context.\n"
            "Choose episodic_memory when the question is not a direct follow-up to the previous turn but does match one or more retrieved older conversation memories.\n"
            "Choose independent when the question introduces a new fault/topic and should be answered as a one-shot question without conversation history.\n"
            "If the current question describes a different symptom/fault from the previous turn, it is not recent_history even if it is the same machine.\n"
            "If uncertain, prefer independent over injecting irrelevant history.\n"
            "For episodic_memory, memory_indexes must be a list of candidate indexes, ordered by usefulness.\n\n"
            "Examples:\n"
            "Previous user question: 我的车子冒蓝烟\n"
            "Previous assistant answer: 这是发动机冒蓝烟案例...\n"
            "Current user question: 为什么\n"
            'Output: {"mode":"recent_history","reason":"asks for explanation of same fault","memory_indexes":[]}\n\n'
            "Previous user question: 我的车子冒蓝烟\n"
            "Previous assistant answer: 这是发动机冒蓝烟案例...\n"
            "Current user question: 车子抖动\n"
            'Output: {"mode":"independent","reason":"different fault symptom","memory_indexes":[]}\n\n'
            "Previous user question: 车子高温报警\n"
            "Previous assistant answer: 这是高温报警案例...\n"
            "Retrieved memory candidate [0]: 蓝烟案例摘要...\n"
            "Current user question: 我的车子也冒蓝烟，是喷油器的问题吗\n"
            'Output: {"mode":"episodic_memory","reason":"matches an older blue-smoke topic rather than the previous high-temperature topic","memory_indexes":[0]}\n\n'
            f"Previous user question:\n{previous_user or '[none]'}\n\n"
            f"Previous assistant answer:\n{previous_assistant or '[none]'}\n\n"
            f"Current retrieved case hint:\n{retrieval_hint or '[none]'}\n\n"
            f"Retrieved episodic memory candidates:\n{_format_memory_candidates_for_prompt(memory_hits)}\n\n"
            f"Current user question:\n{question}"
        )
        return client.models.generate_content(
            model=HISTORY_ROUTER_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=120,
                response_mime_type="application/json",
            ),
        )

    try:
        response = await asyncio.to_thread(_call_router)
        raw = (response.text or "").strip()
        data = json.loads(raw) if raw else {}
        mode = data.get("mode")
        if mode not in {"recent_history", "episodic_memory", "independent"}:
            raise ValueError(f"Invalid router mode: {mode}")
        memory_indexes = data.get("memory_indexes") or []
        if not isinstance(memory_indexes, list):
            memory_indexes = []
        cleaned_indexes = [int(index) for index in memory_indexes if isinstance(index, int) or (isinstance(index, str) and index.isdigit())]
        return {"mode": mode, "reason": data.get("reason", "gemini_router"), "memory_indexes": cleaned_indexes}
    except Exception:
        mode, indexes, reason = _history_router_fallback(question, memory_hits)
        return {"mode": mode, "reason": reason, "memory_indexes": indexes}

def _assemble_messages(
    session: dict,
    question: str,
    context_mode: str,
    retrieval_hint: str = "",
    selected_memories: list[dict] | None = None,
) -> list[dict]:
    current_question = question
    if retrieval_hint and re.search(r"[\u4e00-\u9fff]", question):
        current_question = f"{question}\n\nRelevant maintenance hint: {retrieval_hint}"

    if context_mode == "independent":
        return [{"role": "user", "content": current_question}]

    if context_mode == "recent_history":
        history = _recent_turn_messages(session)
        messages = [{"role": m["role"], "content": m["content"]} for m in history]
        messages.append({"role": "user", "content": current_question})
        return messages

    memory_text = _format_selected_memories(selected_memories or [])
    episodic_prompt = (
        "Retrieved prior conversation memory from this same session.\n"
        "Use it only if it helps the current question. The retrieved knowledge-base case for the current turn still has priority.\n\n"
        f"{memory_text or '[none]'}\n\n"
        f"Current user question:\n{current_question}"
    )
    return [{"role": "user", "content": episodic_prompt}]


def _source_preview_from_reference_chunks(chunks: list[dict]) -> list[dict]:
    previews = []
    for chunk in chunks[:3]:
        previews.append(
            {
                "chunk": (chunk.get("content") or "")[:200],
                "doc": (
                    chunk.get("document_name")
                    or chunk.get("doc")
                    or chunk.get("document_id")
                    or ""
                ),
            }
        )
    return previews


def _diagnose_chat_turn(
    *,
    question: str,
    current_turn_chunks: list[dict],
    raw_answer: str,
    final_answer: str,
    reference_chunks: list[dict],
    completion_fallback_reason: str | None,
    context_mode: str,
) -> dict:
    top_doc = (
        current_turn_chunks[0].get("document_name")
        if current_turn_chunks and current_turn_chunks[0].get("document_name")
        else ""
    )
    source_docs = [
        source.get("doc", "")
        for source in (_source_preview_from_chunks(current_turn_chunks) or _source_preview_from_reference_chunks(reference_chunks))
        if source.get("doc")
    ]
    raw_preview = _preview_text(raw_answer)
    final_preview = _preview_text(final_answer)
    answer_changed = bool(raw_preview and final_preview and raw_preview != final_preview)

    if not current_turn_chunks:
        return {
            "status": "issue",
            "stage": "retrieval",
            "code": "no_hit",
            "summary": "Current question did not retrieve any case chunk from the knowledge base.",
            "retrieval_hits": 0,
            "top_doc": "",
            "source_docs": [],
            "answer_changed": False,
            "fallback_used": False,
        }

    if completion_fallback_reason:
        return {
            "status": "issue",
            "stage": "generation",
            "code": "completion_exception_fallback",
            "summary": "Model completion failed or timed out; answer fell back to the currently matched case.",
            "retrieval_hits": len(current_turn_chunks),
            "top_doc": top_doc,
            "source_docs": source_docs,
            "answer_changed": True,
            "fallback_used": True,
        }

    if _needs_case_answer_fallback(raw_answer, current_turn_chunks):
        return {
            "status": "issue",
            "stage": "generation",
            "code": "missing_detail_corrected",
            "summary": "Model claimed the case lacked detail; answer was corrected using the matched case chunk.",
            "retrieval_hits": len(current_turn_chunks),
            "top_doc": top_doc,
            "source_docs": source_docs,
            "answer_changed": True,
            "fallback_used": True,
        }

    if context_mode == "independent" and answer_changed:
        return {
            "status": "issue",
            "stage": "generation",
            "code": "ungrounded_answer_corrected",
            "summary": "Model draft was not fully grounded in the current case; answer was overridden by the matched case chunk.",
            "retrieval_hits": len(current_turn_chunks),
            "top_doc": top_doc,
            "source_docs": source_docs,
            "answer_changed": True,
            "fallback_used": True,
        }

    return {
        "status": "ok",
        "stage": "none",
        "code": "grounded",
        "summary": "Retrieval and generation stayed grounded on the current matched case.",
        "retrieval_hits": len(current_turn_chunks),
        "top_doc": top_doc,
        "source_docs": source_docs,
        "answer_changed": answer_changed,
        "fallback_used": False,
    }


def _build_retrieval_hint(chunk: dict) -> str:
    content = chunk.get("content", "")
    if not content:
        return ""

    lines = [line.strip() for line in content.splitlines() if line.strip()]
    title = ""
    symptom = ""
    zh_title = ""
    zh_symptom = ""

    for idx, line in enumerate(lines):
        if line.startswith("## ") and not title:
            title = line[3:].strip()
        elif line == "**Symptom:**" and idx + 1 < len(lines) and not symptom:
            symptom = lines[idx + 1]
        elif line.startswith("中文故障名称:") and not zh_title:
            zh_title = line.split(":", 1)[1].strip()
        elif line.startswith("中文症状别名:") and not zh_symptom:
            zh_symptom = line.split(":", 1)[1].strip()

    hint_parts = [part for part in [title, symptom, zh_title, zh_symptom] if part]
    return " | ".join(hint_parts[:4])[:320]


def _parse_case_chunk(content: str) -> dict:
    lines = [line.rstrip() for line in (content or "").splitlines()]
    result = {
        "title": "",
        "case_id": "",
        "case_title": "",
        "machine_info": {},
        "sections": {},
        "image_urls": [],
        "zh_fault_name": "",
        "zh_symptom_aliases": "",
    }

    if not lines:
        return result

    for line in lines:
        if line.startswith("## "):
            result["title"] = line[3:].strip()
            match = re.match(r"^(?P<case_id>\d+(?:\.\d+)+)\.?\s*(?P<title>.+)$", result["title"])
            if match:
                result["case_id"] = match.group("case_id")
                cleaned_title = re.sub(r"^\s*Case Study:\s*", "", match.group("title").strip(), flags=re.IGNORECASE)
                result["case_title"] = cleaned_title
            else:
                result["case_title"] = result["title"]
            break

    current_section = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("![") and "](" in stripped:
            image_match = re.search(r"\((https?://[^)]+)\)", stripped)
            if image_match:
                result["image_urls"].append(image_match.group(1))
            continue
        if stripped.startswith("<img "):
            image_match = re.search(r'src="([^"]+)"', stripped)
            if image_match:
                result["image_urls"].append(image_match.group(1))
            continue
        if stripped == "**Machine info:**":
            current_section = "Machine info"
            continue
        if stripped.startswith("**") and stripped.endswith("**"):
            current_section = stripped.strip("*").strip().rstrip(":")
            result["sections"].setdefault(current_section, [])
            continue
        if not stripped:
            continue
        if current_section == "Machine info" and ":" in stripped:
            key, value = stripped.split(":", 1)
            result["machine_info"][key.strip()] = value.strip()
        elif current_section:
            result["sections"].setdefault(current_section, []).append(stripped)

    for key in list(result["sections"].keys()):
        result["sections"][key] = " ".join(result["sections"][key]).strip()

    result["zh_fault_name"] = result["sections"].get("Multilingual Retrieval Hints", "")
    zh_fault_match = re.search(r"中文故障名称:\s*(.+)", result["zh_fault_name"])
    if zh_fault_match:
        result["zh_fault_name"] = zh_fault_match.group(1).strip()
    zh_symptom_match = re.search(r"中文症状别名:\s*(.+)", result["sections"].get("Multilingual Retrieval Hints", ""))
    if zh_symptom_match:
        result["zh_symptom_aliases"] = zh_symptom_match.group(1).strip()

    return result


def _build_case_answer_from_chunk(question: str, chunk: dict) -> str:
    parsed = _parse_case_chunk(chunk.get("content", ""))
    if not parsed["case_title"]:
        return ""

    is_chinese = bool(re.search(r"[\u4e00-\u9fff]", question or ""))
    machine_info = parsed["machine_info"]
    sections = parsed["sections"]
    image_html = ""
    if parsed["image_urls"]:
        image_html = f'\n<img src="{parsed["image_urls"][0]}" alt="Repair Image" width="300">\n'

    if is_chinese:
        title = parsed["zh_fault_name"] or parsed["case_title"]
        lines = [
            "### 1. 问题识别",
            f"当前匹配到的故障为：{title}。",
        ]
        if machine_info.get("Model"):
            lines.append(f"机型：{machine_info['Model']}")
        if machine_info.get("Machine No."):
            lines.append(f"机号：{machine_info['Machine No.']}")
        lines.extend([
            "",
            "### 2. 对应案例",
            f"案例 ID：{parsed['case_id'] or '未知'}",
            f"案例标题：{parsed['case_title']}",
        ])
        symptom = sections.get("Symptom")
        if symptom:
            lines.extend(["", "### 3. 故障现象", symptom])
        possible_causes = sections.get("Possible Causes")
        if possible_causes:
            lines.extend(["", "### 4. 可能原因", possible_causes])
        cause_analysis = sections.get("Cause Analysis")
        if cause_analysis:
            lines.extend(["", "### 5. 原因分析", cause_analysis])
        troubleshooting = sections.get("Troubleshooting Steps")
        if troubleshooting:
            lines.extend(["", "### 6. 排查步骤", troubleshooting])
        maintenance_plan = sections.get("Maintenance Plan")
        if maintenance_plan:
            lines.extend(["", "### 7. 维修方案", maintenance_plan])
        effect = sections.get("Effect Confirmation")
        if effect:
            lines.extend(["", "### 8. 效果确认", effect])
        prevention = sections.get("Prevention / Suggestion")
        if prevention:
            lines.extend(["", "### 9. 预防建议", prevention])
        if image_html:
            lines.extend(["", "### 10. 相关图片", image_html.strip()])
        return "\n".join(lines).strip()

    lines = [
        "### 1. Problem Identification",
        f"The matched fault is: {parsed['case_title']}.",
    ]
    if machine_info.get("Model"):
        lines.append(f"Model: {machine_info['Model']}")
    if machine_info.get("Machine No."):
        lines.append(f"Machine No.: {machine_info['Machine No.']}")
    lines.extend([
        "",
        "### 2. Case Reference",
        f"Case ID: {parsed['case_id'] or 'Unknown'}",
        f"Case Title: {parsed['case_title']}",
    ])
    for heading, label in [
        ("Symptom", "### 3. Symptom"),
        ("Possible Causes", "### 4. Possible Causes"),
        ("Cause Analysis", "### 5. Cause Analysis"),
        ("Troubleshooting Steps", "### 6. Troubleshooting Steps"),
        ("Maintenance Plan", "### 7. Maintenance Solution"),
        ("Effect Confirmation", "### 8. Effect Confirmation"),
        ("Prevention / Suggestion", "### 9. Prevention Suggestions"),
    ]:
        value = sections.get(heading)
        if value:
            lines.extend(["", label, value])
    if image_html:
        lines.extend(["", "### 10. Related Image", image_html.strip()])
    return "\n".join(lines).strip()


def _needs_case_answer_fallback(answer: str, chunks: list[dict]) -> bool:
    if not answer or not chunks:
        return False

    normalized = answer.lower()
    return any(pattern in normalized for pattern in MISSING_DETAIL_PATTERNS)


def _prefer_grounded_case_answer(question: str, chunks: list[dict], history_mode: str, answer: str) -> str:
    if history_mode != "independent" or not chunks:
        return answer

    grounded = _build_case_answer_from_chunk(question, chunks[0])
    return grounded or answer


def _build_episode_memory_payload(question: str, answer: str, chunk: dict) -> tuple[str, dict]:
    parsed = _parse_case_chunk(chunk.get("content", ""))
    machine_info = parsed.get("machine_info", {})
    sections = parsed.get("sections", {})
    symptom = sections.get("Symptom", "")
    maintenance_plan = sections.get("Maintenance Plan", "")
    cause_analysis = sections.get("Cause Analysis", "")
    answer_summary = _collapse_whitespace(answer)[:320]
    memory_lines = [
        f"User issue: {_collapse_whitespace(question)}",
        f"Matched case: {parsed.get('case_id', '')} {parsed.get('case_title', '')}".strip(),
        f"Chinese fault name: {parsed.get('zh_fault_name', '')}",
        f"Chinese symptom aliases: {parsed.get('zh_symptom_aliases', '')}",
        f"Symptom: {symptom}",
        f"Cause analysis: {cause_analysis}",
        f"Maintenance plan: {maintenance_plan}",
        f"Answer summary: {answer_summary}",
        f"Source document: {chunk.get('document_name', '')}",
        f"Model: {machine_info.get('Model', '')}",
        f"Machine No.: {machine_info.get('Machine No.', '')}",
    ]
    metadata = {
        "memory_kind": "episode",
        "case_id": parsed.get("case_id", ""),
        "case_title": parsed.get("case_title", ""),
        "source_doc": chunk.get("document_name", ""),
        "symptom": symptom,
        "maintenance_plan": maintenance_plan,
        "model": machine_info.get("Model", ""),
        "machine_no": machine_info.get("Machine No.", ""),
    }
    return "\n".join(line for line in memory_lines if line.rstrip(": ")).strip(), metadata


def _store_episode_memory(session_id: str, question: str, answer: str, chunks: list[dict], context_mode: str) -> dict | None:
    if context_mode != "independent" or not chunks:
        return None
    memory_client = _get_mem0_client()
    if memory_client is None:
        return None
    memory_text, metadata = _build_episode_memory_payload(question, answer, chunks[0])
    try:
        result = memory_client.add(
            memory_text,
            infer=False,
            metadata=metadata,
            **_mem0_scope(session_id),
        )
    except Exception:
        return None
    results = (result or {}).get("results", [])
    return results[0] if results else None


# ── Models ─────────────────────────────────────────────────────────────────────

class SessionCreate(BaseModel):
    name: str = "Diagnostic Session"

class ChatMessage(BaseModel):
    session_id: str
    question: str
    stream: bool = True
    metadata_filters: dict | None = None


class RetrievalRequest(BaseModel):
    question: str
    metadata_filters: dict | None = None
    page_size: int = DEFAULT_RETRIEVAL_PAGE_SIZE


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.post("/sessions")
async def create_session(body: SessionCreate):
    """Create a new app-managed diagnostic session."""
    chat = await get_chat_details()
    greeting = ((chat.get("prompt") or {}).get("opener") or "").strip()
    if not greeting:
        greeting = "Hello! Describe the excavator fault and I will help diagnose it."
    session = _new_session_record(body.name, greeting)
    LOCAL_SESSIONS[session["id"]] = session
    return {
        "session_id": session["id"],
        "name": session["name"],
        "created_at": session["create_date"],
        "greeting": session["messages"][0]["content"],
    }


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get app-managed conversation history for a session."""
    session = await _get_local_session(session_id)
    return {
        "session_id": session["id"],
        "name": session["name"],
        "messages": session.get("messages", []),
        "updated_at": session.get("update_date"),
    }


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete an app-managed session."""
    if session_id not in LOCAL_SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
    LOCAL_SESSIONS.pop(session_id, None)
    memory_client = _get_mem0_client()
    if memory_client is not None:
        try:
            memory_client.delete_all(**_mem0_scope(session_id))
        except Exception:
            pass
    return {"deleted": session_id}


@app.get("/sessions/{session_id}/memories")
async def get_session_memories(session_id: str, limit: int = 20):
    await _get_local_session(session_id)
    memory_client = _get_mem0_client()
    if memory_client is None:
        return {"session_id": session_id, "enabled": False, "memories": []}
    try:
        result = memory_client.get_all(limit=max(1, min(limit, 100)), **_mem0_scope(session_id))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Mem0 read failed: {exc}")
    return {
        "session_id": session_id,
        "enabled": True,
        "memories": _compact_memory_candidates((result or {}).get("results", [])),
    }


@app.post("/chat")
async def chat(body: ChatMessage):
    """
    Send a message in an app-managed session.
    History is assembled per request, so we can choose whether to include prior
    turns before delegating retrieval+generation to RAGFlow.
    """
    session = await _get_local_session(body.session_id)
    chat_id = await get_chat_id()
    url = f"{RAGFLOW_BASE_URL}/api/v1/chats_openai/{chat_id}/chat/completions"
    request_id = uuid4().hex
    request_started_at = time.perf_counter()
    langfuse_client = _get_langfuse_client()
    trace_id = None
    trace_url = None

    _emit_observability_log(
        "chat.request.start",
        request_id=request_id,
        session_id=session["id"],
        question=body.question,
        stream=body.stream,
        metadata_filters=body.metadata_filters or {},
    )

    metadata_condition, applied_filters = build_metadata_condition(body.question, body.metadata_filters)
    retrieval_document_ids: list[str] | None = None
    current_turn_chunks: list[dict] = []
    retrieval_started_at = time.perf_counter()
    if applied_filters:
        matching_documents = await get_matching_documents(applied_filters)
        if not matching_documents:
            no_match_answer = (
                "No maintenance case matched the requested metadata filters: "
                + ", ".join(f"{key}={value}" for key, value in applied_filters.items())
            )
            if body.stream:
                async def no_match_stream():
                    yield f"data: {json.dumps({'text': no_match_answer, 'done': True, 'sources': [], 'applied_filters': applied_filters})}\n\n"
                    yield "data: [DONE]\n\n"

                return StreamingResponse(no_match_stream(), media_type="text/event-stream")

                return {
                "answer": no_match_answer,
                "session_id": session["id"],
                "applied_filters": applied_filters,
                    "sources": [],
                }

        retrieval_document_ids = [document["id"] for document in matching_documents]
        current_turn_chunks = await retrieve_case_chunks(
            body.question,
            document_ids=retrieval_document_ids,
            page_size=len(matching_documents),
        )
    else:
        current_turn_chunks = await retrieve_case_chunks(body.question)
    retrieval_duration_ms = round((time.perf_counter() - retrieval_started_at) * 1000, 2)
    retrieval_log_payload = {
        "duration_ms": retrieval_duration_ms,
        "question": body.question,
        "applied_filters": applied_filters,
        "document_ids": retrieval_document_ids or [],
        "hit_count": len(current_turn_chunks),
        "top_doc": (current_turn_chunks[0].get("document_name") if current_turn_chunks else ""),
        "top_doc_ids": _extract_doc_ids(current_turn_chunks),
    }
    _emit_observability_log(
        "chat.retrieval.complete",
        request_id=request_id,
        session_id=session["id"],
        stage="retrieval",
        retrieval=retrieval_log_payload,
        diagnosis={
            "stage": "retrieval" if not current_turn_chunks else "none",
            "code": "no_hit" if not current_turn_chunks else "grounded",
        },
    )

    retrieval_hint = _build_retrieval_hint(current_turn_chunks[0]) if current_turn_chunks else ""
    memory_started_at = time.perf_counter()
    memory_hits = _search_mem0_memories(session["id"], body.question)
    memory_duration_ms = round((time.perf_counter() - memory_started_at) * 1000, 2)
    memory_log_payload = {
        "duration_ms": memory_duration_ms,
        "hit_count": len(memory_hits),
        "top_memory_case_id": (((memory_hits[0] if memory_hits else {}).get("metadata") or {}).get("case_id", "")),
        "top_memory_source_doc": (((memory_hits[0] if memory_hits else {}).get("metadata") or {}).get("source_doc", "")),
    }
    _emit_observability_log(
        "chat.memory.complete",
        request_id=request_id,
        session_id=session["id"],
        stage="memory",
        memory=memory_log_payload,
    )
    router_started_at = time.perf_counter()
    router = await _classify_context_mode(
        session,
        body.question,
        retrieval_hint,
        memory_hits,
    )
    selected_memories = _select_memory_candidates(memory_hits, router.get("memory_indexes", []))
    router, selected_memories = _normalize_router_decision(
        session,
        body.question,
        router,
        current_turn_chunks,
        memory_hits,
        selected_memories,
    )
    router_duration_ms = round((time.perf_counter() - router_started_at) * 1000, 2)
    router_log_payload = {
        "duration_ms": router_duration_ms,
        "mode": router["mode"],
        "reason": router["reason"],
        "selected_memory_count": len(selected_memories),
        "selected_memory_case_ids": [
            ((memory.get("metadata") or {}).get("case_id", ""))
            for memory in selected_memories
        ],
    }
    _emit_observability_log(
        "chat.router.complete",
        request_id=request_id,
        session_id=session["id"],
        stage="router",
        router=router_log_payload,
    )
    contextual_retrieval_query = _build_contextual_retrieval_query(
        session,
        body.question,
        router["mode"],
        selected_memories,
    )
    if contextual_retrieval_query != body.question:
        contextual_retrieval_started_at = time.perf_counter()
        current_turn_chunks = await retrieve_case_chunks(
            contextual_retrieval_query,
            document_ids=retrieval_document_ids,
            page_size=DEFAULT_RETRIEVAL_PAGE_SIZE if not retrieval_document_ids else len(retrieval_document_ids),
        )
        retrieval_hint = _build_retrieval_hint(current_turn_chunks[0]) if current_turn_chunks else retrieval_hint
        contextual_retrieval_payload = {
            "duration_ms": round((time.perf_counter() - contextual_retrieval_started_at) * 1000, 2),
            "query": contextual_retrieval_query,
            "hit_count": len(current_turn_chunks),
            "top_doc": (current_turn_chunks[0].get("document_name") if current_turn_chunks else ""),
        }
        _emit_observability_log(
            "chat.contextual_retrieval.complete",
            request_id=request_id,
            session_id=session["id"],
            stage="retrieval",
            retrieval=contextual_retrieval_payload,
            diagnosis={
                "stage": "retrieval" if not current_turn_chunks else "none",
                "code": "no_hit_after_rewrite" if not current_turn_chunks else "grounded",
            },
        )
    assembled_messages = _assemble_messages(
        session,
        body.question,
        context_mode=router["mode"],
        retrieval_hint=retrieval_hint,
        selected_memories=selected_memories,
    )
    debug_payload = {
        "context_mode": router["mode"],
        "context_reason": router["reason"],
        "memory_candidates": _compact_memory_candidates(memory_hits),
        "selected_memories": _compact_memory_candidates(selected_memories),
        "retrieval_hint": retrieval_hint,
        "contextual_retrieval_query": contextual_retrieval_query,
        "assembled_messages": [
            {
                "role": message["role"],
                "content_preview": _collapse_whitespace(message["content"])[:320],
            }
            for message in assembled_messages
        ],
    }
    debug_payload["request_id"] = request_id
    debug_payload["trace_id"] = trace_id
    debug_payload["trace_url"] = trace_url
    debug_payload["observability"] = {
        "retrieval": retrieval_log_payload,
        "memory": memory_log_payload,
        "router": router_log_payload,
        "log_path": OBSERVABILITY_LOG_PATH,
    }
    payload = {
        "model": "ragflow",
        "messages": assembled_messages,
        "stream": body.stream,
        "extra_body": {
            "reference": True,
        },
    }
    if metadata_condition:
        payload["extra_body"]["metadata_condition"] = metadata_condition

    if body.stream:
        async def stream_response():
            accumulated = ""
            sources: list[dict] = []
            generation_started_at = time.perf_counter()
            root_metadata = {
                "request_id": request_id,
                "stream": True,
                "retrieval": retrieval_log_payload,
                "memory": memory_log_payload,
                "router": router_log_payload,
                "applied_filters": applied_filters,
            }
            with _langfuse_attributes_context(session["id"], request_id):
                root_context = (
                    langfuse_client.start_as_current_observation(
                        name="chat_turn",
                        as_type="chain",
                        input={"question": body.question, "stream": True},
                        metadata=root_metadata,
                    )
                    if langfuse_client is not None else nullcontext(None)
                )
                with root_context as chat_observation:
                    local_trace_id = langfuse_client.get_current_trace_id() if langfuse_client is not None else None
                    local_trace_url = langfuse_client.get_trace_url() if langfuse_client is not None else None
                    generation_context = (
                        langfuse_client.start_as_current_observation(
                            name="ragflow_generation",
                            as_type="generation",
                            input={"messages": assembled_messages, "context_mode": router["mode"]},
                            metadata={"request_id": request_id},
                            model="ragflow",
                        )
                        if langfuse_client is not None else nullcontext(None)
                    )
                    with generation_context as generation_observation:
                        async with httpx.AsyncClient(timeout=120) as client:
                            async with client.stream(
                                "POST", url,
                                headers={"Authorization": f"Bearer {RAGFLOW_API_KEY}"},
                                json=payload,
                            ) as resp:
                                async for line in resp.aiter_lines():
                                    if line.startswith("data:"):
                                        raw = line[5:].strip()
                                        if raw == "[DONE]":
                                            yield "data: [DONE]\n\n"
                                            break
                                        try:
                                            chunk = json.loads(raw)
                                            choices = chunk.get("choices") or []
                                            if not choices:
                                                continue
                                            choice = choices[0]
                                            delta = choice.get("delta", {}) or {}
                                            piece = delta.get("content") or ""
                                            final_content = delta.get("final_content")
                                            finish_reason = choice.get("finish_reason")
                                            if piece:
                                                accumulated += piece
                                            if delta.get("reference"):
                                                sources = _source_preview_from_reference_chunks(delta["reference"])
                                            if finish_reason == "stop":
                                                raw_generated_answer = final_content or accumulated
                                                if final_content:
                                                    accumulated = final_content
                                                accumulated = _prefer_grounded_case_answer(
                                                    body.question,
                                                    current_turn_chunks,
                                                    router["mode"],
                                                    accumulated,
                                                )
                                                if _needs_case_answer_fallback(accumulated, current_turn_chunks):
                                                    fallback_answer = _build_case_answer_from_chunk(body.question, current_turn_chunks[0])
                                                    if fallback_answer:
                                                        accumulated = fallback_answer
                                                if not sources:
                                                    sources = _source_preview_from_chunks(current_turn_chunks)
                                                diagnosis = _diagnose_chat_turn(
                                                    question=body.question,
                                                    current_turn_chunks=current_turn_chunks,
                                                    raw_answer=raw_generated_answer,
                                                    final_answer=accumulated,
                                                    reference_chunks=[],
                                                    completion_fallback_reason=None,
                                                    context_mode=router["mode"],
                                                )
                                                generation_duration_ms = round((time.perf_counter() - generation_started_at) * 1000, 2)
                                                _emit_observability_log(
                                                    "chat.generation.complete",
                                                    request_id=request_id,
                                                    session_id=session["id"],
                                                    stage="generation",
                                                    generation={
                                                        "duration_ms": generation_duration_ms,
                                                        "raw_answer_preview": _preview_text(raw_generated_answer),
                                                        "final_answer_preview": _preview_text(accumulated),
                                                        "source_docs": [source.get("doc") for source in sources],
                                                        "context_mode": router["mode"],
                                                    },
                                                    diagnosis=diagnosis,
                                                )
                                                now = time.time()
                                                session["messages"].append({"role": "user", "content": body.question, "created_at": now})
                                                session["messages"].append({
                                                    "role": "assistant",
                                                    "content": accumulated,
                                                    "created_at": now,
                                                    "context_mode": router["mode"],
                                                    "context_reason": router["reason"],
                                                    "source_doc": (sources[0]["doc"] if sources else ""),
                                                })
                                                session["update_date"] = now
                                                stored_memory = _store_episode_memory(
                                                    session["id"],
                                                    body.question,
                                                    accumulated,
                                                    current_turn_chunks,
                                                    router["mode"],
                                                )
                                                final_debug = {
                                                    **debug_payload,
                                                    "trace_id": local_trace_id,
                                                    "trace_url": local_trace_url,
                                                    "diagnosis": diagnosis,
                                                    "stored_memory": _compact_memory_candidates([stored_memory])[0] if stored_memory else None,
                                                }
                                                _safe_langfuse_update(
                                                    generation_observation,
                                                    output={"answer_preview": _preview_text(accumulated), "diagnosis": diagnosis},
                                                    metadata={"source_docs": [source.get("doc") for source in sources]},
                                                    level="WARNING" if diagnosis["stage"] != "none" else "DEFAULT",
                                                )
                                                _safe_langfuse_update(
                                                    chat_observation,
                                                    output={"answer_preview": _preview_text(accumulated), "diagnosis": diagnosis},
                                                    metadata={**root_metadata, "trace_url": local_trace_url},
                                                    level="WARNING" if diagnosis["stage"] != "none" else "DEFAULT",
                                                )
                                                if langfuse_client is not None and diagnosis["stage"] != "none":
                                                    try:
                                                        langfuse_client.score_current_trace(
                                                            name="issue_stage",
                                                            value=diagnosis["stage"],
                                                            data_type="CATEGORICAL",
                                                            comment=diagnosis["summary"],
                                                            metadata={"issue_code": diagnosis["code"]},
                                                        )
                                                        langfuse_client.score_current_trace(
                                                            name="issue_code",
                                                            value=diagnosis["code"],
                                                            data_type="CATEGORICAL",
                                                        )
                                                    except Exception:
                                                        pass
                                                _emit_observability_log(
                                                    "chat.request.complete",
                                                    request_id=request_id,
                                                    session_id=session["id"],
                                                    duration_ms=round((time.perf_counter() - request_started_at) * 1000, 2),
                                                    trace_id=local_trace_id,
                                                    trace_url=local_trace_url,
                                                    diagnosis=diagnosis,
                                                )
                                                yield f"data: {json.dumps({'text': accumulated, 'done': True, 'sources': sources, 'applied_filters': applied_filters, 'session_id': session['id'], 'context_mode': router['mode'], 'debug': final_debug})}\n\n"
                                            else:
                                                yield f"data: {json.dumps({'text': accumulated, 'done': False})}\n\n"
                                        except json.JSONDecodeError:
                                            pass

        return StreamingResponse(stream_response(), media_type="text/event-stream")

    else:
        reference_chunks: list[dict] = []
        completion_fallback_reason = None
        root_metadata = {
            "request_id": request_id,
            "stream": False,
            "retrieval": retrieval_log_payload,
            "memory": memory_log_payload,
            "router": router_log_payload,
            "applied_filters": applied_filters,
        }
        raw_model_answer = ""
        with _langfuse_attributes_context(session["id"], request_id):
            root_context = (
                langfuse_client.start_as_current_observation(
                    name="chat_turn",
                    as_type="chain",
                    input={"question": body.question, "stream": False},
                    metadata=root_metadata,
                )
                if langfuse_client is not None else nullcontext(None)
            )
            with root_context as chat_observation:
                trace_id = langfuse_client.get_current_trace_id() if langfuse_client is not None else None
                trace_url = langfuse_client.get_trace_url() if langfuse_client is not None else None
                generation_started_at = time.perf_counter()
                generation_context = (
                    langfuse_client.start_as_current_observation(
                        name="ragflow_generation",
                        as_type="generation",
                        input={"messages": assembled_messages, "context_mode": router["mode"]},
                        metadata={"request_id": request_id},
                        model="ragflow",
                    )
                    if langfuse_client is not None else nullcontext(None)
                )
                with generation_context as generation_observation:
                    try:
                        async with httpx.AsyncClient(timeout=120) as client:
                            resp = await client.post(
                                url,
                                headers={"Authorization": f"Bearer {RAGFLOW_API_KEY}"},
                                json={**payload, "stream": False},
                            )
                        data = resp.json()
                        if data.get("code") and data.get("code") != 0:
                            raise HTTPException(status_code=500, detail=data.get("message"))
                        choices = data.get("choices") or []
                        if not choices:
                            raise HTTPException(status_code=500, detail="RAGFlow returned no completion choices")
                        message = choices[0].get("message", {}) or {}
                        reference_chunks = message.get("reference") or []
                        raw_model_answer = message.get("content", "")
                        final_answer = _prefer_grounded_case_answer(
                            body.question,
                            current_turn_chunks,
                            router["mode"],
                            raw_model_answer,
                        )
                        if _needs_case_answer_fallback(raw_model_answer, current_turn_chunks):
                            fallback_answer = _build_case_answer_from_chunk(body.question, current_turn_chunks[0])
                            if fallback_answer:
                                final_answer = fallback_answer
                    except (httpx.HTTPError, HTTPException) as exc:
                        completion_fallback_reason = str(exc)
                        if not current_turn_chunks:
                            raise
                        final_answer = _build_case_answer_from_chunk(body.question, current_turn_chunks[0]) or (
                            "The language-model completion timed out, but the current matched maintenance case is still available."
                        )
                    diagnosis = _diagnose_chat_turn(
                        question=body.question,
                        current_turn_chunks=current_turn_chunks,
                        raw_answer=raw_model_answer,
                        final_answer=final_answer,
                        reference_chunks=reference_chunks,
                        completion_fallback_reason=completion_fallback_reason,
                        context_mode=router["mode"],
                    )
                    generation_duration_ms = round((time.perf_counter() - generation_started_at) * 1000, 2)
                    _emit_observability_log(
                        "chat.generation.complete",
                        request_id=request_id,
                        session_id=session["id"],
                        stage="generation",
                        generation={
                            "duration_ms": generation_duration_ms,
                            "raw_answer_preview": _preview_text(raw_model_answer),
                            "final_answer_preview": _preview_text(final_answer),
                            "source_docs": [
                                source.get("doc")
                                for source in (_source_preview_from_chunks(current_turn_chunks) or _source_preview_from_reference_chunks(reference_chunks))
                            ],
                            "context_mode": router["mode"],
                            "completion_fallback_reason": completion_fallback_reason,
                        },
                        diagnosis=diagnosis,
                    )
                    _safe_langfuse_update(
                        generation_observation,
                        output={"answer_preview": _preview_text(final_answer), "diagnosis": diagnosis},
                        metadata={
                            "reference_docs": [
                                source.get("doc")
                                for source in (_source_preview_from_chunks(current_turn_chunks) or _source_preview_from_reference_chunks(reference_chunks))
                            ],
                            "completion_fallback_reason": completion_fallback_reason,
                        },
                        level="WARNING" if diagnosis["stage"] != "none" else "DEFAULT",
                    )
                    _safe_langfuse_update(
                        chat_observation,
                        output={"answer_preview": _preview_text(final_answer), "diagnosis": diagnosis},
                        metadata={**root_metadata, "trace_url": trace_url},
                        level="WARNING" if diagnosis["stage"] != "none" else "DEFAULT",
                    )
                    if langfuse_client is not None and diagnosis["stage"] != "none":
                        try:
                            langfuse_client.score_current_trace(
                                name="issue_stage",
                                value=diagnosis["stage"],
                                data_type="CATEGORICAL",
                                comment=diagnosis["summary"],
                                metadata={"issue_code": diagnosis["code"]},
                            )
                            langfuse_client.score_current_trace(
                                name="issue_code",
                                value=diagnosis["code"],
                                data_type="CATEGORICAL",
                            )
                        except Exception:
                            pass
        now = time.time()
        session["messages"].append({"role": "user", "content": body.question, "created_at": now})
        session["messages"].append({
            "role": "assistant",
            "content": final_answer,
            "created_at": now,
            "context_mode": router["mode"],
            "context_reason": router["reason"],
            "source_doc": (
                (_source_preview_from_chunks(current_turn_chunks) or _source_preview_from_reference_chunks(reference_chunks) or [{}])[0].get("doc", "")
            ),
        })
        session["update_date"] = now
        stored_memory = _store_episode_memory(
            session["id"],
            body.question,
            final_answer,
            current_turn_chunks,
            router["mode"],
        )
        debug_payload["stored_memory"] = _compact_memory_candidates([stored_memory])[0] if stored_memory else None
        debug_payload["completion_fallback_reason"] = completion_fallback_reason
        debug_payload["trace_id"] = trace_id
        debug_payload["trace_url"] = trace_url
        debug_payload["diagnosis"] = diagnosis
        _emit_observability_log(
            "chat.request.complete",
            request_id=request_id,
            session_id=session["id"],
            duration_ms=round((time.perf_counter() - request_started_at) * 1000, 2),
            trace_id=trace_id,
            trace_url=trace_url,
            diagnosis=diagnosis,
        )
        return {
            "answer": final_answer,
            "session_id": session["id"],
            "applied_filters": applied_filters,
            "sources": _source_preview_from_chunks(current_turn_chunks) or _source_preview_from_reference_chunks(reference_chunks),
            "context_mode": router["mode"],
            "debug": debug_payload,
        }


@app.post("/retrieve_cases")
async def retrieve_cases(body: RetrievalRequest):
    """Run metadata-aware hybrid retrieval directly against the case dataset."""
    request_id = uuid4().hex
    started_at = time.perf_counter()
    chat = await get_chat_details()
    assistant_rerank_model = (chat.get("prompt") or {}).get("rerank_model") or None
    _, applied_filters = build_metadata_condition(body.question, body.metadata_filters)
    matching_documents = await get_matching_documents(applied_filters)
    if applied_filters and not matching_documents:
        chunks = []
    else:
        chunks = await retrieve_case_chunks(
            body.question,
            document_ids=[document["id"] for document in matching_documents] if applied_filters else None,
            page_size=body.page_size if not applied_filters else len(matching_documents),
        )
    diagnosis = {
        "status": "issue" if not chunks else "ok",
        "stage": "retrieval" if not chunks else "none",
        "code": "no_hit" if not chunks else "grounded",
        "summary": "No chunks matched the current retrieval query." if not chunks else "Retrieval returned grounded case chunks.",
    }
    _emit_observability_log(
        "retrieve_cases.complete",
        request_id=request_id,
        session_id=None,
        duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
        question=body.question,
        applied_filters=applied_filters,
        hit_count=len(chunks),
        top_doc=(chunks[0].get("document_name") if chunks else ""),
        diagnosis=diagnosis,
    )

    return {
        "request_id": request_id,
        "question": body.question,
        "applied_filters": applied_filters,
        "matched_documents": [document.get("name", "") for document in matching_documents[:10]],
        "hybrid_retrieval": {
            "keyword_weight": DEFAULT_KEYWORD_WEIGHT,
            "vector_weight": DEFAULT_VECTOR_WEIGHT,
            "similarity_threshold": DEFAULT_SIMILARITY_THRESHOLD,
            "top_k": DEFAULT_RETRIEVAL_TOP_K,
            "assistant_rerank_model": assistant_rerank_model,
        },
        "diagnosis": diagnosis,
        "hits": [
            {
                "document_name": chunk.get("document_name", ""),
                "content": chunk.get("content", ""),
                "similarity": chunk.get("similarity"),
                "vector_similarity": chunk.get("vector_similarity"),
                "term_similarity": chunk.get("term_similarity"),
            }
            for chunk in chunks
        ],
    }


@app.get("/health")
async def health():
    """Health check — also verifies RAGFlow connection."""
    chat_id = await get_chat_id()
    dataset_id = await get_dataset_id()
    chat = await get_chat_details()
    assistant_rerank_model = (chat.get("prompt") or {}).get("rerank_model") or None
    return {
        "status": "ok",
        "chat_id": chat_id,
        "dataset_id": dataset_id,
        "ragflow": RAGFLOW_BASE_URL,
        "history_router": {
            "enabled": bool(_get_gemini_client()),
            "model": HISTORY_ROUTER_MODEL,
            "memory_window": HISTORY_MEMORY_WINDOW,
        },
        "mem0": {
            "enabled": bool(_get_mem0_client()),
            "collection_name": MEM0_COLLECTION_NAME,
            "search_limit": MEM0_SEARCH_LIMIT,
            "distance_threshold": MEM0_DISTANCE_THRESHOLD,
            "max_injected": MEM0_MAX_INJECTED,
        },
        "hybrid_retrieval": {
            "keyword_weight": DEFAULT_KEYWORD_WEIGHT,
            "vector_weight": DEFAULT_VECTOR_WEIGHT,
            "similarity_threshold": DEFAULT_SIMILARITY_THRESHOLD,
            "assistant_rerank_model": assistant_rerank_model,
        },
        "observability": {
            "log_path": OBSERVABILITY_LOG_PATH,
            "log_level": OBSERVABILITY_LOG_LEVEL,
            "langfuse_enabled": bool(_get_langfuse_client()),
            "langfuse_auth_ok": _check_langfuse_auth() if _get_langfuse_client() is not None else False,
            "langfuse_base_url": LANGFUSE_BASE_URL if _get_langfuse_client() is not None else None,
        },
    }


@app.get("/observability/recent")
async def recent_observability(limit: int = 50, session_id: str | None = None, request_id: str | None = None, issue_stage: str | None = None):
    events = _tail_observability_logs(limit=limit, session_id=session_id, request_id=request_id, issue_stage=issue_stage)
    return {
        "log_path": OBSERVABILITY_LOG_PATH,
        "count": len(events),
        "events": events,
    }


@app.get("/", response_class=HTMLResponse)
async def ui():
    """Simple chat UI for testing."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Excavator Maintenance Assistant</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: system-ui, sans-serif; background: #0f0f0f; color: #e0e0e0; height: 100vh; display: flex; flex-direction: column; }
    #header { padding: 16px 24px; background: #1a1a1a; border-bottom: 1px solid #2a2a2a; }
    #header h1 { font-size: 18px; color: #4ade80; }
    #header p { font-size: 12px; color: #888; margin-top: 2px; }
    #messages { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 16px; }
    .msg { max-width: 75%; padding: 12px 16px; border-radius: 12px; font-size: 14px; line-height: 1.6; white-space: pre-wrap; }
    .msg.user { align-self: flex-end; background: #1d4ed8; color: #fff; border-bottom-right-radius: 2px; }
    .msg.assistant { align-self: flex-start; background: #1e1e1e; border: 1px solid #2a2a2a; border-bottom-left-radius: 2px; }
    .msg img { max-width: 300px; display: block; margin-top: 8px; border-radius: 6px; }
    .msg-text { white-space: pre-wrap; }
    .source-ref { display: inline-flex; align-items: center; justify-content: center; min-width: 24px; height: 22px; margin-left: 4px; padding: 0 8px; border: 1px solid #355070; border-radius: 999px; background: #182535; color: #9dc1ff; font-size: 12px; cursor: pointer; }
    .source-ref:hover { background: #22344a; }
    .source-list { margin-top: 12px; display: flex; flex-direction: column; gap: 8px; }
    .source-toggle { width: 100%; text-align: left; padding: 8px 10px; border-radius: 8px; border: 1px solid #2f3d4d; background: #151b22; color: #d6e2f0; cursor: pointer; font-size: 12px; }
    .source-toggle:hover { background: #1b2430; }
    .source-panel { margin-top: 6px; padding: 10px; border-radius: 8px; background: #101418; border: 1px solid #2a2f35; color: #c8d1da; font-size: 12px; line-height: 1.5; white-space: pre-wrap; display: none; }
    #input-bar { padding: 16px 24px; background: #1a1a1a; border-top: 1px solid #2a2a2a; display: flex; gap: 10px; }
    #input-bar input { flex: 1; padding: 10px 14px; background: #2a2a2a; border: 1px solid #3a3a3a; border-radius: 8px; color: #e0e0e0; font-size: 14px; outline: none; }
    #input-bar button { padding: 10px 20px; background: #4ade80; color: #000; border: none; border-radius: 8px; cursor: pointer; font-weight: 600; font-size: 14px; }
    #input-bar button:disabled { background: #3a3a3a; color: #666; cursor: default; }
    #session-info { font-size: 11px; color: #555; padding: 4px 24px; background: #0f0f0f; }
  </style>
</head>
<body>
  <div id="header">
    <h1>🚜 Excavator Maintenance Assistant</h1>
    <p>Multi-turn diagnostic conversation — powered by case-level chunks, metadata filtering, and RAGFlow hybrid retrieval</p>
  </div>
  <div id="session-info">Initializing session...</div>
  <div id="messages"></div>
  <div id="input-bar">
    <input id="q" type="text" placeholder="Describe the fault or ask a maintenance question..." />
    <button id="send-btn" onclick="sendMessage()">Send</button>
  </div>
  <script>
    let sessionId = null;

    async function init() {
      const res = await fetch('/sessions', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({name: 'Diagnostic Session ' + new Date().toLocaleTimeString()})
      });
      const data = await res.json();
      sessionId = data.session_id;
      document.getElementById('session-info').textContent = `Session: ${sessionId}`;
      appendMessage('assistant', data.greeting || 'Hello! Describe the excavator fault and I will help diagnose it.');
    }

    function renderText(text) {
      // Extract raw HTML img tags first, replace with placeholders
      const imgs = [];
      const refs = [];
      const marker = (index) => `__IMG_TOKEN_${index}__`;
      const refMarker = (index) => `__REF_TOKEN_${index}__`;
      let withPlaceholders = text.replace(/<img[\\s][^>]*>/gi, (match) => {
        imgs.push(match);
        return marker(imgs.length - 1);
      });
      // Support standard Markdown image syntax as well.
      withPlaceholders = withPlaceholders.replace(/!\\[([^\\]]*)\\]\\(([^)\\s]+)\\)/g, (_, alt, url) => {
        const safeAlt = String(alt).replace(/&/g, '&amp;').replace(/"/g, '&quot;');
        imgs.push(`<img src="${url}" alt="${safeAlt}">`);
        return marker(imgs.length - 1);
      });
      withPlaceholders = withPlaceholders.replace(/\\[ID:(\\d+)\\]/g, (_, index) => {
        const numericIndex = Number(index);
        refs.push(`<button type="button" class="source-ref" data-source-index="${numericIndex}">[${numericIndex + 1}]</button>`);
        return refMarker(refs.length - 1);
      });
      // Escape remaining HTML
      let safe = withPlaceholders
        .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
        .replace(/[*][*](.*?)[*][*]/g, '<strong>$1</strong>');
      // Restore img tags
      safe = safe.replace(/__IMG_TOKEN_([0-9]+)__/g, (_, i) => imgs[parseInt(i)]);
      safe = safe.replace(/__REF_TOKEN_([0-9]+)__/g, (_, i) => refs[parseInt(i)]);
      return safe;
    }

    function escapeHtml(text) {
      return String(text || '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }

    function renderSources(sources) {
      if (!sources || !sources.length) return '';
      return `<div class="source-list">` + sources.map((source, index) => {
        const doc = escapeHtml(source.doc || `Source ${index + 1}`);
        const chunk = escapeHtml(source.chunk || '');
        return `
          <div>
            <button type="button" class="source-toggle" data-source-index="${index}">Source ${index + 1}: ${doc}</button>
            <div class="source-panel" data-source-panel="${index}">${chunk}</div>
          </div>
        `;
      }).join('') + `</div>`;
    }

    function bindSourceInteractions(div) {
      const toggleSource = (index) => {
        const panel = div.querySelector(`[data-source-panel="${index}"]`);
        if (!panel) return;
        panel.style.display = panel.style.display === 'block' ? 'none' : 'block';
      };

      div.querySelectorAll('[data-source-index]').forEach((el) => {
        el.addEventListener('click', () => toggleSource(el.getAttribute('data-source-index')));
      });
    }

    function setMessageContent(div, text, sources = []) {
      div.innerHTML = `<div class="msg-text">${renderText(text)}</div>${renderSources(sources)}`;
      bindSourceInteractions(div);
    }

    function appendMessage(role, text, sources = []) {
      const div = document.createElement('div');
      div.className = `msg ${role}`;
      setMessageContent(div, text, sources);
      document.getElementById('messages').appendChild(div);
      div.scrollIntoView({behavior: 'smooth'});
      return div;
    }

    async function sendMessage() {
      const input = document.getElementById('q');
      const btn = document.getElementById('send-btn');
      const q = input.value.trim();
      if (!q || !sessionId) return;
      input.value = '';
      btn.disabled = true;

      appendMessage('user', q);
      const bubble = appendMessage('assistant', '...');

      const res = await fetch('/chat', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({session_id: sessionId, question: q, stream: true})
      });

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let full = '';
      let sources = [];

      while (true) {
        const {done, value} = await reader.read();
        if (done) break;
        const lines = decoder.decode(value).split('\\n');
        for (const line of lines) {
          if (!line.startsWith('data:')) continue;
          const raw = line.slice(5).trim();
          if (raw === '[DONE]') break;
          try {
            const chunk = JSON.parse(raw);
            full = chunk.text || full;
            if (chunk.sources) {
              sources = chunk.sources;
            }
            if (chunk.session_id) {
              sessionId = chunk.session_id;
              document.getElementById('session-info').textContent = `Session: ${sessionId}`;
            }
            setMessageContent(bubble, full, sources);
            bubble.scrollIntoView({behavior: 'smooth'});
          } catch(e) {}
        }
      }

      btn.disabled = false;
      document.getElementById('q').focus();
    }

    document.getElementById('q').addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
    });

    init();
  </script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860, reload=False)
