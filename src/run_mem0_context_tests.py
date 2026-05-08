#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run multi-turn context-management tests against the local chat server.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import httpx

BASE_URL = "http://127.0.0.1:7860"
ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"


TEST_SUITES = [
    {
        "name": "suite_1_recent_vs_independent",
        "description": "Immediate follow-ups should use recent history; symptom switches should stay independent.",
        "turns": [
            {"question": "我的车子冒蓝烟", "expect_doc": "excavator_repair_case_en_page1.md", "expect_mode": "independent"},
            {"question": "为什么", "expect_doc": "excavator_repair_case_en_page1.md", "expect_mode": "recent_history"},
            {"question": "车子抖动得厉害", "expect_doc": "excavator_repair_case_en_page2.md", "expect_mode": "independent"},
            {"question": "怎么修", "expect_doc": "excavator_repair_case_en_page2.md", "expect_mode": "recent_history"},
            {"question": "水温高报警一直响", "expect_doc": "excavator_repair_case_en_page3.md", "expect_mode": "independent"},
            {"question": "这个是假高温吗", "expect_doc": "excavator_repair_case_en_page3.md", "expect_mode": "recent_history"},
            {"question": "停久了不好启动", "expect_doc": "excavator_repair_case_en_page4.md", "expect_mode": "independent"},
            {"question": "最后怎么修好的", "expect_doc": "excavator_repair_case_en_page4.md", "expect_mode": "recent_history"},
            {"question": "冒蓝色烟是什么原因", "expect_doc": "excavator_repair_case_en_page1.md", "expect_mode": "episodic_memory"},
            {"question": "给我那个案例的维修方案", "expect_doc": "excavator_repair_case_en_page1.md", "expect_mode": "recent_history"},
        ],
    },
    {
        "name": "suite_2_long_gap_revival",
        "description": "Older topics should be revived through Mem0 episodic memory after several unrelated turns.",
        "turns": [
            {"question": "车子抖动", "expect_doc": "excavator_repair_case_en_page2.md", "expect_mode": "independent"},
            {"question": "冷却液高温报警", "expect_doc": "excavator_repair_case_en_page3.md", "expect_mode": "independent"},
            {"question": "我的车停久了偶尔打不着火", "expect_doc": "excavator_repair_case_en_page4.md", "expect_mode": "independent"},
            {"question": "尾气冒蓝烟", "expect_doc": "excavator_repair_case_en_page1.md", "expect_mode": "independent"},
            {"question": "刚才那个抖动案例是换什么", "expect_doc": "excavator_repair_case_en_page2.md", "expect_mode": "episodic_memory"},
            {"question": "高温那个是不是传感器误报", "expect_doc": "excavator_repair_case_en_page3.md", "expect_mode": "episodic_memory"},
            {"question": "启动困难是不是油箱脏了", "expect_doc": "excavator_repair_case_en_page4.md", "expect_mode": "episodic_memory"},
            {"question": "蓝烟是不是喷油器", "expect_doc": "excavator_repair_case_en_page1.md", "expect_mode": "episodic_memory"},
            {"question": "为什么", "expect_doc": "excavator_repair_case_en_page1.md", "expect_mode": "recent_history"},
            {"question": "给我图片", "expect_doc": "excavator_repair_case_en_page1.md", "expect_mode": "recent_history"},
        ],
    },
    {
        "name": "suite_3_similarity_rephrases",
        "description": "Rephrased symptoms should pull the matching old episode instead of the whole chat history.",
        "turns": [
            {"question": "我的车冒蓝烟", "expect_doc": "excavator_repair_case_en_page1.md", "expect_mode": "independent"},
            {"question": "车子抖动", "expect_doc": "excavator_repair_case_en_page2.md", "expect_mode": "independent"},
            {"question": "停久了不好启动", "expect_doc": "excavator_repair_case_en_page4.md", "expect_mode": "independent"},
            {"question": "高温报警", "expect_doc": "excavator_repair_case_en_page3.md", "expect_mode": "independent"},
            {"question": "蓝烟这个毛病通常怎么处理", "expect_doc": "excavator_repair_case_en_page1.md", "expect_mode": "episodic_memory"},
            {"question": "抖动那个是不是机脚垫", "expect_doc": "excavator_repair_case_en_page2.md", "expect_mode": "episodic_memory"},
            {"question": "高温报警但实测温度不高，是不是线路问题", "expect_doc": "excavator_repair_case_en_page3.md", "expect_mode": "episodic_memory"},
            {"question": "停放很久之后偶尔发动不起来，是不是油箱底部有杂质", "expect_doc": "excavator_repair_case_en_page4.md", "expect_mode": "episodic_memory"},
            {"question": "继续说第一个问题的原因", "expect_doc": "excavator_repair_case_en_page1.md", "expect_mode": "episodic_memory"},
            {"question": "为什么会这样", "expect_doc": "excavator_repair_case_en_page1.md", "expect_mode": "recent_history"},
        ],
    },
]


def _truncate(text: str, limit: int = 240) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _md_escape(text: str) -> str:
    return str(text or "").replace("|", "\\|")


def _run_suite(client: httpx.Client, suite: dict) -> dict:
    session_resp = client.post("/sessions", json={"name": suite["name"]})
    session_resp.raise_for_status()
    session_id = session_resp.json()["session_id"]
    print(f"[suite] {suite['name']} session={session_id}", flush=True)

    suite_results = {
        "name": suite["name"],
        "description": suite["description"],
        "session_id": session_id,
        "turns": [],
    }

    try:
        for turn_index, turn in enumerate(suite["turns"], start=1):
            start = time.perf_counter()
            print(f"  [turn {turn_index:02d}] {turn['question']}", flush=True)
            try:
                response = client.post(
                    "/chat",
                    json={
                        "session_id": session_id,
                        "question": turn["question"],
                        "stream": False,
                    },
                    timeout=90,
                )
                latency_ms = round((time.perf_counter() - start) * 1000, 1)
                response.raise_for_status()
                payload = response.json()
            except Exception as exc:
                latency_ms = round((time.perf_counter() - start) * 1000, 1)
                suite_results["turns"].append(
                    {
                        "turn": turn_index,
                        "question": turn["question"],
                        "expected_doc": turn["expect_doc"],
                        "expected_mode": turn["expect_mode"],
                        "actual_doc": "",
                        "actual_mode": "error",
                        "doc_match": False,
                        "mode_match": False,
                        "latency_ms": latency_ms,
                        "answer_preview": f"ERROR: {exc}",
                        "memory_candidates": [],
                        "selected_memories": [],
                        "retrieval_hint": "",
                        "assembled_messages": [],
                        "stored_memory": None,
                    }
                )
                print(f"    -> ERROR after {latency_ms} ms: {exc}", flush=True)
                continue
            debug = payload.get("debug") or {}
            sources = payload.get("sources") or []
            top_source = sources[0]["doc"] if sources else ""
            context_mode = payload.get("context_mode", "")
            memory_candidates = debug.get("memory_candidates") or []
            selected_memories = debug.get("selected_memories") or []

            suite_results["turns"].append(
                {
                    "turn": turn_index,
                    "question": turn["question"],
                    "expected_doc": turn["expect_doc"],
                    "expected_mode": turn["expect_mode"],
                    "actual_doc": top_source,
                    "actual_mode": context_mode,
                    "doc_match": top_source == turn["expect_doc"],
                    "mode_match": context_mode == turn["expect_mode"],
                    "latency_ms": latency_ms,
                    "answer_preview": _truncate(payload.get("answer", "")),
                    "memory_candidates": memory_candidates,
                    "selected_memories": selected_memories,
                    "retrieval_hint": debug.get("retrieval_hint", ""),
                    "assembled_messages": debug.get("assembled_messages", []),
                    "stored_memory": debug.get("stored_memory"),
                }
            )
            print(
                f"    -> mode={context_mode} doc={top_source} latency={latency_ms}ms",
                flush=True,
            )

        memories_resp = client.get(f"/sessions/{session_id}/memories", timeout=60)
        memories_resp.raise_for_status()
        suite_results["stored_memories"] = memories_resp.json().get("memories", [])
        return suite_results
    finally:
        client.delete(f"/sessions/{session_id}", timeout=60)


def _build_markdown_report(results: list[dict]) -> str:
    lines = [
        "# Mem0 Context Tests",
        "",
        f"Generated at: {datetime.now().isoformat()}",
        "",
    ]

    for suite in results:
        total = len(suite["turns"])
        doc_hits = sum(1 for turn in suite["turns"] if turn["doc_match"])
        mode_hits = sum(1 for turn in suite["turns"] if turn["mode_match"])
        lines.extend(
            [
                f"## {suite['name']}",
                "",
                suite["description"],
                "",
                f"- Session ID: `{suite['session_id']}`",
                f"- Source match: `{doc_hits}/{total}`",
                f"- Context-mode match: `{mode_hits}/{total}`",
                "",
                "| Turn | Question | Expected Doc | Actual Doc | Expected Mode | Actual Mode | Latency (ms) | Doc OK | Mode OK |",
                "| --- | --- | --- | --- | --- | --- | ---: | --- | --- |",
            ]
        )
        for turn in suite["turns"]:
            lines.append(
                "| {turn} | {question} | {expected_doc} | {actual_doc} | {expected_mode} | {actual_mode} | {latency_ms} | {doc_ok} | {mode_ok} |".format(
                    turn=turn["turn"],
                    question=_md_escape(turn["question"]),
                    expected_doc=_md_escape(turn["expected_doc"]),
                    actual_doc=_md_escape(turn["actual_doc"]),
                    expected_mode=_md_escape(turn["expected_mode"]),
                    actual_mode=_md_escape(turn["actual_mode"]),
                    latency_ms=turn["latency_ms"],
                    doc_ok="PASS" if turn["doc_match"] else "FAIL",
                    mode_ok="PASS" if turn["mode_match"] else "FAIL",
                )
            )
        lines.append("")

        for turn in suite["turns"]:
            lines.extend(
                [
                    f"### {suite['name']} Turn {turn['turn']}",
                    "",
                    f"- Question: `{turn['question']}`",
                    f"- Expected: `{turn['expected_doc']}` / `{turn['expected_mode']}`",
                    f"- Actual: `{turn['actual_doc']}` / `{turn['actual_mode']}`",
                    f"- Latency: `{turn['latency_ms']} ms`",
                    f"- Answer Preview: {turn['answer_preview']}",
                    "",
                    "Memory candidates:",
                ]
            )
            if turn["memory_candidates"]:
                for candidate in turn["memory_candidates"]:
                    lines.append(
                        f"- score={candidate['score']:.3f} | case={candidate.get('case_id','')} | doc={candidate.get('source_doc','')} | {_md_escape(_truncate(candidate.get('memory', ''), 160))}"
                    )
            else:
                lines.append("- none")
            lines.append("")
            lines.append("Selected memories:")
            if turn["selected_memories"]:
                for candidate in turn["selected_memories"]:
                    lines.append(
                        f"- score={candidate['score']:.3f} | case={candidate.get('case_id','')} | doc={candidate.get('source_doc','')} | {_md_escape(_truncate(candidate.get('memory', ''), 160))}"
                    )
            else:
                lines.append("- none")
            lines.append("")

        lines.extend(
            [
                "Stored episodic memories after suite:",
                "",
            ]
        )
        if suite["stored_memories"]:
            for memory in suite["stored_memories"]:
                lines.append(
                    f"- score={memory['score']:.3f} | case={memory.get('case_id','')} | doc={memory.get('source_doc','')} | {_md_escape(_truncate(memory.get('memory', ''), 180))}"
                )
        else:
            lines.append("- none")
        lines.append("")

    return "\n".join(lines)


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with httpx.Client(base_url=BASE_URL) as client:
        health = client.get("/health", timeout=30)
        health.raise_for_status()
        results = [_run_suite(client, suite) for suite in TEST_SUITES]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = ARTIFACT_DIR / f"mem0_context_tests_{timestamp}.json"
    md_path = ARTIFACT_DIR / f"mem0_context_tests_{timestamp}.md"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_build_markdown_report(results), encoding="utf-8")

    print(f"Wrote JSON report: {json_path}")
    print(f"Wrote Markdown report: {md_path}")


if __name__ == "__main__":
    main()
