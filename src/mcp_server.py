#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MCP Server — Excavator Maintenance Assistant

Exposes the RAG knowledge base as MCP tools so Claude Desktop (or any
MCP-compatible client) can query maintenance procedures directly.

Requires: pip install mcp httpx python-dotenv

Usage (stdio transport, for Claude Desktop):
    python mcp_server.py

Usage (SSE transport, for remote clients):
    python mcp_server.py --transport sse --port 8765
"""

import os
import sys
import argparse
import httpx
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

CHAT_SERVER_URL = os.getenv('CHAT_SERVER_URL', 'http://localhost:7860')

mcp = FastMCP(
    "Excavator Maintenance Assistant",
    instructions=(
        "You are connected to a multimodal RAG knowledge base for excavator maintenance. "
        "The knowledge base contains detailed repair procedures, fault diagnosis guides, "
        "and technical diagrams extracted from official maintenance manuals. "
        "Use query_maintenance for quick one-off questions. "
        "Use create_diagnostic_session + chat_in_session for multi-turn diagnosis workflows "
        "where follow-up questions depend on earlier context."
    ),
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _format_answer(data: dict) -> str:
    """Format a /chat response dict into human-readable text."""
    answer = data.get("answer", "").strip()
    if not answer:
        return "No answer returned from the knowledge base."
    sources = [s for s in data.get("sources", []) if s.get("doc")]
    if sources:
        refs = "\n\n**Sources:**\n" + "\n".join(
            f"- [{s['doc']}] {s['chunk'][:120]}…" for s in sources
        )
        return answer + refs
    return answer


# ── Tools ──────────────────────────────────────────────────────────────────────

@mcp.tool()
async def query_maintenance(question: str) -> str:
    """Query the excavator maintenance knowledge base with a single question.

    Creates a temporary session internally, retrieves the answer with relevant
    text and image references, then cleans up the session. Use this for
    standalone questions that don't need prior conversation context.

    Args:
        question: Maintenance question or fault description (e.g.
                  "hydraulic oil leak from the swing motor", "track tension adjustment").
    """
    async with httpx.AsyncClient(timeout=90) as client:
        # Create a temporary session
        sess = await client.post(
            f"{CHAT_SERVER_URL}/sessions",
            json={"name": "MCP one-shot"},
        )
        sess.raise_for_status()
        session_id = sess.json()["session_id"]

        try:
            resp = await client.post(
                f"{CHAT_SERVER_URL}/chat",
                json={"session_id": session_id, "question": question, "stream": False},
            )
            resp.raise_for_status()
            return _format_answer(resp.json())
        finally:
            # Best-effort cleanup
            await client.delete(f"{CHAT_SERVER_URL}/sessions/{session_id}")


@mcp.tool()
async def create_diagnostic_session(name: str = "Diagnostic Session") -> str:
    """Create a new multi-turn diagnostic session.

    Use this when you expect to ask several follow-up questions about the same
    fault. The session retains full conversation history across turns.

    Args:
        name: Human-readable label for the session (optional).

    Returns:
        A JSON-like summary with session_id and the assistant's opening greeting.
    """
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{CHAT_SERVER_URL}/sessions",
            json={"name": name},
        )
        resp.raise_for_status()
        data = resp.json()

    session_id = data["session_id"]
    greeting = data.get("greeting", "")
    return (
        f"session_id: {session_id}\n"
        f"name: {data.get('name', name)}\n"
        f"greeting: {greeting}"
    )


@mcp.tool()
async def chat_in_session(session_id: str, question: str) -> str:
    """Ask a follow-up question inside an existing diagnostic session.

    The assistant remembers all previous turns within the session, so you can
    ask questions like "what torque spec did you mention?" without repeating context.

    Args:
        session_id: ID returned by create_diagnostic_session.
        question:   Follow-up question or additional fault detail.
    """
    async with httpx.AsyncClient(timeout=90) as client:
        resp = await client.post(
            f"{CHAT_SERVER_URL}/chat",
            json={"session_id": session_id, "question": question, "stream": False},
        )
        resp.raise_for_status()
        return _format_answer(resp.json())


@mcp.tool()
async def get_session_history(session_id: str) -> str:
    """Retrieve the full conversation history for a diagnostic session.

    Args:
        session_id: ID of the session to inspect.

    Returns:
        Formatted transcript of all USER / ASSISTANT turns.
    """
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(f"{CHAT_SERVER_URL}/sessions/{session_id}")
        resp.raise_for_status()
        data = resp.json()

    messages = data.get("messages", [])
    if not messages:
        return "Session exists but has no messages yet."

    lines = []
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        # Truncate very long assistant answers for readability
        if role == "ASSISTANT" and len(content) > 500:
            content = content[:500] + "…"
        lines.append(f"[{role}] {content}")
    return "\n\n".join(lines)


@mcp.tool()
async def delete_diagnostic_session(session_id: str) -> str:
    """Delete a diagnostic session and free its history.

    Args:
        session_id: ID of the session to delete.
    """
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.delete(f"{CHAT_SERVER_URL}/sessions/{session_id}")
        resp.raise_for_status()
    return f"Session {session_id} deleted."


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Excavator Maintenance MCP Server")
    parser.add_argument(
        "--transport", choices=["stdio", "sse"], default="stdio",
        help="Transport: 'stdio' for Claude Desktop, 'sse' for remote clients (default: stdio)",
    )
    parser.add_argument("--port", type=int, default=8765, help="Port for SSE transport (default: 8765)")
    args = parser.parse_args()

    if args.transport == "sse":
        mcp.run(transport="sse", port=args.port)
    else:
        mcp.run(transport="stdio")
