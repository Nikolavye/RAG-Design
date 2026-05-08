# Context Isolation Strategy: Two-Stage Adaptive Router

## Overview

This system uses a **two-stage routing pipeline** before any retrieval or generation.
The two stages answer two independent questions in sequence:

1. **Stage 1 — Retrieval Complexity Router**: Does this query even need retrieval? If yes, how complex?
2. **Stage 2 — Memory Router**: Does this query depend on prior conversation history?

Both stages use a lightweight model (Gemini 2.5 Flash Lite) as the classifier.

---

## Stage 1: Retrieval Complexity Router

Inspired by **Adaptive-RAG** (Jeong et al., NAACL 2024).

A small model evaluates the incoming query and classifies it into one of three categories:

### Category A — Ambiguous / No Retrieval Needed

The query is vague, conversational, or can be answered directly from general knowledge
without consulting the maintenance knowledge base.

**Examples**: greetings, generic questions, underspecified requests like "tell me more"
with no clear referent.

**Action**: Skip retrieval entirely. Route directly to the LLM for a conversational
chat-style response. Stage 2 is not executed.

```
query → [Classifier] → AMBIGUOUS → LLM chat answer (no RAG)
```

### Category B — Simple / Single Retrieval

The query is a clear, self-contained question that can be answered with one retrieval pass.

**Examples**: "What causes blue smoke in an FR-210?", "Show me the hydraulic pump repair steps for case 3.4.2"

**Action**: Proceed to Stage 2 with single-retrieval execution.

### Category C — Complex / Multi-Step Retrieval + CoT

The query requires reasoning across multiple aspects, sub-questions, or retrieval passes.
A Chain-of-Thought guided approach improves answer quality.

**Examples**: "Compare the root causes of fault X and fault Y", "What's the difference between these two repair approaches?"

**Action**: Proceed to Stage 2 with iterative retrieval + CoT execution.

---

## Stage 2: Memory Router

Executed only for Category B and C queries. The same lightweight model evaluates
whether the current query is related to the prior conversation turn.

### FOLLOW_UP

The current query depends on, references, or continues the prior turn.
Indicators: pronouns ("it", "that", "them"), implicit reference to a previously mentioned
fault or machine, follow-up action requests ("how do I fix that?", "show me the image").

**Action**: Reuse the existing session. Conversation history is **included** in the
generation context. `refine_multiturn = True` allows RAGFlow to use history for
query rewriting before retrieval.

### NEW TOPIC

The current query is independent of the prior turn. It introduces a new fault,
a different machine, or a completely unrelated question.

**Action**: Create a fresh RAGFlow session. Conversation history is **excluded** from
the generation context. The LLM sees only the current query and the retrieved knowledge
base chunks.

> **RAGFlow implementation note**: calling `/completions` without a `session_id` does
> not produce a stateless query — RAGFlow creates a new session and returns the prologue
> without processing the question. NEW TOPIC routing must therefore call
> `POST /sessions` first to obtain a fresh `session_id`, then use it immediately
> for the completion call.

---

## Complete Decision Flow

```
User Query
    │
    ▼
┌──────────────────────────────────┐
│  Stage 1: Retrieval Complexity   │
│  Model: Gemini 2.5 Flash Lite    │
└──────────────────────────────────┘
    │             │                │
    ▼             ▼                ▼
AMBIGUOUS      SIMPLE (B)      COMPLEX (C)
    │           │    │           │    │
    ▼           └────┘           └────┘
LLM chat              │
(no retrieval)        ▼
                ┌──────────────────────────────────┐
                │  Stage 2: Memory Router          │
                │  Model: Gemini 2.5 Flash Lite    │
                └──────────────────────────────────┘
                      │                   │
                      ▼                   ▼
                 FOLLOW_UP           NEW TOPIC
                      │                   │
                      ▼                   ▼
              Existing session       Fresh session
              history included       history excluded
              refine_multiturn=T     refine_multiturn=F
                      │                   │
                      └─────────┬─────────┘
                                ▼
                     Retrieval + Generation
                     (single-pass or CoT)
```

---

## Why Two Separate Stages

These two routing decisions address **orthogonal dimensions** and must be kept separate:

| Dimension | Stage 1 | Stage 2 |
|---|---|---|
| Question | Does this query need retrieval? How complex? | Does this query depend on prior history? |
| Input signal | Query semantics only | Query + prior Q&A pair |
| Output | Retrieval strategy | Session / memory mode |
| Failure mode if skipped | Unnecessary retrieval, hallucination on ambiguous queries | Cross-case context contamination |

Merging them into a single classification step would conflate two independent failure modes.

---

## Pluggable Memory Principle

The core abstraction of Stage 2 is **pluggable memory**: conversation history is not
always loaded into the LLM context. It is injected only when the current query
provably depends on it.

This directly addresses the primary failure mode observed in the system:
when a user asks about a new fault after discussing a previous one, the prior session
history — which contains details about the earlier fault — was being passed to the LLM
alongside retrieved chunks for the new fault. The LLM would blend both, producing
contaminated answers.

The session boundary is the **hard isolation mechanism**. The system prompt instruction
("knowledge base takes priority over history when topics differ") is a secondary soft
guardrail, not a substitute for hard isolation.

---

## Relationship to Prior Work

| Concept | Reference | Application in This System |
|---|---|---|
| Retrieval complexity routing | Adaptive-RAG, Jeong et al., NAACL 2024 | Stage 1 three-way classification |
| Lightweight pre-call classifier | Adaptive-RAG (T5-Large) | Replaced by Gemini 2.5 Flash Lite; zero-shot, no training required |
| Follow-up vs new topic | Kundu et al., ACL 2020; TopiOCQA, TACL 2022 | Stage 2 binary classification |
| History selection over full injection | HAConvDR 2024; TACL 2023 robustness | Fresh session for NEW TOPIC instead of unbounded history |
| Adaptive multi-turn labeling | Amazon EMNLP 2025 | Explanation-based routing signal preferred in Stage 2 |
| Session-level generation isolation | ConvSelect-RAG 2025 | Hard isolation via RAGFlow fresh session in NEW TOPIC branch |

**Key distinction from Adaptive-RAG**: the original paper routes on the retrieval
complexity axis (how many retrieval steps). Stage 1 of this system replicates that axis.
Stage 2 adds an orthogonal memory axis (whether to include conversation history), which
Adaptive-RAG does not address.
