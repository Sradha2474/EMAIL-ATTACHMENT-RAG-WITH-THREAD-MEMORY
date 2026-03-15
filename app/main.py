import uuid
import json
import re
import time
import os
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.memory import SessionMemory
from app.retrieval import retrieve, format_citation, get_all_threads


app = FastAPI()
SESSIONS: dict[str, SessionMemory] = {}


# ── get session or raise error ───────────────────────────────────────────────
def get_session(session_id: str) -> SessionMemory:
    if session_id not in SESSIONS:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Call /start_session first."
        )
    return SESSIONS[session_id]


# ── clean raw email text ─────────────────────────────────────────────────────
def clean_email_text(text: str) -> str:
    """
    Remove forwarding headers, reply separators, and noise
    from raw email body text.
    """
    lines        = text.splitlines()
    clean_lines  = []
    skip_patterns = [
        r"^-{5,}",                          # ------Forwarded by------
        r"^_{5,}",                          # ______
        r"forwarded by",                    # Forwarded by Phillip...
        r"original message",               # ----Original Message----
        r"^from:\s",                        # From: someone
        r"^to:\s",                          # To: someone
        r"^cc:\s",                          # Cc: someone
        r"^sent:\s",                        # Sent: date
        r"^date:\s",                        # Date: ...
        r"^subject:\s",                     # Subject: ...
        r"please respond to",              # Please respond to
        r"^on .{10,} wrote:",              # On Mon, Jan 1 ... wrote:
        r"<[^>]+@[^>]+>",                  # email addresses in angle brackets
        r"^\s*>+",                          # > quoted lines
    ]

    for line in lines:
        line_lower = line.lower().strip()
        skip       = False
        for pattern in skip_patterns:
            if re.search(pattern, line_lower):
                skip = True
                break
        if not skip and line.strip():
            clean_lines.append(line.strip())

    return " ".join(clean_lines).strip()


# ── detect query intent ──────────────────────────────────────────────────────
def detect_intent(query: str) -> str:
    """
    Detect what kind of answer the user wants.
    Returns: 'summarize', 'who', 'what', 'when', 'list', 'general'
    """
    q = query.lower().strip()

    if any(w in q for w in ["summarize", "summary", "main points",
                              "overview", "what happened", "brief"]):
        return "summarize"

    if any(w in q for w in ["who", "sender", "from", "sent by",
                              "author", "person"]):
        return "who"

    if any(w in q for w in ["when", "date", "time", "day", "month",
                              "year", "schedule"]):
        return "when"

    if any(w in q for w in ["list", "files", "attachments",
                              "documents", "items"]):
        return "list"

    return "general"


# ── find the most relevant snippet for a query ───────────────────────────────
def get_relevant_snippet(text: str, query: str, max_words: int = 80) -> str:
    """
    Find the sentence in the chunk most relevant to the query
    and return a short window around it.
    """
    clean       = clean_email_text(text)
    query_words = set(re.findall(r"\w+", query.lower()))
    sentences   = re.split(r"(?<=[.!?\n])\s+", clean)
    sentences   = [s.strip() for s in sentences if len(s.strip()) > 15]

    if not sentences:
        words = clean.split()
        return " ".join(words[:max_words]) + ("..." if len(words) > max_words else "")

    best_score = -1
    best_idx   = 0
    for i, sentence in enumerate(sentences):
        words = set(re.findall(r"\w+", sentence.lower()))
        score = len(words & query_words)
        if score > best_score:
            best_score = score
            best_idx   = i

    start   = max(0, best_idx - 1)
    end     = min(len(sentences), best_idx + 3)
    snippet = " ".join(sentences[start:end])

    words = snippet.split()
    if len(words) > max_words:
        snippet = " ".join(words[:max_words]) + "..."

    return snippet.strip()


# ── format answer based on intent ────────────────────────────────────────────
def build_answer(query: str, retrieved: list) -> tuple[str, list]:
    if not retrieved:
        return (
            "I could not find relevant information in this thread. "
            "Try rephrasing your question or enabling 'search outside thread'.",
            []
        )

    intent    = detect_intent(query)
    used      = retrieved[:5]
    citations = []

    for r in used:
        c = r["chunk"]
        citations.append({
            "message_id" : c["message_id"],
            "type"       : c["type"],
            "page"       : c.get("page_no"),
            "score"      : r["score"],
        })

    # ── SUMMARIZE intent ────────────────────────────────────────────────────
    if intent == "summarize":
        points = []
        seen   = set()

        for r in used:
            c     = r["chunk"]
            clean = clean_email_text(c["text"])
            cite  = format_citation(c)

            # Extract meaningful sentences
            sentences = re.split(r"(?<=[.!?])\s+", clean)
            for sentence in sentences:
                s = sentence.strip()
                # Skip short, duplicate, or noise sentences
                if (
                    len(s) < 20
                    or s.lower() in seen
                    or re.match(r"^\d+$", s)
                ):
                    continue
                seen.add(s.lower())
                points.append(f"• {s} {cite}")
                if len(points) >= 6:
                    break
            if len(points) >= 6:
                break

        if not points:
            return "Could not extract a clear summary from this thread.", citations

        answer = "**Summary of this thread:**\n\n" + "\n".join(points)
        return answer, citations

    # ── WHO intent ──────────────────────────────────────────────────────────
    if intent == "who":
        lines = []
        seen  = set()
        for r in used:
            c    = r["chunk"]
            cite = format_citation(c)
            info = []
            if c.get("from") and c["from"] not in seen:
                info.append(f"From: {c['from']}")
                seen.add(c["from"])
            if c.get("to"):
                info.append(f"To: {c['to']}")
            if info:
                lines.append(" | ".join(info) + f"  {cite}")
        if not lines:
            lines.append("Could not identify senders/recipients from this thread.")
        return "\n".join(lines), citations

    # ── WHEN intent ─────────────────────────────────────────────────────────
    if intent == "when":
        lines = []
        seen  = set()
        for r in used:
            c    = r["chunk"]
            cite = format_citation(c)
            date = c.get("date", "")
            subj = c.get("subject", "")
            if date and date not in seen:
                seen.add(date)
                lines.append(f"Date: {date} | Subject: {subj}  {cite}")
        if not lines:
            lines.append("Could not find specific dates in this thread.")
        return "\n".join(lines), citations

    # ── LIST intent ─────────────────────────────────────────────────────────
    if intent == "list":
        items = []
        for r in used:
            c    = r["chunk"]
            cite = format_citation(c)
            # Look for filenames
            files = re.findall(
                r"\b[\w-]+\.(?:pdf|doc|docx|txt|xls|xlsx|csv|ppt|pptx)\b",
                c["text"], re.I
            )
            for f in files:
                items.append(f"• {f}  {cite}")
            # Also pull sentences with list keywords
            clean = clean_email_text(c["text"])
            for sentence in re.split(r"(?<=[.!?])\s+", clean):
                if any(w in sentence.lower() for w in
                       ["attach", "file", "document", "spreadsheet",
                        "report", "enclosed"]):
                    items.append(f"• {sentence.strip()}  {cite}")
        if not items:
            return "No files or list items found in this thread.", citations
        # Deduplicate
        seen  = set()
        dedup = []
        for item in items:
            if item not in seen:
                seen.add(item)
                dedup.append(item)
        return "\n".join(dedup[:8]), citations

    # ── GENERAL intent ──────────────────────────────────────────────────────
    lines = []
    for r in used:
        c       = r["chunk"]
        cite    = format_citation(c)
        snippet = get_relevant_snippet(c["text"], query)
        meta    = []
        if c.get("from"):
            meta.append(f"From: {c['from']}")
        if c.get("date"):
            meta.append(f"Date: {c['date']}")
        if c.get("subject"):
            meta.append(f"Subject: {c['subject']}")
        meta_str = " | ".join(meta)
        block    = (f"{meta_str}\n" if meta_str else "") + f"{snippet}  {cite}"
        lines.append(block)

    return "\n\n---\n\n".join(lines), citations


# ── write trace log ──────────────────────────────────────────────────────────
def write_trace(
    session_id, user_text, rewrite,
    retrieved, answer, citations, latency
):
    ts      = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = Path(f"runs/{ts}")
    run_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "trace_id"   : str(uuid.uuid4()),
        "session_id" : session_id,
        "user_text"  : user_text,
        "rewrite"    : rewrite,
        "retrieved"  : [
            {"id": r["chunk"]["chunk_id"], "score": r["score"]}
            for r in retrieved
        ],
        "used_msgs"  : [c["message_id"] for c in citations],
        "answer"     : answer,
        "citations"  : citations,
        "latency_ms" : round(latency * 1000),
    }

    with open(run_dir / "trace.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")

    return record["trace_id"]


# ── request models ───────────────────────────────────────────────────────────
class StartSessionRequest(BaseModel):
    thread_id: str

class AskRequest(BaseModel):
    session_id            : str
    text                  : str
    search_outside_thread : bool = False

class SwitchThreadRequest(BaseModel):
    session_id : str
    thread_id  : str

class ResetRequest(BaseModel):
    session_id: str


# ── endpoints ────────────────────────────────────────────────────────────────
@app.post("/start_session")
def start_session(req: StartSessionRequest):
    sid           = str(uuid.uuid4())
    SESSIONS[sid] = SessionMemory(session_id=sid, thread_id=req.thread_id)
    return {"session_id": sid, "thread_id": req.thread_id}


@app.post("/ask")
def ask(req: AskRequest):
    t0      = time.time()
    session = get_session(req.session_id)

    rewrite = session.rewrite_query(req.text)

    retrieved = retrieve(
        query          = rewrite,
        thread_id      = session.thread_id,
        search_outside = req.search_outside_thread,
    )

    answer, citations = build_answer(rewrite, retrieved)

    session.add_turn(req.text, rewrite, answer)
    session.update_entities(answer)

    trace_id = write_trace(
        req.session_id, req.text, rewrite,
        retrieved, answer, citations,
        time.time() - t0,
    )

    return {
        "answer"    : answer,
        "citations" : citations,
        "rewrite"   : rewrite,
        "retrieved" : [
            {
                "id"         : r["chunk"]["chunk_id"],
                "score"      : r["score"],
                "bm25_score" : r["bm25_score"],
                "vec_score"  : r["vec_score"],
            }
            for r in retrieved[:8]
        ],
        "trace_id"  : trace_id,
    }


@app.post("/switch_thread")
def switch_thread(req: SwitchThreadRequest):
    session           = get_session(req.session_id)
    session.thread_id = req.thread_id
    session.turns.clear()
    return {"ok": True, "thread_id": req.thread_id}


@app.post("/reset_session")
def reset_session(req: ResetRequest):
    session = get_session(req.session_id)
    SESSIONS[req.session_id] = SessionMemory(
        session_id=req.session_id,
        thread_id =session.thread_id,
    )
    return {"ok": True}


@app.get("/threads")
def threads():
    return get_all_threads()


@app.get("/health")
def health():
    return {"status": "ok"}