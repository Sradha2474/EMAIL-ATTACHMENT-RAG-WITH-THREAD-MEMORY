import uuid
import json
import re
import time
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.memory import SessionMemory
from app.retrieval import retrieve, format_citation, get_all_threads


app = FastAPI()
SESSIONS: dict[str, SessionMemory] = {}


#get session or raise error 
def get_session(session_id: str) -> SessionMemory:
    if session_id not in SESSIONS:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Call /start_session first."
        )
    return SESSIONS[session_id]


# ── clean raw email text ─────────────────────────────────────
def clean_email_text(text: str) -> str:
    lines = text.splitlines()
    clean_lines = []

    skip_patterns = [
        r"^-{5,}",
        r"^_{5,}",
        r"forwarded by",
        r"original message",
        r"^from:\s",
        r"^to:\s",
        r"^cc:\s",
        r"^sent:\s",
        r"^date:\s",
        r"^subject:\s",
        r"please respond to",
        r"^on .{10,} wrote:",
        r"<[^>]+@[^>]+>",
        r"^\s*>+",
    ]

    for line in lines:
        line_lower = line.lower().strip()
        skip = False

        for pattern in skip_patterns:
            if re.search(pattern, line_lower):
                skip = True
                break

        if not skip and line.strip():
            clean_lines.append(line.strip())

    return " ".join(clean_lines).strip()


# ── detect query intent ──────────────────────────────────────
def detect_intent(query: str) -> str:

    q = query.lower().strip()

    summary_keywords = [
        "summarize",
        "summary",
        "summarise",
        "overview",
        "main points",
        "brief",
        "what happened",
        "topic",
        "purpose",
        "explain",
        "explain this",
        "what is this about",
        "what's this about",
        "what does it say",
        "what does the email say",
        "what does the attachment say",
        "describe the thread",
        "describe the email",
    ]

    if any(k in q for k in summary_keywords):
        return "summarize"

    # "Who is Lucy?" / "Who is X?" — must be before general "who"
    if re.match(r"who is\s+\w+", q):
        return "who_is"

    who_keywords = [
        "who",
        "sender",
        "from",
        "sent by",
        "author",
        "who sent",
        "who wrote",
    ]

    if any(k in q for k in who_keywords):
        return "who"

    when_keywords = [
        "when",
        "date",
        "time",
        "day",
        "month",
        "year",
        "schedule",
        "when was",
        "when did",
    ]

    if any(k in q for k in when_keywords):
        return "when"

    list_keywords = [
        "list",
        "files",
        "attachments",
        "documents",
        "items",
        "what files",
        "which documents",
        "files are being discussed",
    ]

    if any(k in q for k in list_keywords):
        return "list"

    subject_keywords = [
        "subject of",
        "subject of the thread",
        "what is the subject",
        "email subject",
    ]
    if any(k in q for k in subject_keywords):
        return "subject"

    return "general"


# ── detect clearly out‑of‑scope questions (non‑email, world knowledge) ─────
def is_out_of_scope(query: str) -> bool:
    q = query.lower()
    oos_keywords = [
        "weather", "forecast", "temperature",
        "capital of", "population of",
        "stock price", "share price",
        "who is the president", "who is the prime minister",
        "who won", "world cup",
    ]
    return any(k in q for k in oos_keywords)


# ── find relevant snippet ────────────────────────────────────
def get_relevant_snippet(text: str, query: str, max_words: int = 80):

    clean = clean_email_text(text)
    query_words = set(re.findall(r"\w+", query.lower()))

    sentences = re.split(r"(?<=[.!?\n])\s+", clean)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]

    if not sentences:
        words = clean.split()
        return " ".join(words[:max_words])

    best_score = -1
    best_idx = 0

    for i, sentence in enumerate(sentences):
        words = set(re.findall(r"\w+", sentence.lower()))
        score = len(words & query_words)

        if score > best_score:
            best_score = score
            best_idx = i

    start = max(0, best_idx - 1)
    end = min(len(sentences), best_idx + 3)

    snippet = " ".join(sentences[start:end])
    words = snippet.split()

    if len(words) > max_words:
        snippet = " ".join(words[:max_words]) + "..."

    return snippet.strip()


# ── build answer ─────────────────────────────────────────────
def build_answer(query: str, retrieved: list):

    # If we truly have nothing to ground on, or the query is clearly
    # outside the scope of this email thread, return a graceful message.
    if not retrieved or is_out_of_scope(query):
        return (
            "I don't have information about that in this email thread.\n\n"
            "This chatbot only answers questions based on the **selected email thread**.\n\n"
            "Try asking about things like:\n"
            "• Who sent or received emails in this thread\n"
            "• What topics, decisions, or files were discussed\n"
            "• Dates and events mentioned in the emails\n"
            "• The content of any attachments (e.g. 'What does the attachment say?')",
            []
        )

    intent = detect_intent(query)
    used = retrieved[:5]
    citations = []

    for r in used:
        c = r["chunk"]
        citations.append({
            "message_id": c["message_id"],
            "type": c["type"],
            "page": c.get("page_no"),
            "score": r["score"],
        })

    # ── SUMMARIZE ─────────────────────────────
    if intent == "summarize":

        points = []
        seen = set()

        for r in used:
            c = r["chunk"]
            cite = format_citation(c)
            clean = clean_email_text(c["text"])

            sentences = re.split(r"(?<=[.!?])\s+", clean)

            for s in sentences:
                s = s.strip()

                if len(s) < 20:
                    continue

                if s.lower() in seen:
                    continue

                seen.add(s.lower())
                points.append(f"• {s} {cite}")

                if len(points) >= 4:
                    break

            if len(points) >= 4:
                break

        answer = "**Summary of this thread:**\n\n" + "\n".join(points)
        return answer, citations

    # ── WHO ───────────────────────────────────
    if intent == "who":

        lines = []
        seen = set()

        for r in used:
            c = r["chunk"]
            cite = format_citation(c)

            info = []

            if c.get("from") and c["from"] not in seen:
                info.append(f"From: {c['from']}")
                seen.add(c["from"])

            if c.get("to"):
                info.append(f"To: {c['to']}")

            if info:
                lines.append(" | ".join(info) + f" {cite}")

        # If question asks "what is it about", append a short snippet
        if "about" in query.lower() or "what" in query.lower():
            for r in used[:1]:
                c = r["chunk"]
                cite = format_citation(c)
                snippet = get_relevant_snippet(c["text"], "about topic", max_words=50)
                if snippet:
                    lines.append(f"{snippet} {cite}")
                    break

        return "\n".join(lines), citations

    # ── WHEN ──────────────────────────────────
    if intent == "when":

        lines = []
        seen = set()

        for r in used:
            c = r["chunk"]
            cite = format_citation(c)

            date = c.get("date", "")
            subj = c.get("subject", "")

            if date and date not in seen:
                seen.add(date)
                lines.append(f"Date: {date} | Subject: {subj} {cite}")

        return "\n".join(lines), citations

    # ── LIST FILES ────────────────────────────
    if intent == "list":

        items = []
        seen = set()

        for r in used:
            c = r["chunk"]
            cite = format_citation(c)

            files = re.findall(
                r"\b[\w-]+\.(pdf|doc|docx|xls|xlsx|csv|ppt|pptx)\b",
                c["text"],
                re.I
            )
            for f in files:
                items.append(f"• {f} {cite}")

            # Include sentences about files / spreadsheet / attach
            clean = clean_email_text(c["text"])
            for sent in re.split(r"(?<=[.!?])\s+", clean):
                sent = sent.strip()
                if len(sent) < 15 or sent.lower() in seen:
                    continue
                if any(w in sent.lower() for w in ["file", "files", "attachment", "attach", "spreadsheet", "two file"]):
                    seen.add(sent.lower())
                    items.append(f"• {sent} {cite}")
                    if len(items) >= 8:
                        break
            if len(items) >= 8:
                break

        return "\n".join(items[:8]), citations

    # ── SUBJECT ─────────────────────────────────
    if intent == "subject":

        lines = []
        seen = set()

        for r in used:
            c = r["chunk"]
            subj = (c.get("subject") or "").strip()
            if subj and subj not in seen:
                seen.add(subj)
                cite = format_citation(c)
                lines.append(f"Subject: {subj} {cite}")

        if not lines:
            return "No subject found in this thread.", citations
        return "\n".join(lines), citations

    # ── WHO IS (person in thread) ───────────────
    if intent == "who_is":

        lines = []
        for r in used:
            c = r["chunk"]
            cite = format_citation(c)
            from_val = c.get("from") or ""
            to_val = c.get("to") or ""
            text = (c.get("text") or "").lower()

            # Prefer To: if it contains the name (recipient)
            if to_val:
                lines.append(f"To: {to_val} {cite}")
            if from_val:
                lines.append(f"From: {from_val} {cite}")

            snippet = get_relevant_snippet(c["text"], query, max_words=60)
            if snippet and snippet.strip():
                lines.append(f"{snippet} {cite}")
            if lines:
                break

        if not lines:
            return "I couldn't identify that person in this thread.", citations
        return "\n".join(lines[:3]), citations

    # ── GENERAL ───────────────────────────────
    lines = []

    for r in used:
        c = r["chunk"]
        cite = format_citation(c)

        snippet = get_relevant_snippet(c["text"], query)

        meta = []

        if c.get("from"):
            meta.append(f"From: {c['from']}")

        if c.get("date"):
            meta.append(f"Date: {c['date']}")

        if c.get("subject"):
            meta.append(f"Subject: {c['subject']}")

        meta_str = " | ".join(meta)

        block = (f"{meta_str}\n" if meta_str else "") + f"{snippet} {cite}"
        lines.append(block)

    return "\n\n---\n\n".join(lines), citations


# ── write trace log ──────────────────────────────────────────
def write_trace(session_id, user_text, rewrite, retrieved, answer, citations, latency):

    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = Path(f"runs/{ts}")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Ensure all numeric values are native Python (JSON-serializable)
    record = {
        "trace_id": str(uuid.uuid4()),
        "session_id": session_id,
        "user_text": user_text,
        "rewrite": rewrite,
        "retrieved": [
            {"id": r["chunk"]["chunk_id"], "score": float(r["score"])}
            for r in retrieved
        ],
        "used_msgs": [c["message_id"] for c in citations],
        "answer": answer,
        "citations": [
            {
                "message_id": c["message_id"],
                "type": c["type"],
                "page": c.get("page_no"),
                "score": float(c.get("score") or 0),
            }
            for c in citations
        ],
        "latency_ms": int(round(latency * 1000)),
    }

    with open(run_dir / "trace.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")

    return record["trace_id"]


# ── request models ───────────────────────────────────────────
class StartSessionRequest(BaseModel):
    thread_id: str


class AskRequest(BaseModel):
    session_id: str
    text: str
    search_outside_thread: bool = False


class SwitchThreadRequest(BaseModel):
    session_id: str
    thread_id: str


class ResetRequest(BaseModel):
    session_id: str


# ── endpoints ─────────────────────────────────────────────────
@app.post("/start_session")
def start_session(req: StartSessionRequest):

    sid = str(uuid.uuid4())
    SESSIONS[sid] = SessionMemory(session_id=sid, thread_id=req.thread_id)

    return {"session_id": sid, "thread_id": req.thread_id}


@app.post("/ask")
def ask(req: AskRequest):

    t0 = time.time()

    session = get_session(req.session_id)

    rewrite = session.rewrite_query(req.text)

    retrieved = retrieve(
        query=rewrite,
        thread_id=session.thread_id,
        search_outside=req.search_outside_thread,
    )

    answer, citations = build_answer(rewrite, retrieved)

    session.add_turn(req.text, rewrite, answer)
    session.update_entities(answer)

    trace_id = write_trace(
        req.session_id,
        req.text,
        rewrite,
        retrieved,
        answer,
        citations,
        time.time() - t0,
    )

    return {
        "answer": answer,
        "citations": citations,
        "rewrite": rewrite,
        "retrieved": [
            {
                "id": r["chunk"]["chunk_id"],
                "score": r["score"],
                "bm25_score": r["bm25_score"],
                "vec_score": r["vec_score"],
            }
            for r in retrieved[:8]
        ],
        "trace_id": trace_id,
    }


@app.post("/switch_thread")
def switch_thread(req: SwitchThreadRequest):

    session = get_session(req.session_id)
    session.thread_id = req.thread_id
    session.turns.clear()

    return {"ok": True, "thread_id": req.thread_id}


@app.post("/reset_session")
def reset_session(req: ResetRequest):

    session = get_session(req.session_id)

    SESSIONS[req.session_id] = SessionMemory(
        session_id=req.session_id,
        thread_id=session.thread_id,
    )

    return {"ok": True}


@app.get("/threads")
def threads():
    return get_all_threads()


@app.get("/health")
def health():
    return {"status": "ok"}
