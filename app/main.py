import uuid
import json
import time
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.memory import SessionMemory
from app.retrieval import retrieve, format_citation, get_all_threads

app = FastAPI()

# stores all active sessions in memory
SESSIONS: dict[str, SessionMemory] = {}

# ── helper: get session or raise 404 ────────────────────────────────────────
def get_session(session_id: str) -> SessionMemory:
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found. Call /start_session first.")
    return SESSIONS[session_id]

# ── helper: build a grounded answer from retrieved chunks ────────────────────
def build_answer(retrieved: list) -> tuple[str, list]:
    if not retrieved:
        return (
            "I could not find relevant information in this thread. "
            "Try rephrasing your question or enabling 'search outside thread'.",
            []
        )
    # Use top 3 chunks
    used      = retrieved[:3]
    sentences = []
    citations = []

    for r in used:
        c       = r["chunk"]
        snippet = c["text"][:350].replace("\n", " ").strip()
        cite    = format_citation(c)
        sentences.append(f"{snippet} {cite}")
        citations.append({
            "message_id" : c["message_id"],
            "type"       : c["type"],
            "page"       : c.get("page_no"),
            "score"      : r["score"],
        })

    answer = "\n\n".join(sentences)
    return answer, citations

# ── helper: write a trace record ─────────────────────────────────────────────
def write_trace(session_id, user_text, rewrite,
                retrieved, answer, citations, latency):
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

# ── request/response models ──────────────────────────────────────────────────
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
    sid            = str(uuid.uuid4())
    SESSIONS[sid]  = SessionMemory(session_id=sid, thread_id=req.thread_id)
    return {"session_id": sid, "thread_id": req.thread_id}

@app.post("/ask")
def ask(req: AskRequest):
    t0      = time.time()
    session = get_session(req.session_id)

    # rewrite query using conversation memory
    rewrite   = session.rewrite_query(req.text)

    # retrieve relevant chunks
    retrieved = retrieve(
        query          = rewrite,
        thread_id      = session.thread_id,
        search_outside = req.search_outside_thread,
    )

    # build grounded answer
    answer, citations = build_answer(retrieved)

    # update memory
    session.add_turn(req.text, rewrite, answer)
    session.update_entities(answer)

    # log trace
    trace_id = write_trace(
        req.session_id, req.text, rewrite,
        retrieved, answer, citations,
        time.time() - t0
    )

    return {
        "answer"     : answer,
        "citations"  : citations,
        "rewrite"    : rewrite,
        "retrieved"  : [
            {
                "id"         : r["chunk"]["chunk_id"],
                "score"      : r["score"],
                "bm25_score" : r["bm25_score"],
                "vec_score"  : r["vec_score"],
            }
            for r in retrieved[:8]
        ],
        "trace_id"   : trace_id,
    }

@app.post("/switch_thread")
def switch_thread(req: SwitchThreadRequest):
    session           = get_session(req.session_id)
    session.thread_id = req.thread_id
    session.turns.clear()
    return {"ok": True, "thread_id": req.thread_id}

@app.post("/reset_session")
def reset_session(req: ResetRequest):
    session       = get_session(req.session_id)
    SESSIONS[req.session_id] = SessionMemory(
        session_id=req.session_id,
        thread_id=session.thread_id
    )
    return {"ok": True}

@app.get("/threads")
def threads():
    return get_all_threads()

@app.get("/health")
def health():
    return {"status": "ok"}