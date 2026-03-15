import json
import pickle
import re
import numpy as np
import faiss
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ── load everything once at startup ─────────────────────────────────────────
print("Loading index files...")

CHUNKS: list = json.loads(Path("index/chunks.json").read_text())

with open("index/bm25.pkl", "rb") as f:
    BM25: BM25Okapi = pickle.load(f)

FAISS_INDEX = faiss.read_index(str(Path("index/faiss.index")))
EMBED_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print(f"Index loaded. Total chunks: {len(CHUNKS)}")


# ── helpers ──────────────────────────────────────────────────────────────────
def tokenize(text: str):
    return re.findall(r"\w+", text.lower())


def min_max_norm(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-9)


# ── main retrieve function ───────────────────────────────────────────────────
def retrieve(
    query        : str,
    thread_id    : str  = None,
    top_k        : int  = 8,
    search_outside: bool = False,
) -> list:

    # BM25 scores
    bm25_raw = np.array(BM25.get_scores(tokenize(query)), dtype="float32")

    # Vector scores
    q_emb = EMBED_MODEL.encode(
        [query], normalize_embeddings=True
    ).astype("float32")
    scores, indices = FAISS_INDEX.search(q_emb, len(CHUNKS))
    vec_raw = np.zeros(len(CHUNKS), dtype="float32")
    for idx, score in zip(indices[0], scores[0]):
        if idx >= 0:
            vec_raw[idx] = float(score)

    # Normalise + fuse (60% BM25, 40% vector)
    bm25_norm = min_max_norm(bm25_raw)
    vec_norm  = min_max_norm(vec_raw)
    fused     = 0.6 * bm25_norm + 0.4 * vec_norm
    ranked    = np.argsort(fused)[::-1]

    results = []

    for idx in ranked:
        # Small floor to avoid extremely tiny scores / noise
        if fused[idx] < 0.01:
            break

        chunk  = CHUNKS[int(idx)]
        msg_id = chunk["message_id"]

        # Allow max 2 chunks per message to ensure answer variety
        msg_count = sum(
            1 for r in results
            if r["chunk"]["message_id"] == msg_id
        )
        if msg_count >= 2:
            continue

        # Thread filter
        if (
            not search_outside
            and thread_id
            and chunk["thread_id"] != thread_id
        ):
            continue

        results.append({
            "chunk"      : chunk,
            "score"      : round(float(fused[idx]), 4),
            "bm25_score" : round(float(bm25_norm[idx]), 4),
            "vec_score"  : round(float(vec_norm[idx]), 4),
        })

        if len(results) >= top_k:
            break

    return results


# ── citation formatter ───────────────────────────────────────────────────────
def format_citation(chunk: dict) -> str:
    if chunk.get("type") == "attachment" and chunk.get("page_no") is not None:
        return f"[msg: {chunk['message_id']}, page: {chunk['page_no']}]"
    return f"[msg: {chunk.get('message_id', '')}]"


def get_thread_chunks(thread_id: str) -> list:
    """All chunks for a thread (for timeline)."""
    return [c for c in CHUNKS if c.get("thread_id") == thread_id]


def get_thread_timeline(thread_id: str) -> list:
    """Timeline: who said what, when — with citations."""
    chunks = get_thread_chunks(thread_id)
    if not chunks:
        return []
    by_msg = {}
    for c in chunks:
        mid = c.get("message_id", "")
        if mid not in by_msg:
            by_msg[mid] = []
        by_msg[mid].append(c)
    ordered = sorted(by_msg.items(), key=lambda x: (x[1][0].get("date", ""), x[0]))
    timeline = []
    for msg_id, list_chunks in ordered:
        from_val = date_val = subject_val = ""
        preview = ""
        for ch in list_chunks:
            if ch.get("type") == "email":
                from_val = ch.get("from", "") or from_val
                date_val = ch.get("date", "") or date_val
                subject_val = ch.get("subject", "") or subject_val
                if ch.get("text") and not preview:
                    preview = (ch["text"][:200] + "…") if len(ch["text"]) > 200 else ch["text"]
                break
        if not preview and list_chunks:
            t = list_chunks[0].get("text", "")
            preview = (t[:200] + "…") if len(t) > 200 else t
        citation = format_citation(list_chunks[0])
        timeline.append({
            "message_id": msg_id,
            "from": from_val or list_chunks[0].get("from", ""),
            "date": date_val or list_chunks[0].get("date", ""),
            "subject": subject_val,
            "preview": (preview or "").replace("\n", " ").strip(),
            "citation": citation,
        })
    return timeline


# ── thread listing ───────────────────────────────────────────────────────────
def get_all_threads() -> dict:
    seen = {}
    for c in CHUNKS:
        tid = c["thread_id"]
        if tid not in seen:
            seen[tid] = c.get("subject", tid)
    return seen