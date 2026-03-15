#searches the index
import json
import pickle
import re
import numpy as np
import faiss
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ── load everything once at startup ─────────────────────────────────────────
print("Loading index...")
CHUNKS: list = json.loads(Path("index/chunks.json").read_text())

with open("index/bm25.pkl", "rb") as f:
    BM25: BM25Okapi = pickle.load(f)

FAISS_INDEX = faiss.read_index(str(Path("index/faiss.index")))
EMBED_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Index loaded.")

# ── helpers ──────────────────────────────────────────────────────────────────
def tokenize(text: str):
    return re.findall(r"\w+", text.lower())

def min_max_norm(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-9)

# ── main retrieve function ───────────────────────────────────────────────────
def retrieve(query: str, thread_id: str = None,
             top_k: int = 8, search_outside: bool = False) -> list:

    # BM25 scores
    bm25_raw = np.array(BM25.get_scores(tokenize(query)), dtype="float32")

    # Vector scores
    q_emb = EMBED_MODEL.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = FAISS_INDEX.search(q_emb, len(CHUNKS))
    vec_raw = np.zeros(len(CHUNKS), dtype="float32")
    for idx, score in zip(indices[0], scores[0]):
        if idx >= 0:
            vec_raw[idx] = float(score)

    # Normalise then fuse (60% BM25, 40% vector)
    bm25_norm = min_max_norm(bm25_raw)
    vec_norm  = min_max_norm(vec_raw)
    fused     = 0.6 * bm25_norm + 0.4 * vec_norm

    ranked = np.argsort(fused)[::-1]

    results = []
    for idx in ranked:
        if fused[idx] < 0.01:
            break
        chunk = CHUNKS[int(idx)]
        if not search_outside and thread_id and chunk["thread_id"] != thread_id:
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

def format_citation(chunk: dict) -> str:
    if chunk["type"] == "attachment":
        return f"[msg: {chunk['message_id']}, page: {chunk['page_no']}]"
    return f"[msg: {chunk['message_id']}]"

def get_all_threads() -> dict:
    seen = {}
    for c in CHUNKS:
        tid = c["thread_id"]
        if tid not in seen:
            seen[tid] = c.get("subject", tid)
    return seen