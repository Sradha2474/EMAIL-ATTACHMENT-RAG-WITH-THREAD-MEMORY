import email
import os
import json
import re
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import fitz  # pymupdf

# ── paths ──────────────────────────────────────────────────────────────────
EMAILS_DIR  = Path("data/emails")
INDEX_DIR   = Path("index")
BM25_FILE   = INDEX_DIR / "bm25.pkl"
FAISS_FILE  = INDEX_DIR / "faiss.index"
CHUNKS_FILE = INDEX_DIR / "chunks.json"

INDEX_DIR.mkdir(exist_ok=True)

# ── helpers ─────────────────────────────────────────────────────────────────
def get_thread_id(msg):
    refs = msg.get("References") or msg.get("In-Reply-To") or ""
    if refs.strip():
        return refs.strip().split()[0]
    return (msg.get("Message-ID") or "no-thread").strip()

def tokenize(text):
    return re.findall(r"\w+", text.lower())

def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    stride = chunk_size - overlap
    for i in range(0, max(1, len(words)), stride):
        chunk = " ".join(words[i : i + chunk_size])
        if len(chunk.strip()) > 30:
            chunks.append(chunk)
    return chunks

# ── parse one .eml file ─────────────────────────────────────────────────────
def parse_email_file(path: Path):
    raw = path.read_text(errors="replace")
    msg = email.message_from_string(raw)

    message_id = (msg.get("Message-ID") or path.stem).strip()
    thread_id  = get_thread_id(msg)
    date       = msg.get("Date", "")
    sender     = msg.get("From", "")
    to         = msg.get("To", "")
    subject    = msg.get("Subject", "")

    chunks = []

    # ── body ──
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            ct  = part.get_content_type()
            cd  = str(part.get("Content-Disposition") or "")
            if ct == "text/plain" and "attachment" not in cd:
                payload = part.get_payload(decode=True)
                if payload:
                    body += payload.decode(errors="replace")
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            body = payload.decode(errors="replace")

    if body.strip():
        chunks.append({
            "chunk_id"   : f"{message_id}::body",
            "message_id" : message_id,
            "thread_id"  : thread_id,
            "date"       : date,
            "from"       : sender,
            "to"         : to,
            "subject"    : subject,
            "text"       : body.strip(),
            "type"       : "email",
        })

    # ── attachments ──
    if msg.is_multipart():
        for part in msg.walk():
            cd    = str(part.get("Content-Disposition") or "")
            fname = part.get_filename() or ""
            if "attachment" in cd and fname.lower().endswith(".pdf"):
                payload = part.get_payload(decode=True)
                if payload:
                    pdf_chunks = parse_pdf(payload, fname, message_id, thread_id)
                    chunks.extend(pdf_chunks)

    return chunks

# ── parse a PDF attachment ───────────────────────────────────────────────────
def parse_pdf(data: bytes, filename: str, message_id: str, thread_id: str):
    chunks = []
    try:
        doc = fitz.open(stream=data, filetype="pdf")
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            for i, chunk_text_piece in enumerate(chunk_text(text)):
                chunks.append({
                    "chunk_id"   : f"{message_id}::{filename}::p{page_num}::{i}",
                    "message_id" : message_id,
                    "thread_id"  : thread_id,
                    "filename"   : filename,
                    "page_no"    : page_num,
                    "text"       : chunk_text_piece,
                    "type"       : "attachment",
                })
    except Exception as e:
        print(f"  Could not parse PDF {filename}: {e}")
    return chunks

# ── main build function ──────────────────────────────────────────────────────
def build_index():
    print("Step 1/4 — Parsing emails...")
    all_chunks = []
    eml_files  = list(EMAILS_DIR.glob("*.eml"))
    for i, path in enumerate(eml_files):
        chunks = parse_email_file(path)
        all_chunks.extend(chunks)
        if (i + 1) % 20 == 0:
            print(f"  Parsed {i+1}/{len(eml_files)} files...")

    print(f"  Total chunks: {len(all_chunks)}")
    thread_ids = set(c["thread_id"] for c in all_chunks)
    print(f"  Total threads: {len(thread_ids)}")

    # save chunks
    with open(CHUNKS_FILE, "w") as f:
        json.dump(all_chunks, f)

    # ── BM25 ──
    print("Step 2/4 — Building BM25 index...")
    corpus = [tokenize(c["text"]) for c in all_chunks]
    bm25   = BM25Okapi(corpus)
    with open(BM25_FILE, "wb") as f:
        pickle.dump(bm25, f)
    print("  BM25 done.")

    # ── Vectors ──
    print("Step 3/4 — Encoding with sentence-transformers (takes a few minutes)...")
    model      = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    texts      = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    embeddings = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)

    print("Step 4/4 — Building FAISS index...")
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(FAISS_FILE))
    print("  FAISS done.")

    print("\nAll done! Index is ready.")

if __name__ == "__main__":
    build_index()