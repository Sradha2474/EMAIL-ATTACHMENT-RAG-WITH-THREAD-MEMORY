import email
import html.parser
import json
import re
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import fitz  # pymupdf

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None

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
    cc         = msg.get("Cc", "")
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
            "cc"         : cc,
            "subject"    : subject,
            "text"       : body.strip(),
            "type"       : "email",
        })

    # ── attachments ──
    if msg.is_multipart():
        for part in msg.walk():
            cd    = str(part.get("Content-Disposition") or "")
            fname = part.get_filename() or ""
            if "attachment" not in cd or not fname:
                continue
            payload = part.get_payload(decode=True)
            if not payload:
                continue
            fname_lower = fname.lower()
            if fname_lower.endswith(".pdf"):
                chunks.extend(parse_pdf(payload, fname, message_id, thread_id))
            elif fname_lower.endswith(".txt"):
                chunks.extend(parse_txt(payload, fname, message_id, thread_id))
            elif fname_lower.endswith((".html", ".htm")):
                chunks.extend(parse_html(payload, fname, message_id, thread_id))
            elif fname_lower.endswith(".docx") and DocxDocument is not None:
                chunks.extend(parse_docx(payload, fname, message_id, thread_id))

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


def _attachment_chunks(
    message_id: str,
    thread_id: str,
    filename: str,
    text_pieces: list,
    chunk_type: str = "attachment",
) -> list:
    """Build chunk dicts for attachment text pieces; page_no = 1-based chunk index."""
    chunks = []
    for i, piece in enumerate(text_pieces):
        if not (piece and str(piece).strip()):
            continue
        chunks.append({
            "chunk_id"   : f"{message_id}::{filename}::{i}",
            "message_id" : message_id,
            "thread_id"  : thread_id,
            "filename"   : filename,
            "page_no"    : i + 1,
            "text"       : piece.strip(),
            "type"       : chunk_type,
        })
    return chunks


def parse_txt(data: bytes, filename: str, message_id: str, thread_id: str) -> list:
    """Extract text from .txt attachment and chunk with overlap."""
    try:
        text = data.decode("utf-8", errors="replace")
        pieces = chunk_text(text)
        return _attachment_chunks(message_id, thread_id, filename, pieces)
    except Exception as e:
        print(f"  Could not parse TXT {filename}: {e}")
        return []


class _HTMLTextExtractor(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.text_parts = []

    def handle_data(self, data):
        self.text_parts.append(data)

    def get_text(self):
        return " ".join(self.text_parts)


def parse_html(data: bytes, filename: str, message_id: str, thread_id: str) -> list:
    """Extract text from .html/.htm attachment and chunk with overlap."""
    try:
        raw = data.decode("utf-8", errors="replace")
        parser = _HTMLTextExtractor()
        parser.feed(raw)
        text = parser.get_text()
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []
        pieces = chunk_text(text)
        return _attachment_chunks(message_id, thread_id, filename, pieces)
    except Exception as e:
        print(f"  Could not parse HTML {filename}: {e}")
        return []


def parse_docx(data: bytes, filename: str, message_id: str, thread_id: str) -> list:
    """Extract text from .docx attachment and chunk with overlap."""
    if DocxDocument is None:
        print("  python-docx not installed; skip DOCX:", filename)
        return []
    try:
        import io
        doc = DocxDocument(io.BytesIO(data))
        full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        if not full_text.strip():
            return []
        pieces = chunk_text(full_text)
        return _attachment_chunks(message_id, thread_id, filename, pieces)
    except Exception as e:
        print(f"  Could not parse DOCX {filename}: {e}")
        return []


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

    if not all_chunks:
        print("\nNo email data found. Add .eml files to data/emails/ and run again.")
        return

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