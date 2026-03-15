# Email + Attachment RAG with Thread Memory

A retrieval-augmented generation (RAG) chatbot that answers questions
about email threads and their attachments. Every factual statement is
grounded with inline citations pointing to the exact email message ID
and attachment page number.

Built using only free and open-source tools. No paid API key required
to run the system end-to-end.

---

## Demo Video

> [Click here to watch the demo](ADD_YOUR_VIDEO_LINK_HERE)
>
> The demo covers:
> - Thread selection and locking to `Re: checkbook and budget`
> - Asking who sent the email — grounded answer with citation
> - Pronoun handling — "what did they say about it?" triggers rewrite
> - Correction handling — changing question mid-conversation
> - Graceful failure — "What is the weather in New York?" returns no hallucination
> - Trace log walkthrough in VS Code showing user_text, rewrite, scores, latency

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Running the Project](#running-the-project)
- [Running with Docker](#running-with-docker)
- [Dataset](#dataset)
- [Retrieval Approach](#retrieval-approach)
- [Design Choices](#design-choices)
- [Known Limitations](#known-limitations)
- [API Endpoints](#api-endpoints)
- [Sample Questions and Expected Citations](#sample-questions-and-expected-citations)
- [Sample Trace Log](#sample-trace-log)
- [How to Test](#how-to-test)
- [Evaluation Criteria Coverage](#evaluation-criteria-coverage)

---

## Overview

This system lets you:

1. Select an email thread from the Enron dataset
2. Ask natural language questions about that thread and its attachments
3. Receive answers grounded with inline citations:
   - Email: `[msg: <message_id>]`
   - Attachment: `[msg: <message_id>, page: N]`
4. Carry a multi-turn conversation with pronoun and ellipsis resolution
5. See the full debug trace — query rewrite, chunk scores, sources used

---

## Features

### Core
- Parses `.eml` files and extracts clean plain-text email bodies
- Extracts and chunks PDF attachments with page-level citation tracking
- Hybrid retrieval: BM25 (keyword) + FAISS (dense vector) with late fusion
- Minimum relevance score threshold — graceful failure instead of
  hallucination when no relevant content is found
- Thread locking — retrieval scoped to selected thread by default
- Global search fallback via `search_outside_thread` toggle

### Conversation Memory
- Rolling window of last 5 turns stored per session
- Pronoun resolution: "it", "that", "this", "they" resolved using
  previous answer as context before retrieval
- Short follow-up handling: queries under 5 words get the previous
  question prepended
- Entity memory: dates, dollar amounts, filenames extracted and
  remembered across turns

### Answer Quality
- Intent detection: summarize / who / when / list / general
- Each intent type produces a differently formatted answer
- Raw email forwarding headers stripped before display
- Every answer block ends with an inline citation

### Transparency
- Debug panel shows: query rewrite + top-k chunk IDs with
  fused / BM25 / vector scores + citations used
- Full trace log written to `runs/<timestamp>/trace.jsonl` per turn
- Trace includes: user_text, rewrite, retrieved ids+scores,
  used_msgs, answer, citations, latency_ms

---

## Project Structure
```
EMAIL-ATTACHMENT-RAG-WITH-THREAD-MEMORY/
│
├── data/
│   ├── emails/               # parsed .eml files (gitignored)
│   └── attachments/          # reserved for future use
│
├── index/
│   ├── chunks.json           # all parsed chunks with metadata
│   ├── bm25.pkl              # serialised BM25Okapi index
│   └── faiss.index           # FAISS IndexFlatIP vector index
│
├── runs/                     # trace logs — auto-created on first question
│   └── <timestamp>/
│       └── trace.jsonl       # one JSON record per turn
│
├── app/
│   ├── __init__.py
│   ├── main.py               # FastAPI server + answer builder
│   ├── retrieval.py          # hybrid BM25 + vector retrieval
│   ├── memory.py             # session memory + query rewriting
│   └── ui.py                 # Streamlit chat interface
│
├── ingest.py                 # parse emails + build BM25 + FAISS index
├── slice_data.py             # create dataset slice from emails.csv
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── README.md
└── DATASET.md
```

---

## Setup and Installation

### Prerequisites

- Python 3.11
- Git
- Docker Desktop (only needed for Docker path)
- Kaggle account (to download dataset)

### Step 1 — Clone the repository
```bash
git clone https://github.com/Sradha2474/EMAIL-ATTACHMENT-RAG-WITH-THREAD-MEMORY.git
cd EMAIL-ATTACHMENT-RAG-WITH-THREAD-MEMORY
```

### Step 2 — Create and activate virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
python -m pip install --upgrade pip
pip install pymupdf==1.24.3 --only-binary=:all:
pip install -r requirements.txt
```

### Step 4 — Download the dataset

1. Go to https://www.kaggle.com/datasets/wcukierski/enron-email-dataset
2. Download and extract the zip file
3. Place `emails.csv` in the project root folder

### Step 5 — Create the dataset slice
```bash
python slice_data.py
```

Expected output:
```
Saved 180-200 emails to data/emails/
```

### Step 6 — Build the search index
```bash
python ingest.py
```

Expected output:
```
Step 1/4 — Parsing emails...
  Total chunks: ~400-600
  Total threads: ~15-25
Step 2/4 — Building BM25 index...
  BM25 done.
Step 3/4 — Encoding with sentence-transformers...
Step 4/4 — Building FAISS index...
  FAISS done.
All done! Index is ready.
```

> First run downloads the sentence-transformers model (~90 MB)
> and takes 5–10 minutes. Subsequent runs are faster.

---

## Running the Project

Open **two terminals**. Activate `venv` in both.

**Terminal 1 — API server:**
```bash
uvicorn app.main:app --reload --port 8000
```

**Terminal 2 — UI:**
```bash
streamlit run app/ui.py
```

Open **http://localhost:8501** in your browser.

### Usage

1. Pick a thread from the left sidebar dropdown
2. Click **Start / Switch to this thread**
3. Type a question in the chat input
4. See the grounded answer with citations in the chat area
5. Check the debug panel on the right for retrieval details

---

## Running with Docker

Make sure Docker Desktop is running:
```bash
docker compose up --build
```

| Service | URL |
|---------|-----|
| API | http://localhost:8000 |
| UI  | http://localhost:8501 |

To stop:
```bash
docker compose down
```

---

## Dataset

See [DATASET.md](DATASET.md) for full details.

| Property | Value |
|----------|-------|
| Source | Enron Email Dataset (Kaggle) |
| Link | https://www.kaggle.com/datasets/wcukierski/enron-email-dataset |
| Mailbox | allen-p (Philip Allen, Enron trader) |
| Date range | 2000–2001 |
| Emails | ~180–200 messages |
| Threads | ~15–25 threads |
| Attachments | PDF files found in the emails |
| License | Public domain (released by FERC) |

---

## Retrieval Approach

### Baseline — BM25 (required, free path)

- Library: `rank_bm25` (BM25Okapi)
- Tokenizer: `re.findall(r"\w+", text.lower())`
- Scores every chunk against the query using TF-IDF weighting
- No GPU, no model download needed
- Latency: p95 ≤ 2.5s at top-k=8 on warm cache

### Bonus — Hybrid BM25 + Dense Vector Search

- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (90 MB, CPU)
- Vector store: FAISS `IndexFlatIP` with L2 normalisation
- Both scores min-max normalised before fusion

**Late fusion formula:**
```
fused_score = 0.6 × BM25_normalised + 0.4 × vector_normalised
```

BM25 weighted higher (0.6) because keyword precision matters more
for short email queries. Vector search (0.4) handles paraphrases
and semantic variation.

**Relevance threshold:**
```
MIN_SCORE = 0.35
```

If the top chunk scores below 0.35 the system returns empty results
and produces a graceful failure message — no hallucination.

**Deduplication:**
Maximum 2 chunks per `message_id` in top-k to ensure answers draw
from multiple emails rather than one dominant message.

**Latency:** p95 ≤ 3.0s at top-k=8 on warm cache (CPU only)

---

## Design Choices

### Chunking

| Content | Strategy | Reason |
|---------|----------|--------|
| Email body | One chunk per message | Emails are short; splitting loses context |
| PDF attachment | 300-word chunks, 50-word overlap | Preserves page boundaries for citations |

### Thread locking

`start_session(thread_id)` scopes all retrieval to that thread.
Switching threads clears conversation history to prevent cross-thread
confusion. The "Search outside thread" toggle bypasses the filter.

### Conversation memory

`SessionMemory` stores last 5 turns + extracted entities.

Rule-based query rewriting (no model needed):

| Condition | Action |
|-----------|--------|
| Query starts with pronoun (it/that/this/they/them/those/he/she) | Prepend last answer as context |
| Query is under 5 words | Prepend previous question |
| Otherwise | Return query unchanged |

Rewritten query is always shown in the debug panel.

### Answer formatting by intent

| Intent | Trigger words | Output format |
|--------|--------------|---------------|
| summarize | summarize, summary, main points, overview | Bullet points per key fact |
| who | who, sender, from, sent by | Sender/recipient lines with dates |
| when | when, date, time, day, month | Date lines with subjects |
| list | list, files, attachments, documents | Bullet list of filenames/items |
| general | anything else | Relevant snippet + metadata per chunk |

### Email cleaning

Before snippets are shown, raw forwarding headers are stripped:
`---Forwarded by---`, `From:`, `To:`, `Cc:`, `> quoted lines`,
`Please respond to`, `On ... wrote:` etc.

### Citation format

| Source | Format |
|--------|--------|
| Email | `[msg: <message_id>]` |
| Attachment | `[msg: <message_id>, page: N]` |

---

## Known Limitations

1. **No generative LLM on free path** — Answers are grounded snippets
   from retrieved chunks, not fluent summaries. Set `GROQ_API_KEY`
   env variable for LLM-generated answers (free at console.groq.com).

2. **Thread ID quality** — Some Enron emails have missing
   `Message-ID` / `References` headers so thread grouping may be
   imperfect for a few messages.

3. **No OCR** — Scanned PDFs without a text layer are not indexed.

4. **In-memory sessions** — Restarting the API server clears all
   active sessions.

5. **Small dataset** — With ~200 emails some queries may not find
   relevant results if the topic is not in the selected slice.

---

## API Endpoints

### `POST /start_session`
```json
Request:  { "thread_id": "<thread_id>" }
Response: { "session_id": "<uuid>", "thread_id": "<thread_id>" }
```

### `POST /ask`
```json
Request:
{
  "session_id": "16a13df8-4235-4bf0-adc2-0758639420d5",
  "text": "who sent this email and what it is about",
  "search_outside_thread": false
}

Response:
{
  "answer": "From: phillip.allen@enron.com | To: stagecoachmama@hotmail.com
             [msg: <5946371.1075855689136.JavaMail.evans@thyme>]
             Lucy, We can discuss your email later. How is progress on
             creating the spreadsheets. You will probably need to close
             the file before you attach to an email.
             [msg: <5946371.1075855689136.JavaMail.evans@thyme>]",
  "citations": [
    {
      "message_id": "<5946371.1075855689136.JavaMail.evans@thyme>",
      "type": "email",
      "page": null,
      "score": 0.817
    }
  ],
  "rewrite": "what is this email about",
  "retrieved": [
    {
      "id": "<5946371.1075855689136.JavaMail.evans@thyme>::body",
      "score": 0.817,
      "bm25_score": 1.000,
      "vec_score": 0.542
    }
  ],
  "trace_id": "c9b6e19f-aa5c-4ba4-a2c5-96275711ce1e"
}
```

### `POST /switch_thread`
```json
Request:  { "session_id": "<uuid>", "thread_id": "<new_thread_id>" }
Response: { "ok": true, "thread_id": "<new_thread_id>" }
```

### `POST /reset_session`
```json
Request:  { "session_id": "<uuid>" }
Response: { "ok": true }
```

### `GET /threads`
```json
Response:
{
  "<5946371.1075855689136.JavaMail.evans@thyme>": "Re: checkbook and budget",
  ...
}
```

### `GET /health`
```json
Response: { "status": "ok" }
```

---

## Sample Questions and Expected Citations

Tested against thread: **Re: checkbook and budget**
Thread ID: `<5946371.1075855689136.JavaMail.evans@thyme>`

| # | Question | Expected answer contains | Citation |
|---|----------|--------------------------|----------|
| 1 | Who sent this email and what is it about? | phillip.allen@enron.com, checkbook, spreadsheets | `[msg: <5946371.1075855689136.JavaMail.evans@thyme>]` |
| 2 | What files are being discussed? | spreadsheets, two files, close before attach | `[msg: <5946371.1075855689136.JavaMail.evans@thyme>]` |
| 3 | What did Phillip ask Lucy to do? | close the file before attaching to email | `[msg: <5946371.1075855689136.JavaMail.evans@thyme>]` |
| 4 | Summarise the main points | bullet: spreadsheets, progress, two files, Lucy | `[msg: <5946371.1075855689136.JavaMail.evans@thyme>]` |
| 5 | When was this email sent? | Wed 23 Aug 2000 07:09:00 | `[msg: <5946371.1075855689136.JavaMail.evans@thyme>]` |
| 6 | What is the subject of this thread? | Re: checkbook and budget | `[msg: <5946371.1075855689136.JavaMail.evans@thyme>]` |
| 7 | Who is Lucy? | stagecoachmama@hotmail.com, creating spreadsheets | `[msg: <5946371.1075855689136.JavaMail.evans@thyme>]` |
| 8 | What time was the email sent? | 07:09:00 -0700 PDT | `[msg: <5946371.1075855689136.JavaMail.evans@thyme>]` |

### Pronoun resolution test

| Turn | Input | Expected behaviour |
|------|-------|--------------------|
| 1 | What files are being discussed? | Returns spreadsheet info with citation |
| 2 | What did they say about it? | Rewrite box shows context from turn 1 prepended |

### Graceful failure test

| Input | Expected |
|-------|----------|
| What is the weather in New York? | I don't have information about that in this email thread |
| Who won the World Cup? | I don't have information about that in this email thread |

---

## Sample Trace Log

Real record from `runs/<timestamp>/trace.jsonl` captured during
a live test session on 15 March 2026:
```json
{
  "trace_id": "c057f8d3-4b3b-4e0f-a14d-0387f7455163",
  "session_id": "c605e143-20d8-4364-b2c0-9d4447fc7ac3",
  "user_text": "what is the 2 files she is talking about",
  "rewrite": "what is the 2 files she is talking about",
  "retrieved": [
    {
      "id": "<5946371.1075855689136.JavaMail.evans@thyme>::body",
      "score": 0.9045
    }
  ],
  "used_msgs": [
    "<5946371.1075855689136.JavaMail.evans@thyme>"
  ],
  "answer": "Lucy, We can discuss your email later. How is progress on
             creating the spreadsheets. You will probably need to close
             the file before you attach to an email. It is 2:00. I really
             want to make some progress on these two files. Phillip
             [msg: <5946371.1075855689136.JavaMail.evans@thyme>]",
  "citations": [
    {
      "message_id": "<5946371.1075855689136.JavaMail.evans@thyme>",
      "type": "email",
      "page": null,
      "score": 0.9045
    }
  ],
  "latency_ms": 36
}
```

**Fields explained:**

| Field | Value in this record | Meaning |
|-------|---------------------|---------|
| `user_text` | what is the 2 files she is talking about | Exactly what the user typed |
| `rewrite` | same as user_text | No pronoun detected — query unchanged |
| `retrieved` | 1 chunk, score 0.9045 | High confidence match found |
| `used_msgs` | `<5946371...>` | The email message used in the answer |
| `answer` | Lucy, We can discuss... | Grounded snippet from that message |
| `citations` | type: email, page: null | Email source, no attachment page |
| `latency_ms` | 36 | Full response in 36 milliseconds |

---

## How to Test

### Quick API test
```bash
# Health check
curl http://localhost:8000/health

# List all threads
curl http://localhost:8000/threads

# Start session on the checkbook thread
curl -X POST http://localhost:8000/start_session \
  -H "Content-Type: application/json" \
  -d "{\"thread_id\": \"<5946371.1075855689136.JavaMail.evans@thyme>\"}"

# Ask who sent the email (replace SESSION_ID)
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": \"SESSION_ID\", \"text\": \"Who sent this email?\"}"

# Follow-up with pronoun — tests memory
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": \"SESSION_ID\", \"text\": \"What did they say about it?\"}"

# Graceful failure test
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": \"SESSION_ID\", \"text\": \"What is the weather in New York?\"}"
```

### UI checklist

- [ ] Thread dropdown shows `Re: checkbook and budget` and others
- [ ] Start / Switch locks the thread (Thread ID shown in sidebar)
- [ ] Answers include inline `[msg: ...]` citations
- [ ] Follow-up pronouns trigger rewrite shown in debug panel
- [ ] Debug panel shows fused / BM25 / vector scores per chunk
- [ ] Search outside thread toggle returns results from all threads
- [ ] Reset conversation clears chat history
- [ ] Irrelevant question returns graceful failure message
- [ ] `runs/` folder has new `trace.jsonl` after each question

---

## Evaluation Criteria Coverage

| Criterion | How satisfied |
|-----------|--------------|
| **Grounding and faithfulness** | Every answer ends with `[msg: <5946371...>]` or `[msg: <id>, page: N]` for attachments. Score threshold of 0.35 prevents returning irrelevant chunks. No hallucination — confirmed by graceful failure test. |
| **Completeness and relevance** | Top-5 chunks per answer. Max 2 chunks per message_id forces variety. Intent detection gives summaries as bullets, who-questions as sender lines, when-questions as date lines. |
| **Thread discipline** | `start_session(thread_id)` filters every retrieval call to `Re: checkbook and budget`. Switching thread clears history. Explicit toggle required to search outside. |
| **Conversational understanding** | Pronoun resolution tested: "What did they say about it?" rewrites using previous answer. Short follow-ups prepend previous question. Last 5 turns + entities (dates, amounts, filenames) stored per session. Rewrite always visible in debug panel. |
| **Clarity** | Answers formatted by intent. Email headers stripped. Graceful failure on out-of-scope questions. One citation per source block. 36ms latency on warm cache. |
| **Transparency** | Debug panel: rewrite + chunk IDs + fused/BM25/vec scores. `trace.jsonl` per turn with all fields. Real trace shown above with field-by-field explanation. |
```

---

## Now do this — push it to GitHub

**Step 1** — Add your video link. Find this line at the top:
```
>