import streamlit as st
import httpx

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Email RAG Chatbot", layout="wide")
st.title("📧 Email Thread RAG Chatbot")


# ── initialise session state ─────────────────────────────────────────────────
for key, default in [
    ("session_id", None),
    ("messages",   []),
    ("debug_log",  []),
    ("thread_id",  None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Thread Selection")

    try:
        threads_resp = httpx.get(f"{API_URL}/threads", timeout=5)
        threads      = threads_resp.json()
    except Exception:
        st.error("Cannot connect to API. Make sure uvicorn is running on port 8000.")
        st.stop()

    thread_ids = list(threads.keys())

    if not thread_ids:
        st.warning("No threads found. Run ingest.py first.")
        st.stop()

    selected_thread = st.selectbox(
        "Pick a thread",
        thread_ids,
        format_func=lambda t: (
            threads[t][:50] + "..." if len(threads[t]) > 50 else threads[t]
        ),
    )

    if st.button("Start / Switch to this thread", type="primary"):
        if st.session_state.session_id is None:
            resp = httpx.post(
                f"{API_URL}/start_session",
                json={"thread_id": selected_thread},
            ).json()
            st.session_state.session_id = resp["session_id"]
        else:
            httpx.post(
                f"{API_URL}/switch_thread",
                json={
                    "session_id" : st.session_state.session_id,
                    "thread_id"  : selected_thread,
                },
            )
        st.session_state.thread_id = selected_thread
        st.session_state.messages  = []
        st.session_state.debug_log = []
        st.success("Thread locked!")

    if st.button("Reset conversation"):
        if st.session_state.session_id:
            httpx.post(
                f"{API_URL}/reset_session",
                json={"session_id": st.session_state.session_id},
            )
            st.session_state.messages  = []
            st.session_state.debug_log = []
            st.success("Conversation reset.")

    st.divider()
    search_outside = st.toggle("Search outside thread", value=False)

    st.divider()
    st.caption(f"Session ID : {st.session_state.session_id or 'None'}")
    st.caption(f"Thread ID  : {st.session_state.thread_id  or 'None'}")


# ── main layout: chat + debug ─────────────────────────────────────────────────
chat_col, debug_col = st.columns([3, 2])

with chat_col:
    st.subheader("Chat")

    # show full conversation
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # chat input box
    user_input = st.chat_input("Ask something about this email thread...")

    if user_input:
        if not st.session_state.session_id:
            st.error(
                "Please select a thread and click "
                "'Start / Switch to this thread' first."
            )
        else:
            # show user message immediately
            st.session_state.messages.append({
                "role"    : "user",
                "content" : user_input,
            })

            # call API
            with st.spinner("Searching..."):
                try:
                    resp = httpx.post(
                        f"{API_URL}/ask",
                        json={
                            "session_id"            : st.session_state.session_id,
                            "text"                  : user_input,
                            "search_outside_thread" : search_outside,
                        },
                        timeout=30,
                    ).json()
                except Exception as e:
                    st.error(f"API error: {e}")
                    st.stop()

            # show assistant reply
            st.session_state.messages.append({
                "role"    : "assistant",
                "content" : resp["answer"],
            })
            st.session_state.debug_log.append(resp)
            st.rerun()


with debug_col:
    st.subheader("Debug Panel")

    if not st.session_state.debug_log:
        st.info("Ask a question to see debug info here.")
    else:
        latest = st.session_state.debug_log[-1]

        # Query rewrite
        st.markdown("**Query rewrite**")
        st.code(latest.get("rewrite", "—"), language=None)

        # Top retrieved chunks with all 3 scores
        st.markdown("**Top retrieved** (fused | bm25 | vec)")
        for r in latest.get("retrieved", [])[:5]:
            fused = r.get("score", 0)
            bm25  = r.get("bm25_score", 0)
            vec   = r.get("vec_score", 0)
            label = r["id"][:45]
            st.text(f"{fused:.3f} | {bm25:.3f} | {vec:.3f}  —  {label}")

        # Citations
        st.markdown("**Citations**")
        for c in latest.get("citations", []):
            page_str = f", page {c['page']}" if c.get("page") else ""
            st.code(
                f"[msg: {c['message_id']}{page_str}]",
                language=None,
            )

        # Trace ID
        st.markdown("**Trace ID**")
        st.caption(latest.get("trace_id", "—"))