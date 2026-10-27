import os

import streamlit as st

st.set_page_config(page_title="Book RAG Chat", page_icon="📚", layout="wide")

# يدعم العربي (RTL تلقائي حسب لغة الجملة)
st.markdown("""
<style>
.stMarkdown p, .stMarkdown li { unicode-bidi: plaintext; text-align: start; }
</style>
""", unsafe_allow_html=True)

if not os.getenv("GOOGLE_API_KEY"):
    from dotenv import load_dotenv
    load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    st.error("مفيش GOOGLE_API_KEY. اعمل ملف `.env` وحط فيه: GOOGLE_API_KEY=...")
    st.stop()

try:
    with st.spinner("⏳ بحمّل الكتاب وبعمل الـ indexing (أول مرة بس)..."):
        import rag
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ الإعدادات")
    st.caption(f"📖 {rag.KNOWLEDGE_BASE}")
    st.caption(f"{len(rag.chunks)} chunk • {rag.num_pages} صفحة")

    k = st.slider("عدد النتايج لكل query (k)", 2, 10, 4)
    show_queries = st.toggle("اعرض الـ Queries", value=True)

    st.divider()
    use_filter = st.checkbox("🔎 دوّر في صفحات معينة بس (Metadata Filter)")
    page_range = None
    if use_filter and rag.num_pages > 1:
        lo, hi = st.slider("الصفحات", 1, rag.num_pages, (1, min(50, rag.num_pages)))
        page_range = (lo - 1, hi - 1)   # PyPDFLoader بيبدأ من 0

    st.divider()
    if st.button("🗑️ امسح المحادثة", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------- Chat ----------------
st.title("📚 اسأل الكتاب")
st.caption("Rewrite → Expand → Hybrid Search (Qdrant + BM25) → Gemini")

if "messages" not in st.session_state:
    st.session_state.messages = []


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("queries") and show_queries:
            with st.expander("🔎 الـ Queries"):
                for q in m["queries"]:
                    st.markdown(f"- {q}")

if question := st.chat_input("اكتب سؤالك عن الكتاب..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.status("🤔 بفكر وبدور...", expanded=show_queries) as status:
                st.write("✍️ Query Rewriting")
                rewritten = rag.rewrite_query(question)
                st.write(f"→ {rewritten}")

                st.write("🧩 Query Expansion")
                expanded = rag.expand_query(rewritten)
                queries = [rewritten] + expanded
                for q in expanded:
                    st.write(f"- {q}")

                st.write("🔀 Hybrid Search")
                docs = rag.retrieve_docs(queries, page_range, k)
                status.update(label=f"✅ لقيت {len(docs)} مقطع", state="complete", expanded=False)

            answer = st.write_stream(rag.stream_answer(question, docs))
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "queries": queries}
            )
        except Exception as e:
            st.error(f"حصل خطأ: {e}")