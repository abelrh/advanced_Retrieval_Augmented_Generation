import os
import re

from dotenv import load_dotenv
load_dotenv()


from langchain_community.document_loaders import PyPDFLoader

PDF_PATH = r"D:\LLM\AI Agent\project\python-for-data-analysis.pdf"

if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(f"الملف مش موجود: {PDF_PATH}")

documents = PyPDFLoader(PDF_PATH).load()
print(f"تم تحميل {len(documents)} صفحة")

num_pages = max(d.metadata.get("page", 0) for d in documents) + 1



from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
print(f"عدد الـ chunks: {len(chunks)}")


from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    location=":memory:",
    collection_name="advanced_rag",
)

from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)




from qdrant_client.models import Filter, FieldCondition, Range
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever


def build_hybrid_retriever(page_range=None, k=4):
    conditions, bm25_docs = [], chunks

    if page_range:
        lo, hi = page_range
        conditions.append(FieldCondition(key="metadata.page", range=Range(gte=lo, lte=hi)))
        bm25_docs = [c for c in bm25_docs if lo <= c.metadata.get("page", -1) <= hi]

    vec_kwargs = {"k": k}
    if conditions:
        vec_kwargs["filter"] = Filter(must=conditions)

    vector_retriever = vectorstore.as_retriever(search_kwargs=vec_kwargs)
    bm25_retriever = BM25Retriever.from_documents(bm25_docs)
    bm25_retriever.k = k

    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5],
    )


# ============ 6) Query Rewriting + Query Expansion ============
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

KNOWLEDGE_BASE = "Python for Data Analysis"   # غيّرها لاسم كتابك

rewrite_prompt = ChatPromptTemplate.from_template("""
You are a query rewriting system for a RAG system.
Rewrite the user's question into one clear, specific search query in English.
Keep the original meaning.
Do not answer the question.
Do not add unrelated information.
The knowledge base is: "{kb}"

User Question:
{question}

Return ONLY the rewritten query.
""")

expansion_prompt = ChatPromptTemplate.from_template("""
You are a query expansion system for a RAG system.
The knowledge base is: "{kb}"
Given the user's question, generate 5 different English search queries
that explore the same information using different relevant terminology.
The queries should:
- Preserve the original intent.
- Use different wording.
- Include important technical terms when relevant.
- Not answer the question.

User Question:
{question}

Return exactly 5 queries, one per line, with no numbering.
""")

rewrite_chain = rewrite_prompt | llm | StrOutputParser()
expansion_chain = expansion_prompt | llm | StrOutputParser()


def rewrite_query(question):
    return rewrite_chain.invoke({"question": question, "kb": KNOWLEDGE_BASE}).strip()


def expand_query(question):
    text = expansion_chain.invoke({"question": question, "kb": KNOWLEDGE_BASE})
    lines = [re.sub(r"^\s*(\d+[\.\)]|[-*•])\s*", "", l).strip() for l in text.splitlines()]
    return [l for l in lines if l][:5]


# ============ 7) Retrieval + إجابة Gemini ============
answer_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say you could not find it.
Answer in the same language as the question (Egyptian Arabic is fine).
Keep code and technical terms in English. Use markdown and code blocks when useful.

Context:
{context}

Question:
{question}
""")
answer_chain = answer_prompt | llm | StrOutputParser()


def retrieve_docs(queries, page_range=None, k=4, max_docs=8):
    retriever = build_hybrid_retriever(page_range, k)
    seen, docs = set(), []
    for q in queries:
        for d in retriever.invoke(q):
            key = (d.metadata.get("page"), d.page_content)
            if key not in seen:
                seen.add(key)
                docs.append(d)
    return docs[:max_docs]


def stream_answer(question, docs):
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    yield from answer_chain.stream({"context": context, "question": question})


# ============ تجربة من الـ Terminal (اختياري): python rag.py ============
if __name__ == "__main__":
    q = "إزاي أتعامل مع القيم الفاضية في pandas؟"
    rewritten = rewrite_query(q)
    queries = [rewritten] + expand_query(rewritten)
    print("Queries:", *queries, sep="\n - ")
    docs = retrieve_docs(queries)
    print("\n💬", "".join(stream_answer(q, docs)))