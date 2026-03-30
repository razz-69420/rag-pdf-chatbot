import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

st.set_page_config(page_title="RAG PDF Chatbot", page_icon="📄", layout="wide")
st.title("📄 RAG PDF Chatbot")
st.caption("Upload a PDF → ask questions → get answers grounded in the document")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "doc_summary" not in st.session_state:
    st.session_state.doc_summary = ""

with st.sidebar:
    st.header("📁 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file and uploaded_file.name != st.session_state.pdf_name:
        with st.spinner("Processing PDF..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)
            documents = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=512,
                chunk_overlap=80,  # ~15% — sweet spot per benchmarks
                separators=["\n\n", "\n", ".", " ", ""]
            )
            chunks = splitter.split_documents(documents)

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings
            )

            st.session_state.vectorstore = vectorstore
            st.session_state.pdf_name = uploaded_file.name
            st.session_state.chat_history = []
            os.unlink(tmp_path)

            # Generate document summary for query expansion
            total = len(chunks)
            sample_indices = (
                list(range(min(5, total))) +
                list(range(total//2 - 2, total//2 + 3)) +
                list(range(max(0, total-5), total))
            )
            
            sample_indices = sorted(set(i for i in sample_indices if 0 <= i < total))
            full_text = " ".join([chunks[i].page_content[:400] for i in sample_indices])
            
            summary_llm = ChatGroq(
                model="llama-3.1-8b-instant",
                api_key=os.getenv("GROQ_API_KEY"),
                temperature=0
            )
            summary_response = summary_llm.invoke(
                f"""Extract the key topics, named entities, technical terms, and domain-specific vocabulary from this document.
                Be exhaustive with terminology — prioritize coverage over brevity.
                Format: comma-separated list of terms and short phrases only. No sentences.

                Document:
                {full_text}"""
            )

            st.session_state.doc_summary = summary_response.content.strip()

        st.success(f"✅ {uploaded_file.name} processed — {len(chunks)} chunks indexed")

    if st.session_state.pdf_name:
        st.info(f"Active doc: **{st.session_state.pdf_name}**")

PROMPT = PromptTemplate.from_template("""You answer questions using ONLY the retrieved context below. No outside knowledge. Ever.

Context:
{context}

RULES (follow all, always):
- Answer immediately. Zero preamble, no restating the question.
- If the context contains the answer, give it — interpret typos silently.
- If context has zero relevant info → say only: "I couldn't find that in the document."
- If the message is a greeting with no question → say only: "Hey! Ask me anything about the document."
- Never repeat a point. Stop when done. No filler.
- Use bullets for multiple facts. Bold **key terms**. No walls of text.
- Cite page numbers inline where available, e.g. *(p. 3)*.

Question: {question}

Answer:""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def expand_query(question, llm, doc_summary=""):
    response = llm.invoke(
        f"""Fix any spelling mistakes in this search query using only terms from the document summary below.
Do NOT substitute terms not in the summary. Return only the corrected query, nothing else.

Summary: {doc_summary}
Query: {question}"""
    )
    return response.content.strip()

if not st.session_state.vectorstore:
    st.info("👈 Upload a PDF from the sidebar to get started.")
else:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                valid_sources = [c for c in msg["sources"] if c.page_content.strip()]
                if valid_sources:
                    with st.expander("📎 Sources"):
                        for i, chunk in enumerate(valid_sources, 1):
                            st.markdown(f"**Chunk {i} — Page {chunk.metadata.get('page', 0) + 1}**")
                            st.caption(chunk.page_content)

    question = st.chat_input("Ask something about the document...")

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                llm = ChatGroq(
                    model="llama-3.3-70b-versatile",
                    api_key=os.getenv("GROQ_API_KEY"),
                    temperature=0,
                    max_tokens=768
                )

                retriever = st.session_state.vectorstore.as_retriever(
                    search_kwargs={"k": 4}
                )

                corrected = expand_query(question, llm, st.session_state.doc_summary)
                source_docs = retriever.invoke(corrected)
                context = format_docs(source_docs)

                chain = (
                    {"context": lambda _: context, "question": RunnablePassthrough()}
                    | PROMPT
                    | llm
                    | StrOutputParser()
                )

                answer = chain.invoke(question)
                
                if "I couldn't find that in the document" in answer:
                    source_docs = []
            
            st.markdown(answer)
            valid_sources = [c for c in source_docs if c.page_content.strip()]
            if valid_sources:
                with st.expander("📎 Sources"):
                    for i, chunk in enumerate(valid_sources, 1):
                        st.markdown(f"**Chunk {i} — Page {chunk.metadata.get('page', 0) + 1}**")
                        st.caption(chunk.page_content)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": source_docs
        })