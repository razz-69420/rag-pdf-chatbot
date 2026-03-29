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
                chunk_size=700,
                chunk_overlap=200
            )
            chunks = splitter.split_documents(documents)

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory="chroma_db"
            )

            st.session_state.vectorstore = vectorstore
            st.session_state.pdf_name = uploaded_file.name
            st.session_state.chat_history = []
            os.unlink(tmp_path)

            # Generate document summary for query expansion
            full_text = " ".join([doc.page_content[:300] for doc in chunks[:15]])
            summary_llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                api_key=os.getenv("GROQ_API_KEY"),
                temperature=0
            )
            summary_response = summary_llm.invoke(
                f"Summarize the key topics, names, companies, and terms in this document in 100 words max:\n\n{full_text}"
            )
            st.session_state.doc_summary = summary_response.content.strip()

        st.success(f"✅ {uploaded_file.name} processed — {len(chunks)} chunks indexed")

    if st.session_state.pdf_name:
        st.info(f"Active doc: **{st.session_state.pdf_name}**")

PROMPT = PromptTemplate.from_template(""""I want to answer user questions using ONLY the retrieved document context so that every response is accurate, grounded, and easy to read with zero hallucinations.

If the question contains a name or term that closely matches something in the context, treat it as referring to that term and answer accordingly.

FORMATTING RULES:
- Use bullet points for lists or multiple facts
- Use short paragraphs separated by line breaks for explanations
- Bold important terms using **term**
- Never return a single blob of text
- Never repeat the same point twice
- Stop as soon as the answer is complete
- Interpret typos/messy phrasing silently. 
- Never mention you corrected anything.
- Reference the document naturally e.g. "According to the document..."
- Aim for under 150 words unless depth is genuinely needed.

For complex answers, structure as: direct answer first → supporting detail → broader context if needed.                                                                                                                                                   

If the entire message is a greeting with no question (e.g. "yo", "hey", "hi"), respond with "Hey! Ask me anything about the document." and nothing else.
                             
Only say "I don't have enough information from the provided sources." if there is truly zero relevant information in the context.

Context:
{context}

Question: {question}

Answer:""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def expand_query(question, llm, doc_summary=""):
    response = llm.invoke(
        f"""You are a spelling corrector for a document search system.
Fix any spelling mistakes in this search query by matching to the closest term that actually appears in the document summary below.
Do NOT invent or substitute terms that are not in the summary.
Return only the corrected query, nothing else.

Document summary: {doc_summary}

Query: {question}"""
    )
    return response.content.strip()

if not st.session_state.vectorstore:
    st.info("👈 Upload a PDF from the sidebar to get started.")
else:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("📎 Sources"):
                    for i, chunk in enumerate(msg["sources"], 1):
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
                    max_tokens=1024
                )

                retriever = st.session_state.vectorstore.as_retriever(
                    search_kwargs={"k": 4}
                )

                corrected = expand_query(question, llm, st.session_state.doc_summary)
                source_docs = retriever.invoke(corrected)

                chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | PROMPT
                    | llm
                    | StrOutputParser()
                )

                answer = chain.invoke(question)

            st.markdown(answer)
            with st.expander("📎 Sources"):
                for i, chunk in enumerate(source_docs, 1):
                    st.markdown(f"**Chunk {i} — Page {chunk.metadata.get('page', 0) + 1}**")
                    st.caption(chunk.page_content)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": source_docs
        })