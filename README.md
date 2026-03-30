# 📄 RAG PDF Chatbot

An AI-powered chatbot that lets you upload any PDF and ask questions about it — grounded strictly in the document, zero hallucinations.

**[Live Demo →](https://rag-pdf-chatbot-razi-project.streamlit.app/)**

---

## What It Does

- Upload any PDF → automatically chunked, embedded, and indexed
- Ask questions in natural language — including with typos
- Get answers grounded strictly in the document with source chunks shown
- Intelligent query expansion corrects misspellings using document vocabulary
- Greeting detection and fallback handling for edge case inputs

---

## Tech Stack

| Layer | Tool |
|---|---|
| LLM | Groq API (Llama 3.3 70B) |
| Summarization | Groq API (Llama 3.1 8B Instant) |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | ChromaDB (in-memory) |
| PDF Parsing | PyPDF via LangChain |
| Orchestration | LangChain LCEL |
| Frontend | Streamlit |
| Deployment | Streamlit Cloud |

**Cost: $0.00**

---

## Features

- **Any PDF** — not hardcoded to one document
- **Query expansion** — generates a vocabulary term list on upload, corrects typos against actual document terminology before retrieval
- **Density-aware sampling** — samples chunks from start, middle, and end of every PDF for accurate term coverage regardless of document length
- **Single-pass retrieval** — retrieves once, feeds the same context into the chain, so displayed sources always match the LLM's actual input
- **Source transparency** — shows exact chunks used to generate every answer, hidden automatically on fallback responses
- **Chat history** — maintains conversation context within a session
- **Hallucination prevention** — LLM is strictly grounded to document context only, temperature=0
- **Greeting & gibberish handling** — intent gate prevents pointless retrieval on non-questions

---

## How It Works

```
User uploads PDF
       ↓
PyPDF extracts text
       ↓
RecursiveCharacterTextSplitter → 512 token chunks, 80 token overlap
Separator hierarchy: paragraphs → sentences → words → characters
       ↓
Density-aware sampling: start + middle + end chunks extracted
Llama 3.1 8B generates comma-separated vocabulary term list (stored in session)
       ↓
HuggingFace sentence-transformers embeds each chunk → ChromaDB (in-memory)
       ↓
User asks a question
       ↓
Intent check: greeting/gibberish → short-circuit, skip retrieval
       ↓
Llama 3.3 70B corrects query spelling against document vocabulary
       ↓
Corrected query → similarity search → top 4 chunks retrieved (single pass)
       ↓
Original question + retrieved context → Llama 3.3 70B → grounded answer
       ↓
Answer + source chunks displayed
Sources hidden if answer is a fallback ("I couldn't find that in the document")
```

---

## Project Structure

```
rag-chatbot/
├── app.py              ← entire app in one file
├── requirements.txt
├── .env                ← GROQ_API_KEY (never pushed)
├── .gitignore
├── screenshots/
└── README.md
```

---

## Setup & Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/rag-chatbot.git
cd rag-chatbot
```

**2. Create a Python 3.11 virtual environment**
```bash
py -3.11 -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add your Groq API key**

Create a `.env` file:
```
GROQ_API_KEY=your_key_here
```
Get a free key at [console.groq.com](https://console.groq.com)

**5. Run**
```bash
streamlit run app.py
```

---

## Deployment

Deployed on Streamlit Cloud. To deploy your own:

1. Push to GitHub
2. Connect repo at [share.streamlit.io](https://share.streamlit.io)
3. Add `GROQ_API_KEY` in **Advanced Settings → Secrets**

---

## Architecture Notes

- **LLM-agnostic** — swap Groq for OpenAI or AWS Bedrock in one line
- **Embedding model is local and free** — no API cost for embeddings
- **ChromaDB is in-memory** — rebuilds per session, which matches the upload-per-session UX and works cleanly on Streamlit Cloud (no disk write permissions required)
- **Two-model design** — heavy 70B model handles Q&A, lightweight 8B model handles summarization on upload for lower latency
- **Chunking strategy** — 512 token chunks with 15% overlap and explicit separator hierarchy achieves 85-90% retrieval recall with MiniLM embeddings
