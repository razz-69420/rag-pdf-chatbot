# RAG PDF Chatbot

An AI-powered chatbot that lets you upload any PDF and ask questions about it. Built with LangChain, Groq, ChromaDB, and Streamlit.

## Live Demo
[Insert Streamlit Cloud URL here]

## What It Does
- Upload any PDF → automatically chunked, embedded, and indexed
- Ask questions in natural language — including with typos
- Get answers grounded strictly in the document with source chunks shown
- Intelligent query expansion corrects misspellings using a document summary

## Tech Stack
| Layer | Tool |
|---|---|
| LLM | Groq API (Llama 3.3 70B) |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | ChromaDB (local) |
| PDF Parsing | PyPDF via LangChain |
| Orchestration | LangChain LCEL |
| Frontend | Streamlit |
| Deployment | Streamlit Cloud |

**Cost: $0.00**

## Features
- **Any PDF** — not hardcoded to one document
- **Query expansion** — generates a document summary on upload, uses it to correct typos before retrieval
- **Source transparency** — shows exact chunks used to generate every answer
- **Chat history** — maintains conversation context within a session
- **Hallucination prevention** — LLM is strictly grounded to document context only

## How It Works
```
User uploads PDF
       ↓
PyPDF extracts text
       ↓
RecursiveCharacterTextSplitter → chunks (700 chars, 200 overlap)
       ↓
Groq generates a 100-word document summary (stored in session)
       ↓
HuggingFace sentence-transformers embeds each chunk → ChromaDB
       ↓
User asks a question
       ↓
Groq corrects query using document summary
       ↓
Corrected query → similarity search → top 4 chunks retrieved
       ↓
Original question + chunks → Groq LLM → answer
       ↓
Answer + source chunks displayed in Streamlit
```

## Project Structure
```
rag-chatbot/
├── app.py              ← entire app in one file
├── requirements.txt
├── .env                ← GROQ_API_KEY (never pushed)
├── .gitignore
├── chroma_db/          ← auto-generated, never pushed
├── screenshots/
└── README.md
```

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
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add your Groq API key**
```
GROQ_API_KEY=your_key_here
```
Get a free key at [console.groq.com](https://console.groq.com)

**5. Run**
```bash
streamlit run app.py
```

## Deployment
Deployed on Streamlit Cloud. To deploy your own:
1. Push to GitHub
2. Connect repo at [share.streamlit.io](https://share.streamlit.io)
3. Add `GROQ_API_KEY` in Advanced Settings → Secrets

## Architecture Notes
- **LLM-agnostic** — swap Groq for OpenAI or AWS Bedrock in one line
- **Embedding model is local and free** — no API cost for embeddings
- **ChromaDB is ephemeral on Streamlit Cloud** — rebuilds per session, which matches the upload-per-session UX