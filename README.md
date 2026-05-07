#  WHO_AI Assistant

 An intelligent Q&A assistant powered by WHO (World Health Organization) publications using Retrieval-Augmented Generation (RAG).


##  Overview

WHO_AI Assistant is a RAG based application that allows users to ask questions and get accurate, grounded answers strictly from WHO documents. It combines semantic search (FAISS) with a powerful LLM (Groq) to retrieve relevant chunks from your WHO PDFs and generate cited responses.


##  Project Structure


WHO_AI_Assitant_RAG/

├── app/

│   ├── ingest.py        # Load PDFs → Chunk → Embed → Store in FAISS

│   ├── retriever.py     # Query → Embed → Search FAISS → Return chunks

│   ├── rag_chain.py     # Retrieve context + Generate answer via Groq

│   ├── agents.py        # Validate + Classify + Refine user queries

│   └── main.py          # Streamlit web UI

├── data/

│   └── WHO Clarified/   # Place your WHO PDF files here

├── store/               # Auto-created: stores FAISS index + chunks

├── .env                 # Your API keys (never commit this)

└── requirements.txt     # All dependencies



##  How It Works


User Question

      ↓

Agent Pipeline (agents.py)

  ├── 1. Validation Agent   → Is it relevant to WHO publications?

  ├── 2. Classifier Agent   → What category? (Treatment, Policy, Procurement...)

  └── 3. Query Refiner      → Rewrite for better search results

      ↓
Retriever (retriever.py)

  └── Convert query → embedding → search FAISS → return top 5 chunks

      ↓

RAG Chain (rag_chain.py)

  └── Build prompt with context → call Groq LLM → return answer + sources

      ↓
Streamlit UI (main.py)

  └── Display answer + source citations (filename + page number)



##  Tech Stack

| Layer | Tool |

|---|---|

| PDF Parsing | 'PyPDFLoader' (LangChain) |

| Text Splitting | 'RecursiveCharacterTextSplitter' |

| Embeddings | 'sentence-transformers' (all-MiniLM-L6-v2) |

| Vector Database | 'FAISS' (Facebook AI Similarity Search) |

| LLM | 'Groq' (llama-3.3-70b-versatile) |

| Agent Layer | 'Groq' (Validation + Classifier + Refiner) |

| Frontend | 'Streamlit' |


##  Getting Started

### 1. Clone the repository
---bash

git clone https://github.com/yourusername/WHO_AI_Assistant.git
cd WHO_AI_Assistant


### 2. Create and activate virtual environment
---bash

uv venv

# Windows

.venv\Scripts\activate

# Mac/Linux

source .venv/bin/activate
```

### 3. Install dependencies

---bash

pip install -r requirements.txt


### 4. Set up environment variables

Create a ".env" file in the project root:

GROQ_API_KEY=your_groq_api_key_here
MODEL_NAME=llama-3.3-70b-versatile

 Get your free Groq API key at: https://console.groq.com

### 5. Add your WHO PDF files

Place your PDF files in:

data/WHO Clarified/


### 6. Build the knowledge base (run once)
---bash

python app/ingest.py


### 7. Launch the app
---bash

streamlit run app/main.py


##  Example Questions

| Category | Example Question |
|---|---|
| Procurement | "What are WHO procurement procedures for medicines?" |
| Partnerships | "What are the risks of public-private partnerships in health?" |
| Community | "How should communities be engaged during a disease outbreak?" |
| Treatment | "What are WHO recommended treatments for Tuberculosis?" |
| Policy | "What is WHO's approach to risk communication in emergencies?" |
| General | "What is the role of WHO in global health?" |


##  Agent Pipeline

The app uses 3 intelligent agents before every query hits the RAG pipeline:

| Agent | Job |

|---|---|

| **Validation Agent** | Checks if the question is relevant to WHO publications. Rejects unrelated topics like sports, cooking, coding. |
| **Classifier Agent** | Labels the query into a category: Treatment, Procurement, Policy, Community Engagement, etc. |
| **Query Refiner Agent** | Rewrites vague short queries into detailed search queries for better FAISS retrieval. |


## Query Categories

- Treatment & Clinical Care
- Diagnosis
- Prevention & Vaccination
- Symptoms & Disease
- Dosage / Medication
- Risk Factors
- Health Guidelines & Policy
- Procurement & Supply Chain
- Public-Private Partnerships
- Community Engagement & Communication
- General WHO Knowledge


##  Disclaimer

This tool is for **research and informational purposes only**. Always consult a qualified healthcare professional for medical advice. Answers are strictly grounded in the WHO documents provided — the assistant will not answer questions outside the scope of these documents.




Built with  using LangChain, FAISS, Groq, and Streamlit.