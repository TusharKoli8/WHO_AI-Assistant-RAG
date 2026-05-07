import os
from dotenv import load_dotenv
from groq import Groq
from retriever import retriever
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")

##  System Prompt 

SYSTEM_PROMPT = """You are WHO_AI-Assistant, an intelligent Q&A assistant specialized in WHO (World Health Organization) publications and documents.

You can answer questions related to:
- Medical conditions, treatments, and clinical guidelines
- WHO health policies and recommendations
- Public health and disease prevention
- Procurement and supply chain topics
- Public private partnerships in health
- Community engagement and health communication
- Any other topic covered in the provided WHO documents

Your rules:
1. Answer ONLY using the provided context from WHO documents.
2. If the answer is NOT present in the context, respond with:
   "I don't have enough information in the provided WHO documents to answer this question."
3. Always be factual, clear, and concise.
4. Do NOT make up or assume any information outside the context.
5. At the end of your answer, always mention the source document and page number provided in the context.

Remember: You are a research assistant grounded strictly in WHO publications."""


## BUILD PROMPT

def build_prompt(query: str, context_chunks: list) -> str:
    """
    Combine retrieved chunks into a structured prompt for the LLM.
    Each chunk includes its source and page for citation.
    """

    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        context_parts.append(
            f"[Chunk {i} | Source: {chunk['source']} | Page: {chunk['page']}]\n"
            f"{chunk['text']}"
        )

    context_text = "\n\n".join(context_parts)

    prompt = f"""
Context from WHO Documents:
{context_text}

---
Question: {query}

Answer based strictly on the above context:"""

    return prompt


## GENERATE ANSWER (Main RAG Function)

def generate_answer(query: str):
    """
    Full RAG pipeline: retrieve → build prompt → generate answer.

    Args:
        query : The user's question about WHO publications

    Returns:
        dict with:
          - 'answer'  : The LLM's generated answer
          - 'sources' : List of source documents used
    """

    ## Retrieve relevant chunks
    context_chunks = retriever(query, k=5)

    ## Build the prompt 
    prompt = build_prompt(query, context_chunks)

    ## Call Groq LLM 
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.1,   
        max_tokens=1024    
    )

    answer = response.choices[0].message.content

    ## Collect unique sources for citation 
    
    seen = set()
    sources = []
    for chunk in context_chunks:
        key = (chunk['source'], chunk['page'])
        if key not in seen:
            seen.add(key)
            sources.append({
                'file' : chunk['source'],
                'page' : chunk['page']
            })

    return {
        'answer'  : answer,
        'sources' : sources
    }
