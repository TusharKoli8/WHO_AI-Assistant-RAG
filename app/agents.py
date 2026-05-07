import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL  = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")

##  Query categories 
QUERY_TYPES = [
    "Treatment & Clinical Care",
    "Diagnosis",
    "Prevention & Vaccination",
    "Symptoms & Disease",
    "Dosage / Medication",
    "Risk Factors",
    "Health Guidelines & Policy",
    "Procurement & Supply Chain",
    "Public-Private Partnerships",
    "Community Engagement & Communication",
    "General WHO Knowledge",
    "Out of Scope"
]


##  VALIDATION AGENT

def validation_agent(query: str) -> dict:
    """
    Check if the query is relevant to WHO publications.

    Returns:
        dict:
          - 'is_valid' : True / False
          - 'reason'   : Why it was accepted or rejected
    """

    prompt = f"""You are a query validator for a WHO Publications assistant.

Your ONLY job is to decide if the following question can be answered
from WHO (World Health Organization) documents. Accept questions related to:
- Medical conditions, diseases, symptoms, treatments, medications
- Health guidelines, clinical recommendations, policies
- Disease prevention, vaccination, risk factors
- Public health, health emergencies, epidemics
- WHO procurement and supply chain topics
- Public-private partnerships in health
- Community health engagement and communication
- Any topic that WHO publishes reports or guidelines about

Reject ONLY questions completely unrelated to WHO or public health
(e.g. sports, cooking, entertainment, coding, math).

Question: "{query}"

Respond in this EXACT format (no extra text):
VALID: yes
REASON: <one sentence reason>

OR

VALID: no
REASON: <one sentence reason>"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=100
    )

    output = response.choices[0].message.content.strip()
    lines    = output.split('\n')
    is_valid = 'yes' in lines[0].lower() if lines else False
    reason   = lines[1].replace('REASON:', '').strip() if len(lines) > 1 else ''

    return {
        'is_valid' : is_valid,
        'reason'   : reason
    }

##  CLASSIFIER AGENT

def classifier_agent(query: str) -> str:
    """
    Classify the query into a WHO publication category.

    Returns:
        str: One of the QUERY_TYPES categories
    """

    categories = "\n".join([f"- {q}" for q in QUERY_TYPES])

    prompt = f"""You are a query classifier for WHO Publications.

Classify the following question into EXACTLY ONE of these categories:
{categories}

Question: "{query}"

Respond with ONLY the category name, nothing else."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=20
    )

    category = response.choices[0].message.content.strip()

    if category not in QUERY_TYPES:
        category = "General WHO Knowledge"

    return category

##  QUERY REFINER AGENT

def query_refiner_agent(query: str) -> str:
    """
    Rewrite a vague query into a clear, detailed search query
    optimized for retrieving relevant WHO document chunks.

    Example:
        Input : "who procurement?"
        Output: "What are the WHO guidelines and procedures
                 for procurement and supply chain management?"

    Returns:
        str: Refined query string
    """

    prompt = f"""You are a query refinement assistant for a WHO Publications Q&A system.

Your job is to rewrite the user's query into a clear, specific, and detailed
question that will work well for searching WHO documents.

Rules:
- Keep the same meaning and intent
- Make it more specific and complete
- Use proper terminology relevant to WHO topics
- Output ONLY the refined query, nothing else

Original query: "{query}"

Refined query:"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=100
    )

    refined = response.choices[0].message.content.strip()
    refined = refined.strip('"').strip("'")

    return refined

## MASTER AGENT PIPELINE

def run_agent_pipeline(query: str) -> dict:
    """
    Run the full agent pipeline before sending to RAG.

    Pipeline:
        1. Validate  → Is it relevant to WHO publications?
        2. Classify  → What category does it fall under?
        3. Refine    → Rewrite for better FAISS retrieval

    Returns:
        dict:
          - 'is_valid'       : True / False
          - 'reject_reason'  : Why rejected (if not valid)
          - 'category'       : Query type label
          - 'refined_query'  : Improved query for FAISS search
          - 'original_query' : Original user query
    """

    ##  Validate
    validation = validation_agent(query)

    if not validation['is_valid']:
        return {
            'is_valid'       : False,
            'reject_reason'  : validation['reason'],
            'category'       : 'Out of Scope',
            'refined_query'  : query,
            'original_query' : query
        }

    ##  Classify
    category = classifier_agent(query)

    ## Refine
    refined_query = query_refiner_agent(query)

    return {
        'is_valid'       : True,
        'reject_reason'  : None,
        'category'       : category,
        'refined_query'  : refined_query,
        'original_query' : query
    }
