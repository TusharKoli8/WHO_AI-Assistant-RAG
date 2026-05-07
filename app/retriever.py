from pathlib import Path
import pickle

import faiss
from sentence_transformers import SentenceTransformer


MODEL_NAME  = 'all-MiniLM-L6-v2'                 
BASE_DIR    = Path(__file__).resolve().parent.parent
STORE_DIR   = BASE_DIR / 'store'
INDEX_PATH  = STORE_DIR / 'faiss_index'
CHUNKS_PATH = STORE_DIR / 'chunks'


model = SentenceTransformer(MODEL_NAME)


## RETRIEVER FUNCTION

def retriever(query: str, k: int = 5):
    """
    Retrieve the top-k most relevant document chunks for a query.

    Args:
        query : The user's natural language question
        k     : Number of top chunks to return (default = 5)

    Returns:
        List of dicts, each with:
          - 'text'   : the chunk's raw text content
          - 'source' : which PDF file it came from
          - 'page'   : which page number it came from
    """

    if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            "  FAISS index or chunks not found.\n"
            "  Please run ingest.py first:\n"
            "  >> python app/ingest.py"
        )

    ##Load FAISS index
    
    index = faiss.read_index(str(INDEX_PATH))
    with open(CHUNKS_PATH, 'rb') as f:
        chunks = pickle.load(f)

    ## Embed 

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False  
    ).astype('float32')


    distances, indices = index.search(query_embedding, k)

    ##  Collect results 
    
    results = []
    for rank, i in enumerate(indices[0]):
        chunk = chunks[i]

        
        source = chunk.metadata.get('source', 'Unknown')
        page   = chunk.metadata.get('page', '?')

       
        source_name = Path(source).name

        results.append({
            'text'   : chunk.page_content,
            'source' : source_name,
            'page'   : page + 1,            
            'score'  : float(distances[0][rank]) 
        })

    return results
