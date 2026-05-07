from pathlib import Path
import pickle
import faiss
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'  

BASE_DIR  = Path(__file__).resolve().parent.parent  

DATA_DIR  = Path(r"C:\Class_Intellbi\Gen_AI\Generative_AI\Projects\WHO_AI_Assitant_RAG\data\WHO Clarified")

STORE_DIR = BASE_DIR / 'store'                       

INDEX_PATH  = STORE_DIR / 'faiss_index'  
CHUNKS_PATH = STORE_DIR / 'chunks'       

## LOAD DOCUMENTS

def load_docs(data_path: Path):
    docs = []
    pdf_files = list(data_path.glob('*.pdf'))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {data_path}")

    print(f"\n Found {len(pdf_files)} PDF(s) in /data:")
    for file in pdf_files:
        print(f"   └── {file.name}")
        loader = PyPDFLoader(str(file))
        loaded = loader.load()
        docs.extend(loaded)
        print(f" Loaded {len(loaded)} pages")

    return docs


## SPLIT DOCUMENTS INTO CHUNKS


def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)
    return chunks

## CREATE EMBEDDINGS

def create_embeddings(chunks):
    print(f"\n Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    texts = [chunk.page_content for chunk in chunks]  ## Extract plain text

    print(f" Embedding {len(texts)} chunks... (this may take a minute)")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,      
        convert_to_numpy=True,        
        normalize_embeddings=True      
    ).astype('float32')               

    return embeddings, chunks


## STORE IN FAISS VECTOR DATABASE

def store_faiss(embeddings, chunks):
    STORE_DIR.mkdir(exist_ok=True)  

    dim = embeddings.shape[1]     
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)          

    faiss.write_index(index, str(INDEX_PATH))
    print(f"\n FAISS index saved → {INDEX_PATH}")

    with open(CHUNKS_PATH, 'wb') as f:
        pickle.dump(chunks, f)
    print(f" Chunks saved      → {CHUNKS_PATH}")


## MAIN — Run all  steps in order

if __name__ == '__main__':
    print("=" * 55)
    print(" WHO_AI — Ingestion Pipeline")
    print("=" * 55)

    ## Step 1
    docs = load_docs(DATA_DIR)
    print(f"\n Total pages loaded: {len(docs)}")

    ## Step 2
    chunks = split_docs(docs)
    print(f" Total chunks created: {len(chunks)}")

    ## Step 3
    embeddings, chunks = create_embeddings(chunks)
    print(f" Embedding shape: {embeddings.shape}")

    ## Step 4
    store_faiss(embeddings, chunks)

    print("\n" + "=" * 55)
    print("    Knowledge base built successfully!")
    print("   You can now run: streamlit run app/main.py")
    print("=" * 55)
