import os
import sqlite3
import pandas as pd
import glob
import pickle
import json
import re
from dotenv import load_dotenv
from openai import OpenAI
import hashlib

# Import pypdf
try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False
    print("Warning: pypdf not found. PDF ingestion will be skipped.")

# Import SentenceTransformer
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_BERT = True
except ImportError:
    HAS_SENTENCE_BERT = False
    print("Warning: sentence-transformers not found. Please install it.")

DATA_DIR = "data"
DOCS_DIR = "docs"
DB_PATH = "sales_data.db"
VECTOR_STORE_PATH = "vector_store.pkl"

# Load env for OpenAI summaries
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = None
if api_key and not api_key.startswith("sk-placeholder"):
    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        print(f"Failed to init OpenAI for ingestion: {e}")

def ingest_structured_data():
    print("Ingesting structured data into SQLite...")
    conn = sqlite3.connect(DB_PATH)
    
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    for file in csv_files:
        table_name = os.path.splitext(os.path.basename(file))[0]
        df = pd.read_csv(file)
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        print(f"Loaded {table_name} ({len(df)} rows)")
    
    conn.close()

def generate_summary(text, filename):
    """Generates a brief summary of the document using LLM or fallback."""
    if client:
        try:
            # Truncate text to avoid token limits for summary
            sample_text = text[:4000]
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Summarize the following document in 1-2 sentences for the purpose of information retrieval context."},
                    {"role": "user", "content": f"Filename: {filename}\nContent Sample: {sample_text}..."}
                ],
                max_tokens=100,
                temperature=0.5
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Summary generation failed for {filename}: {e}")
    
    # Fallback
    return f"Document {filename} context. ({len(text)} chars)"

def extract_text_from_pdf(filepath):
    """Extracts text from a PDF file."""
    if not HAS_PYPDF:
        return "", 0
    
    text = ""
    pages = 0
    try:
        reader = PdfReader(filepath)
        pages = len(reader.pages)
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                # Add page marker for context (optional, mostly for human reading)
                text += f"\n[Page {i+1}]\n{page_text}"
    except Exception as e:
        print(f"Error reading PDF {filepath}: {e}")
    
    return text, pages

def build_vector_store():
    if not HAS_SENTENCE_BERT:
        print("Skipping Vector Store build: SentenceTransformer not installed.")
        return

    print("Loading Embedding Model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Building RAG Index (PDFs & TXTs)...")
    
    # New Data Structure
    # documents: { doc_id (filename): { summary, summary_vec, page_count, type } }
    # chunks: [ { chunk_id, doc_id, content, embedding } ]
    
    documents_map = {}
    chunks_list = []
    
    # helper for processing
    def process_file(filepath, content, file_type, page_count=0):
        filename = os.path.basename(filepath)
        print(f"Processing {filename}...")
        
        # 1. Summary
        summary = generate_summary(content, filename)
        summary_vec = model.encode(summary)
        
        documents_map[filename] = {
            "summary": summary,
            "embedding": summary_vec,
            "page_count": page_count,
            "type": file_type
        }
        
        # 2. Chunking
        # Split by [Page X] or double newlines
        # For PDFs, let's try to split by some logic or just simple chunks. 
        # Simple paragraph split:
        raw_chunks = [c.strip() for c in re.split(r'\n\s*\n', content) if len(c.strip()) > 50]
        
        for i, chunk_text in enumerate(raw_chunks):
            # clean up page markers slightly if they are alone? No, keep context.
            
            chunk_vec = model.encode(chunk_text)
            chunk_id = hashlib.md5(f"{filename}_{i}".encode()).hexdigest()
            
            chunks_list.append({
                "chunk_id": chunk_id,
                "doc_id": filename,
                "content": chunk_text,
                "embedding": chunk_vec
            })
            
    # Process TXT
    for filepath in glob.glob(os.path.join(DOCS_DIR, "*.txt")):
        if os.path.basename(filepath) == "requirements.txt": continue # skip non-doc txts if any
        with open(filepath, "r", encoding='utf-8', errors='ignore') as f:
            content = f.read()
            process_file(filepath, content, "txt")

    # Process PDF
    for filepath in glob.glob(os.path.join(DOCS_DIR, "*.pdf")):
        content, pages = extract_text_from_pdf(filepath)
        if content:
            process_file(filepath, content, "pdf", page_count=pages)

    # Store
    store = {
        "documents": documents_map,
        "chunks": chunks_list,
        "model_name": 'all-MiniLM-L6-v2'
    }
    
    with open(VECTOR_STORE_PATH, "wb") as f:
        pickle.dump(store, f)
    
    print(f"Vector Store saved. Docs: {len(documents_map)}, Chunks: {len(chunks_list)}")

if __name__ == "__main__":
    # ingest_structured_data() # Skip to save time if not needed, but safe to run
    # fetch_manual_sample() # Skip if already exists or run to ensure
    ingest_structured_data()
    # Don't overwrite dynamic docs if possible, but for demo we can check exist
    if not os.path.exists(os.path.join(DOCS_DIR, "Manual_Yaris_Cross.txt")):
         print("Creating sample manual txt...")
         # (Simple rewrite of fetch if needed, or just let existing implementation stand)
         # Re-implementing simplified fetch for completeness of script:
         with open(os.path.join(DOCS_DIR, "Manual_Yaris_Cross.txt"), "w") as f:
            f.write("TOYOTA YARES CROSS MANUAL (SAMPLE)\n\n1. TIRE REPAIR KIT: Under deck board.")
            
    build_vector_store()
