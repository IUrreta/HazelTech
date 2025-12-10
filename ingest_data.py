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
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import requests 


MANUAL_SOURCES = [
    {
        "id": "rav4_da_om42841e",
        "brand": "Toyota",
        "model": "RAV4",
        "year": 2019,
        "url": "https://myportalcontent.toyota-europe.com/Manuals/Toyota/Rav4_DA_OM42841E.pdf",
        "region": "EU",
        "doc_type": "owner_manual",
    },
    {
        "id": "yaris_hv_om52g10e",
        "brand": "Toyota",
        "model": "Yaris Hybrid",
        "year": 2020,
        "url": "https://myportalcontent.toyota-europe.com/Manuals/toyota/YARIS%2BHV_OM_Europe_OM52G10E.pdf",
        "region": "EU",
        "doc_type": "owner_manual",
    },
    {
        "id": "auris_hv_touring_sports_om12j36e",
        "brand": "Toyota",
        "model": "Auris HV Touring Sports",
        "year": 2014,
        "url": "https://myportalcontent.toyota-europe.com/Manuals/toyota/AURIS%2BHV%2BTouring%2BSports_OM_EE_OM12J36E.pdf",
        "region": "EU",
        "doc_type": "owner_manual",
    },
]

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    # A veces ayuda indicar el referer de la propia herramienta:
    "Referer": "https://www.toyota-europe.com/customer/manuals",
}

MANUAL_METADATA = {f"{m['id']}.pdf": m for m in MANUAL_SOURCES}

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

        # Si es un manual conocido, añade metadata extra
        meta = MANUAL_METADATA.get(filename)
        if meta:
            documents_map[filename].update({
                "brand": meta.get("brand"),
                "model": meta.get("model"),
                "year": meta.get("year"),
                "url": meta.get("url"),
                "doc_type": meta.get("doc_type"),
                "origin": "toyota_lexus_official"
            })

        # 2. Chunking
        raw_chunks = [c.strip() for c in re.split(r'\n\s*\n', content) if len(c.strip()) > 50]

        for i, chunk_text in enumerate(raw_chunks):
            chunk_vec = model.encode(chunk_text)
            chunk_id = hashlib.md5(f"{filename}_{i}".encode()).hexdigest()

            chunks_list.append({
                "chunk_id": chunk_id,
                "doc_id": filename,  # <--- aquí el 'Source:' de tu RAG serán los filenames .pdf
                "content": chunk_text,
                "embedding": chunk_vec
            })

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




def fetch_manual_sample():
    os.makedirs(DOCS_DIR, exist_ok=True)

    for m in MANUAL_SOURCES:
        filename = f"{m['id']}.pdf"
        filepath = os.path.join(DOCS_DIR, filename)

        if os.path.exists(filepath):
            print(f"Manual already exists, skipping: {filename}")
            continue

        print(f"Downloading manual {m['brand']} {m['model']} from {m['url']} ...")
        try:
            resp = requests.get(m["url"], headers=REQUEST_HEADERS, timeout=30)
            # Si Toyota devuelve 403 o lo que sea, raise_for_status lanza excepción
            resp.raise_for_status()

            with open(filepath, "wb") as f:
                f.write(resp.content)

            print(f"Saved manual to {filepath}")

        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else "unknown"
            print(f"HTTP error {status} downloading {m['id']}: {e}")
            if status == 403:
                print(
                    f" -> {filename} parece protegido (cookies/region/auth). "
                    f"Para la demo, descárgalo manualmente desde la web y guárdalo "
                    f"como {filepath}"
                )
        except Exception as e:
            print(f"Failed to download {m['id']} from {m['url']}: {e}")



if __name__ == "__main__":
    ingest_structured_data()

    # 1) Descargar muestra de manuales oficiales (Toyota/Lexus)
    fetch_manual_sample()

    # 3) Construir índice RAG con contratos + warranty + manuales
    build_vector_store()
