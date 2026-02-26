"""
Dodaj novi PDF u postojeću FAISS bazu bez dupliciranja
"""
import os
import asyncio
from pathlib import Path
from typing import List, Dict
import argparse

# PDF Processing
from pypdf import PdfReader

# Database and Embeddings
import faiss
import numpy as np
import pickle
from openai import AsyncOpenAI
import uuid
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./faiss_data/serbian_history.index")
FAISS_METADATA_PATH = os.getenv("FAISS_METADATA_PATH", "./faiss_data/metadata.pkl")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# Period mapping
PERIODS = {
    "novi_vek": "Нови век",
    "rani_vek": "Рани средњи век",
    "srednji_vek": "Средњи век",
    "ostalo": "Остало"
}

# Chunk settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Initialize clients
openai_client = AsyncOpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)

# Global index and metadata
faiss_index = None
faiss_metadata = []


def load_faiss_index():
    """Load or create FAISS index"""
    global faiss_index, faiss_metadata
    
    Path(FAISS_INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
    
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_METADATA_PATH):
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        with open(FAISS_METADATA_PATH, 'rb') as f:
            faiss_metadata = pickle.load(f)
        print(f"✓ Učitana postojeća baza: {faiss_index.ntotal} vektora")
    else:
        faiss_index = faiss.IndexFlatL2(1536)
        faiss_metadata = []
        print(f"✓ Kreirana nova baza")


def save_faiss_index():
    """Save FAISS index to disk"""
    global faiss_index, faiss_metadata
    
    Path(FAISS_INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
    
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)
    with open(FAISS_METADATA_PATH, 'wb') as f:
        pickle.dump(faiss_metadata, f)
    print(f"✓ Sačuvana baza: {faiss_index.ntotal} vektora")


def is_already_indexed(filename: str) -> bool:
    """Check if file is already in the index"""
    for doc in faiss_metadata:
        if doc.get('source_file') == filename:
            return True
    return False


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from PDF file"""
    try:
        reader = PdfReader(str(pdf_path))
        text_parts = []
        
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        
        text = "\n".join(text_parts)
        
        if text.strip():
            print(f"    ✓ Ekstraktovano {len(text)} karaktera iz {len(reader.pages)} stranica")
        else:
            print(f"    ⚠️  Nema teksta (potreban OCR)")
        
        return text
    except Exception as e:
        print(f"    ✗ Greška: {e}")
        return ""


def split_text_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, 
                          overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into chunks with overlap"""
    if not text or len(text) < chunk_size:
        return [text] if text else []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        if end < len(text):
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            
            best_break = max(last_period, last_newline)
            if best_break > chunk_size - 100:
                chunk = chunk[:best_break + 1]
                end = start + best_break + 1
        
        chunks.append(chunk.strip())
        start = end - overlap
    
    return [c for c in chunks if c]


async def get_embedding(text: str) -> List[float]:
    """Get embedding for text"""
    try:
        response = await openai_client.embeddings.create(
            input=text[:8000],
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"    ✗ Greška pri kreiranju embeddings: {e}")
        return None


async def add_chunk_to_db(chunk: str, metadata: Dict) -> bool:
    """Add one chunk to FAISS database"""
    global faiss_index, faiss_metadata
    
    try:
        embedding = await get_embedding(chunk)
        if not embedding:
            return False
        
        vector = np.array([embedding], dtype='float32')
        faiss_index.add(vector)
        
        doc_metadata = {
            "id": str(uuid.uuid4()),
            "text": chunk,
            **metadata
        }
        faiss_metadata.append(doc_metadata)
        
        return True
    except Exception as e:
        print(f"    ✗ Greška pri dodavanju u bazu: {e}")
        return False


async def process_pdf_file(pdf_path: Path, period: str, force: bool = False) -> int:
    """Process one PDF file and add to FAISS"""
    filename = pdf_path.name
    
    print(f"\n📖 {filename}")
    print(f"   Period: {PERIODS.get(period, period)}")
    
    # Check if already indexed
    if not force and is_already_indexed(filename):
        print(f"    ⚠️  Fajl već indeksiran! Koristite --force da ponovo dodate.")
        return 0
    
    # Extract text
    text = extract_text_from_pdf(pdf_path)
    
    if not text or len(text) < 50:
        print(f"    ⚠️  Premaло teksta, preskačem")
        return 0
    
    # Split into chunks
    chunks = split_text_into_chunks(text)
    print(f"    📄 Kreirano {len(chunks)} chunkova")
    
    # Base metadata
    base_metadata = {
        "period": period,
        "period_name": PERIODS.get(period, period),
        "source_file": filename,
        "source_path": str(pdf_path.relative_to(Path.cwd()) if pdf_path.is_relative_to(Path.cwd()) else pdf_path)
    }
    
    # Add chunks to database
    successful = 0
    print(f"    💾 Dodavanje u bazu...")
    
    for i, chunk in enumerate(chunks, 1):
        metadata = {
            **base_metadata,
            "chunk_index": i,
            "total_chunks": len(chunks)
        }
        
        if await add_chunk_to_db(chunk, metadata):
            successful += 1
        
        await asyncio.sleep(0.3)
        
        if i % 5 == 0:
            print(f"    ⏳ {i}/{len(chunks)} chunkova dodato")
    
    # Save after adding this file
    save_faiss_index()
    
    print(f"    ✅ Završeno - {successful}/{len(chunks)} chunkova uspešno")
    return successful


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Dodaj PDF u FAISS bazu')
    parser.add_argument('pdf_file', help='Putanja do PDF fajla')
    parser.add_argument('--period', choices=['novi_vek', 'rani_vek', 'srednji_vek', 'ostalo'],
                       default='ostalo', help='Period dokumenta (default: ostalo)')
    parser.add_argument('--force', action='store_true', 
                       help='Dodaj ponovo čak i ako je već indeksiran')
    
    args = parser.parse_args()
    
    pdf_path = Path(args.pdf_file)
    
    if not pdf_path.exists():
        print(f"❌ Fajl ne postoji: {pdf_path}")
        return
    
    if not pdf_path.suffix.lower() == '.pdf':
        print(f"❌ Nije PDF fajl: {pdf_path}")
        return
    
    print("\n" + "="*70)
    print("🇷🇸 DODAVANJE PDF-a U FAISS BAZU")
    print("="*70)
    
    # Load existing index
    load_faiss_index()
    
    # Process the PDF
    chunks_added = await process_pdf_file(pdf_path, args.period, args.force)
    
    print("\n" + "="*70)
    print(f"✅ Završeno! Dodato {chunks_added} chunkova")
    print(f"💾 Ukupno u bazi: {faiss_index.ntotal} vektora")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
