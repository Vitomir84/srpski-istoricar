"""
Dodaj tekstualne dokumente iz docs/tekstualni_fajlovi u FAISS bazu
"""
import os
import asyncio
from pathlib import Path
from typing import List, Dict
import argparse
import json

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

# Folder sa tekstualnim dokumentima
TEXT_DOCS_FOLDER = Path("docs/tekstualni_fajlovi")
BIBLIOGRAPHY_FILE = Path("bibliografija.json")

# Chunk settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Initialize OpenAI client
openai_client = AsyncOpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)

# Global index and metadata
faiss_index = None
faiss_metadata = []
bibliography = {}


def load_bibliography():
    """Load bibliography data from JSON file"""
    global bibliography
    
    if BIBLIOGRAPHY_FILE.exists():
        try:
            with open(BIBLIOGRAPHY_FILE, 'r', encoding='utf-8') as f:
                bibliography = json.load(f)
            print(f"✓ Učitana bibliografija: {len(bibliography)} unosa")
        except Exception as e:
            print(f"⚠️  Greška pri učitavanju bibliografije: {e}")
            bibliography = {}
    else:
        print(f"⚠️  Bibliografija nije pronađena: {BIBLIOGRAPHY_FILE}")
        bibliography = {}


def format_citation(filename: str) -> str:
    """Format citation in the required format: Autor (godina). Naslov. Izdavač. strane"""
    if filename not in bibliography:
        return filename
    
    bib = bibliography[filename]
    parts = []
    
    # Autor
    if bib.get('autor'):
        parts.append(bib['autor'])
    
    # Godina
    if bib.get('godina'):
        if parts:
            parts[-1] += f" ({bib['godina']})"
        else:
            parts.append(f"({bib['godina']})")
    
    # Naslov
    if bib.get('naslov'):
        parts.append(bib['naslov'])
    
    # Izdavač
    if bib.get('izdavac'):
        parts.append(bib['izdavac'])
    
    # Strane
    if bib.get('strane'):
        parts.append(f"str. {bib['strane']}")
    
    return '. '.join(parts) if parts else filename


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


def load_text_file(file_path: Path) -> str:
    """Load text from file with multiple encoding attempts"""
    encodings = ['utf-8', 'utf-8-sig', 'windows-1252', 'cp1252', 'latin1']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                text = f.read()
            return text
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception as e:
            print(f"    ⚠️  Greška pri čitanju sa {encoding}: {e}")
    
    print(f"    ⚠️  Nije moguće učitati fajl ni sa jednim encoding-om")
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
        
        # Try to break at sentence/paragraph boundaries
        if end < len(text):
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            
            best_break = max(last_period, last_newline)
            if best_break > chunk_size - 200:  # Don't break too early
                chunk = chunk[:best_break + 1]
                end = start + best_break + 1
        
        chunks.append(chunk.strip())
        start = end - overlap
    
    return [c for c in chunks if c]


async def get_embedding(text: str) -> List[float]:
    """Get embedding for text"""
    try:
        # Limit text to 8000 characters for embedding
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


async def process_text_file(file_path: Path, force: bool = False) -> int:
    """Process one text file and add to FAISS"""
    filename = file_path.name
    
    print(f"\n📄 {filename}")
    
    # Check if already indexed
    if not force and is_already_indexed(filename):
        print(f"    ⚠️  Fajl već indeksiran! Koristite --force da ponovo dodate.")
        return 0
    
    # Load text
    print(f"    📖 Učitavanje...")
    text = load_text_file(file_path)
    
    if not text or len(text) < 50:
        print(f"    ⚠️  Premalo teksta ({len(text)} karaktera), preskačem")
        return 0
    
    print(f"    ✓ Učitano {len(text)} karaktera")
    
    # Split into chunks
    chunks = split_text_into_chunks(text)
    print(f"    📄 Kreirano {len(chunks)} chunkova")
    
    # Get bibliographic data
    citation = format_citation(filename)
    bib_data = bibliography.get(filename, {})
    
    print(f"    📚 Referenca: {citation}")
    
    # Base metadata
    base_metadata = {
        "source_file": filename,
        "source_path": str(file_path.relative_to(Path.cwd()) if file_path.is_relative_to(Path.cwd()) else file_path),
        "source_type": "text",
        "citation": citation,
        "autor": bib_data.get('autor', ''),
        "godina": bib_data.get('godina', ''),
        "naslov": bib_data.get('naslov', ''),
        "izdavac": bib_data.get('izdavac', ''),
        "strane": bib_data.get('strane', '')
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
        else:
            print(f"    ⚠️  Neuspelo dodavanje chunka {i}")
        
        # Small delay to avoid rate limiting
        await asyncio.sleep(0.3)
        
        if i % 10 == 0:
            print(f"    ⏳ {i}/{len(chunks)} chunkova dodato")
    
    print(f"    ✅ Završeno - {successful}/{len(chunks)} chunkova uspešno")
    return successful


async def process_all_text_files(force: bool = False):
    """Process all text files in the folder"""
    if not TEXT_DOCS_FOLDER.exists():
        print(f"❌ Folder ne postoji: {TEXT_DOCS_FOLDER}")
        return
    
    # Find all .txt files
    txt_files = list(TEXT_DOCS_FOLDER.glob("*.txt"))
    
    if not txt_files:
        print(f"⚠️  Nema .txt fajlova u folderu {TEXT_DOCS_FOLDER}")
        return
    
    print(f"\n📁 Pronađeno {len(txt_files)} tekstualnih fajlova")
    
    total_chunks = 0
    processed_files = 0
    skipped_files = 0
    
    for file_path in txt_files:
        chunks_added = await process_text_file(file_path, force)
        
        if chunks_added > 0:
            total_chunks += chunks_added
            processed_files += 1
            # Save after each file
            save_faiss_index()
        else:
            skipped_files += 1
    
    print("\n" + "="*70)
    print(f"✅ ZAVRŠENO!")
    print(f"   📝 Obrađeno fajlova: {processed_files}")
    print(f"   ⏭️  Preskočeno fajlova: {skipped_files}")
    print(f"   📦 Dodato chunkova: {total_chunks}")
    print(f"   💾 Ukupno u bazi: {faiss_index.ntotal} vektora")
    print("="*70 + "\n")


async def process_single_file(file_path: Path, force: bool = False):
    """Process a single text file"""
    if not file_path.exists():
        print(f"❌ Fajl ne postoji: {file_path}")
        return
    
    if file_path.suffix.lower() != '.txt':
        print(f"❌ Nije .txt fajl: {file_path}")
        return
    
    chunks_added = await process_text_file(file_path, force)
    
    if chunks_added > 0:
        save_faiss_index()
    
    print("\n" + "="*70)
    print(f"✅ Završeno! Dodato {chunks_added} chunkova")
    print(f"💾 Ukupno u bazi: {faiss_index.ntotal} vektora")
    print("="*70 + "\n")


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Dodaj tekstualne dokumente u FAISS bazu',
        epilog='Primer: python add_txt_to_faiss.py --all'
    )
    parser.add_argument('--all', action='store_true',
                       help='Procesuj sve .txt fajlove iz docs/tekstualni_fajlovi/')
    parser.add_argument('--file', type=str,
                       help='Procesuj samo jedan fajl (putanja do fajla)')
    parser.add_argument('--force', action='store_true',
                       help='Dodaj ponovo čak i ako je već indeksiran')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🇷🇸 DODAVANJE TEKSTUALNIH DOKUMENATA U FAISS BAZU")
    print("="*70)
    
    # Load bibliography
    load_bibliography()
    
    # Load existing index
    load_faiss_index()
    
    if args.all:
        await process_all_text_files(args.force)
    elif args.file:
        file_path = Path(args.file)
        await process_single_file(file_path, args.force)
    else:
        print("\n❌ Morate navesti --all ili --file")
        print("   Primer 1: python add_txt_to_faiss.py --all")
        print("   Primer 2: python add_txt_to_faiss.py --file docs/tekstualni_fajlovi/primer.txt")
        print("\nZa pomoć: python add_txt_to_faiss.py --help\n")


if __name__ == "__main__":
    asyncio.run(main())
