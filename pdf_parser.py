"""
OCR Parser za PDF dokumente o srpskoj istoriji
Parsuje PDF fajlove iz foldera (novi_vek, rani_vek, srednji_vek, ostalo) 
i dodaje ih u Qdrant vektorsku bazu sa metadata o vremenskoj periodizaciji.
"""
import os
import asyncio
from pathlib import Path
from typing import List, Dict, Optional
import re
from tqdm import tqdm

# PDF Processing
from pypdf import PdfReader
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

# Database and Embeddings
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from openai import AsyncOpenAI
import uuid
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "serbian_history"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# Directories to scan
DOCS_DIR = Path("docs")
PERIODS = {
    "novi_vek": "Нови век",
    "rani_vek": "Рани средњи век", 
    "srednji_vek": "Средњи век",
    "ostalo": "Остало"
}

# Chunk size for text splitting (characters)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Initialize clients
qdrant_client = QdrantClient(url=QDRANT_URL)
openai_client = AsyncOpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)


def ensure_collection_exists():
    """Create collection if it doesn't exist"""
    try:
        qdrant_client.get_collection(COLLECTION_NAME)
        print(f"✓ Kolekcija '{COLLECTION_NAME}' postoji")
    except Exception:
        print(f"Kreiram novu kolekciju '{COLLECTION_NAME}'...")
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )
        print("✓ Kolekcija kreirana")


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Ekstraktuje tekst iz PDF fajla.
    Prvo pokušava direktnu ekstrakciju, zatim OCR ako nema teksta.
    """
    text = ""
    
    try:
        # Pokušaj direktnu ekstrakciju teksta
        reader = PdfReader(str(pdf_path))
        text_parts = []
        
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        
        text = "\n".join(text_parts)
        
        # Proveri da li ima dovoljno teksta
        if len(text.strip()) < 100:
            print(f"    ⚠️  Malo teksta ekstraktovano, pokušavam OCR...")
            text = extract_text_with_ocr(pdf_path)
        else:
            print(f"    ✓ Ekstraktovano {len(text)} karaktera")
            
    except Exception as e:
        print(f"    ✗ Greška pri ekstrakciji: {e}")
        print(f"    Pokušavam OCR...")
        text = extract_text_with_ocr(pdf_path)
    
    return text


def extract_text_with_ocr(pdf_path: Path, max_pages: int = 50) -> str:
    """
    Koristi OCR (pytesseract) za ekstrakciju teksta iz PDF-a.
    Konvertuje PDF u slike i radi OCR na svakoj stranici.
    """
    try:
        # Konvertuj PDF u slike (ograniči na max_pages da ne traje predugo)
        print(f"    🔍 OCR u toku...")
        images = convert_from_path(str(pdf_path), first_page=1, last_page=max_pages)
        
        text_parts = []
        for i, image in enumerate(images, 1):
            # Pytesseract za OCR - koristi srpski jezik
            # Dodaj eng za bolje rezultate sa latiničnim tekstom
            page_text = pytesseract.image_to_string(image, lang='srp+eng')
            text_parts.append(page_text)
            
            if i % 10 == 0:
                print(f"    📄 Procesovano {i}/{len(images)} stranica")
        
        text = "\n".join(text_parts)
        print(f"    ✓ OCR završen - {len(text)} karaktera")
        return text
        
    except Exception as e:
        print(f"    ✗ OCR greška: {e}")
        return ""


def split_text_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, 
                          overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Deli tekst na chunke sa preklapanjem za bolju kontekstualizaciju.
    """
    if not text or len(text) < chunk_size:
        return [text] if text else []
    
    chunks = []
    start = 0
    
    while start < len(text):
        # Uzmi chunk
        end = start + chunk_size
        chunk = text[start:end]
        
        # Pokušaj da završiš na kraju rečenice ili paragrafa
        if end < len(text):
            # Traži kraj rečenice u poslednjih 100 karaktera chunka
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            
            best_break = max(last_period, last_newline)
            if best_break > chunk_size - 100:  # Ako je dovoljno blizu kraja
                chunk = chunk[:best_break + 1]
                end = start + best_break + 1
        
        chunks.append(chunk.strip())
        start = end - overlap  # Preklapanje za kontinuitet
    
    return [c for c in chunks if c]  # Filtriraj prazne chunke


async def get_embedding(text: str) -> List[float]:
    """Dobavi embedding za tekst koristeći OpenAI"""
    try:
        response = await openai_client.embeddings.create(
            input=text[:8000],  # Limit na 8000 karaktera za embedding
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"    ✗ Greška pri kreiranju embeddings: {e}")
        return None


async def add_chunk_to_db(chunk: str, metadata: Dict) -> bool:
    """Dodaj jedan chunk u Qdrant bazu"""
    try:
        # Dobavi embedding
        embedding = await get_embedding(chunk)
        if not embedding:
            return False
        
        # Kreiraj point
        point_id = str(uuid.uuid4())
        point = PointStruct(
            id=point_id,
            vector=embedding,
            payload={
                "text": chunk,
                **metadata
            }
        )
        
        # Upload u Qdrant
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[point]
        )
        
        return True
        
    except Exception as e:
        print(f"    ✗ Greška pri dodavanju u bazu: {e}")
        return False


async def process_pdf_file(pdf_path: Path, period: str) -> int:
    """
    Procesuj jedan PDF fajl: ekstraktuj tekst, podeli na chunke, dodaj u bazu.
    Vraća broj uspešno dodatih chunkova.
    """
    print(f"\n📖 Procesovanje: {pdf_path.name}")
    print(f"   Period: {PERIODS[period]}")
    
    # Ekstraktuj tekst
    text = extract_text_from_pdf(pdf_path)
    
    if not text or len(text) < 50:
        print(f"    ⚠️  Presedak tekst, preskačem fajl")
        return 0
    
    # Podeli na chunke
    chunks = split_text_into_chunks(text)
    print(f"    📄 Kreirano {len(chunks)} chunkova")
    
    # Metadata za sve chunke iz ovog dokumenta
    base_metadata = {
        "period": period,
        "period_name": PERIODS[period],
        "source_file": pdf_path.name,
        "source_path": str(pdf_path.relative_to(DOCS_DIR))
    }
    
    # Dodaj chunke u bazu
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
        
        # Rate limiting
        await asyncio.sleep(0.3)
        
        # Progress indicator
        if i % 5 == 0:
            print(f"    ⏳ {i}/{len(chunks)} chunkova dodato")
    
    print(f"    ✅ Završeno - {successful}/{len(chunks)} chunkova uspešno")
    return successful


async def process_directory(period: str) -> Dict:
    """Procesuj sve PDF fajlove iz jednog perioda"""
    period_dir = DOCS_DIR / period
    
    if not period_dir.exists():
        print(f"⚠️  Folder ne postoji: {period_dir}")
        return {"processed": 0, "chunks": 0, "failed": 0}
    
    # Pronađi sve PDF fajlove
    pdf_files = list(period_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"ℹ️  Nema PDF fajlova u: {period_dir}")
        return {"processed": 0, "chunks": 0, "failed": 0}
    
    print(f"\n{'='*70}")
    print(f"📁 Period: {PERIODS[period]}")
    print(f"📄 Pronađeno {len(pdf_files)} PDF fajlova")
    print(f"{'='*70}")
    
    stats = {"processed": 0, "chunks": 0, "failed": 0}
    
    for pdf_file in pdf_files:
        try:
            chunks_added = await process_pdf_file(pdf_file, period)
            if chunks_added > 0:
                stats["processed"] += 1
                stats["chunks"] += chunks_added
            else:
                stats["failed"] += 1
        except Exception as e:
            print(f"✗ Greška pri procesovanju {pdf_file.name}: {e}")
            stats["failed"] += 1
    
    return stats


async def process_all_documents():
    """Glavna funkcija - procesuje sve PDF fajlove iz svih perioda"""
    print("\n" + "="*70)
    print("🇷🇸 SRPSKI ISTORIČAR - PDF OCR Parser")
    print("="*70)
    
    # Proveri da li postoji kolekcija
    ensure_collection_exists()
    
    # Statistike
    total_stats = {
        "processed": 0,
        "chunks": 0,
        "failed": 0,
        "by_period": {}
    }
    
    # Procesuj svaki period
    for period in PERIODS.keys():
        stats = await process_directory(period)
        total_stats["processed"] += stats["processed"]
        total_stats["chunks"] += stats["chunks"]
        total_stats["failed"] += stats["failed"]
        total_stats["by_period"][period] = stats
    
    # Finalni izveštaj
    print("\n" + "="*70)
    print("📊 FINALNI IZVEŠTAJ")
    print("="*70)
    print(f"✅ Ukupno procesovanih fajlova:  {total_stats['processed']}")
    print(f"📝 Ukupno kreiranih chunkova:   {total_stats['chunks']}")
    print(f"❌ Neuspešnih fajlova:          {total_stats['failed']}")
    
    print(f"\n📂 Po periodima:")
    for period, stats in total_stats["by_period"].items():
        print(f"   {PERIODS[period]:25} - {stats['processed']} fajlova, {stats['chunks']} chunkova")
    
    # Proveri bazu
    try:
        collection_info = qdrant_client.get_collection(COLLECTION_NAME)
        print(f"\n💾 Ukupno dokumenata u bazi:    {collection_info.points_count}")
    except Exception as e:
        print(f"⚠️  Ne mogu proveriti stanje baze: {e}")
    
    print("="*70)


async def main():
    """Main entry point"""
    print("\nOCR Parser za srpsku istoriju")
    print("Napomena: Za OCR potreban je instaliran Tesseract OCR")
    print("Download: https://github.com/UB-Mannheim/tesseract/wiki\n")
    
    try:
        await process_all_documents()
    except KeyboardInterrupt:
        print("\n\n⚠️  Prekinuto od strane korisnika")
    except Exception as e:
        print(f"\n❌ Kritična greška: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
