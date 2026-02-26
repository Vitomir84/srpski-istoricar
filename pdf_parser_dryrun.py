"""
PDF Parser - DRY RUN verzija
Ekstraktuje tekst iz PDF-ova i čuva u JSON fajlove (bez FAISS baze).
Kasnije možeš uploadovati te fajlove u FAISS.
"""
import os
import json
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import time

# PDF Processing
from pypdf import PdfReader

# Directories to scan
DOCS_DIR = Path("docs")
OUTPUT_DIR = Path("extracted_documents")
PERIODS = {
    "novi_vek": "Нови век",
    "rani_vek": "Рани средњи век",
    "srednji_vek": "Средњи век",
    "ostalo": "Остало"
}

# Chunk size for text splitting
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def ensure_output_dir():
    """Kreiraj output direktorijum"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    for period in PERIODS.keys():
        (OUTPUT_DIR / period).mkdir(exist_ok=True)
    print(f"✓ Output folder: {OUTPUT_DIR}")


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Ekstraktuje tekst iz PDF fajla"""
    text = ""
    
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
            print(f"    ⚠️  Nema teksta (verovatno skeniran dokument - potreban OCR)")
            
    except Exception as e:
        print(f"    ✗ Greška: {e}")
    
    return text


def split_text_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, 
                          overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Deli tekst na chunke sa preklapanjem"""
    if not text or len(text) < chunk_size:
        return [text] if text else []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Pokušaj završiti na kraju rečenice
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


def process_pdf_file(pdf_path: Path, period: str) -> Dict:
    """Procesuj jedan PDF fajl i sačuvaj kao JSON"""
    print(f"\n📖 {pdf_path.name}")
    
    # Ekstraktuj tekst
    text = extract_text_from_pdf(pdf_path)
    
    if not text or len(text) < 50:
        print(f"    ⚠️  Presedak tekst, preskačem")
        return None
    
    # Podeli na chunke
    chunks = split_text_into_chunks(text)
    print(f"    📄 {len(chunks)} chunkova")
    
    # Pripremi dokument
    document = {
        "source_file": pdf_path.name,
        "source_path": str(pdf_path.relative_to(DOCS_DIR)),
        "period": period,
        "period_name": PERIODS[period],
        "total_chars": len(text),
        "total_chunks": len(chunks),
        "chunks": []
    }
    
    # Dodaj chunke
    for i, chunk in enumerate(chunks, 1):
        document["chunks"].append({
            "chunk_index": i,
            "text": chunk,
            "char_count": len(chunk)
        })
    
    # Sačuvaj kao JSON
    output_file = OUTPUT_DIR / period / f"{pdf_path.stem}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(document, f, ensure_ascii=False, indent=2)
    
    print(f"    ✅ Sačuvano: {output_file.name}")
    
    return {
        "file": pdf_path.name,
        "chunks": len(chunks),
        "chars": len(text)
    }


def process_directory(period: str) -> Dict:
    """Procesuj sve PDF fajlove iz jednog perioda"""
    period_dir = DOCS_DIR / period
    
    if not period_dir.exists():
        return {"processed": 0, "chunks": 0, "failed": 0}
    
    pdf_files = list(period_dir.glob("*.pdf"))
    
    if not pdf_files:
        return {"processed": 0, "chunks": 0, "failed": 0}
    
    print(f"\n{'='*70}")
    print(f"📁 {PERIODS[period]} - {len(pdf_files)} PDF fajlova")
    print(f"{'='*70}")
    
    stats = {
        "processed": 0, 
        "chunks": 0, 
        "failed": 0,
        "total_chars": 0,
        "files": []
    }
    
    for pdf_file in pdf_files:
        try:
            result = process_pdf_file(pdf_file, period)
            if result:
                stats["processed"] += 1
                stats["chunks"] += result["chunks"]
                stats["total_chars"] += result["chars"]
                stats["files"].append(result)
            else:
                stats["failed"] += 1
        except Exception as e:
            print(f"    ✗ Greška: {e}")
            stats["failed"] += 1
    
    return stats


def create_upload_script(all_stats: Dict):
    """Kreiraj skriptu za upload u Qdrant kada bude spreman"""
    script_content = '''"""
Upload ekstraktovanih dokumenata u Qdrant
Pokreni ovu skriptu kada Qdrant server bude aktivan.
"""
import os
import json
import asyncio
from pathlib import Path
from openai import AsyncOpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
import uuid
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "serbian_history"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

qdrant_client = QdrantClient(url=QDRANT_URL)
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)


def ensure_collection():
    """Kreiraj kolekciju ako ne postoji"""
    try:
        qdrant_client.get_collection(COLLECTION_NAME)
        print(f"✓ Kolekcija '{COLLECTION_NAME}' postoji")
    except:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )
        print(f"✓ Kolekcija '{COLLECTION_NAME}' kreirana")


async def get_embedding(text: str):
    """Kreiraj embedding preko OpenAI"""
    response = await openai_client.embeddings.create(
        input=text[:8000],
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding


async def upload_document(json_path: Path):
    """Upload jednog dokumenta iz JSON fajla"""
    with open(json_path, 'r', encoding='utf-8') as f:
        doc = json.load(f)
    
    uploaded = 0
    for chunk_data in doc["chunks"]:
        try:
            # Kreiraj embedding
            embedding = await get_embedding(chunk_data["text"])
            
            # Metadata
            metadata = {
                "text": chunk_data["text"],
                "period": doc["period"],
                "period_name": doc["period_name"],
                "source_file": doc["source_file"],
                "source_path": doc["source_path"],
                "chunk_index": chunk_data["chunk_index"],
                "total_chunks": doc["total_chunks"]
            }
            
            # Upload
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload=metadata
            )
            
            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=[point]
            )
            
            uploaded += 1
            await asyncio.sleep(0.3)  # Rate limiting
            
        except Exception as e:
            print(f"  ✗ Greška pri uploadu chunka {chunk_data['chunk_index']}: {e}")
    
    return uploaded


async def main():
    """Upload svih dokumenata"""
    print("="*70)
    print("Upload ekstraktovanih dokumenata u Qdrant")
    print("="*70 + "\\n")
    
    # Proveri Qdrant
    try:
        ensure_collection()
    except Exception as e:
        print(f"✗ Qdrant nije dostupan: {e}")
        print("  Pokreni Qdrant prvo!")
        return
    
    # Nađi sve JSON fajlove
    json_files = list(Path("extracted_documents").rglob("*.json"))
    print(f"\\nPronađeno {len(json_files)} dokumenata\\n")
    
    total_chunks = 0
    for json_file in tqdm(json_files, desc="Upload dokumenata"):
        uploaded = await upload_document(json_file)
        total_chunks += uploaded
    
    print(f"\\n✅ Upload završen - {total_chunks} chunkova u bazi")
    
    # Proveri status
    info = qdrant_client.get_collection(COLLECTION_NAME)
    print(f"💾 Ukupno u bazi: {info.points_count} dokumenata")


if __name__ == "__main__":
    asyncio.run(main())
'''
    
    with open("upload_to_qdrant.py", 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print(f"\n✓ Kreirana skripta: upload_to_qdrant.py")


def main():
    """Glavna funkcija"""
    print("\n" + "="*70)
    print("🇷🇸 PDF OCR Parser - DRY RUN (bez FAISS-a)")
    print("="*70)
    print("Ekstraktuje tekst iz PDF-ova i čuva u JSON fajlove")
    print("="*70 + "\n")
    
    ensure_output_dir()
    
    # Statistike
    all_stats = {
        "processed": 0,
        "chunks": 0,
        "failed": 0,
        "total_chars": 0,
        "by_period": {}
    }
    
    start_time = time.time()
    
    # Procesuj svaki period
    for period in PERIODS.keys():
        stats = process_directory(period)
        all_stats["processed"] += stats["processed"]
        all_stats["chunks"] += stats["chunks"]
        all_stats["failed"] += stats["failed"]
        all_stats["total_chars"] += stats.get("total_chars", 0)
        all_stats["by_period"][period] = stats
    
    elapsed = time.time() - start_time
    
    # Finalni izveštaj
    print("\n" + "="*70)
    print("📊 FINALNI IZVEŠTAJ")
    print("="*70)
    print(f"✅ Procesovano fajlova:   {all_stats['processed']}")
    print(f"📝 Kreirano chunkova:     {all_stats['chunks']}")
    print(f"📄 Ukupno karaktera:      {all_stats['total_chars']:,}")
    print(f"❌ Neuspešno:             {all_stats['failed']}")
    print(f"⏱️  Vreme:                 {elapsed:.1f} sekundi")
    
    print(f"\n📂 Po periodima:")
    for period, stats in all_stats["by_period"].items():
        if stats["processed"] > 0:
            print(f"   {PERIODS[period]:25} - {stats['processed']:2} fajlova, {stats['chunks']:4} chunkova")
    
    print(f"\n💾 JSON fajlovi sačuvani u: {OUTPUT_DIR}/")
    
    # Kreiraj upload skriptu
    create_upload_script(all_stats)
    
    print("\n" + "="*70)
    print("SLEDEĆI KORACI:")
    print("="*70)
    print("1. Pokreni: poetry run python upload_to_faiss.py")
    print("2. To će uploadovati sve ekstraktovane dokumente u FAISS")
    print("3. FAISS baza je potpuno lokalna - nema potrebe za serverom")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
