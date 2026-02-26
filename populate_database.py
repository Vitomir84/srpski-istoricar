"""
Script to populate FAISS database with Serbian history documents
"""
import os
import asyncio
from typing import List
from openai import AsyncOpenAI
import faiss
import numpy as np
import pickle
from pathlib import Path
import uuid
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./faiss_data/serbian_history.index")
FAISS_METADATA_PATH = os.getenv("FAISS_METADATA_PATH", "./faiss_data/metadata.pkl")
COLLECTION_NAME = "serbian_history"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

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
    
    # Create directory if needed
    Path(FAISS_INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
    
    # Try to load existing index
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_METADATA_PATH):
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        with open(FAISS_METADATA_PATH, 'rb') as f:
            faiss_metadata = pickle.load(f)
        print(f"✓ Loaded existing index: {faiss_index.ntotal} vectors")
    else:
        # Create new index
        faiss_index = faiss.IndexFlatL2(1536)  # OpenAI embedding dimension
        faiss_metadata = []
        print(f"✓ Created new empty index")


def save_faiss_index():
    """Save FAISS index to disk"""
    global faiss_index, faiss_metadata
    
    # Ensure directory exists
    Path(FAISS_INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
    
    # Save index and metadata
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)
    with open(FAISS_METADATA_PATH, 'wb') as f:
        pickle.dump(faiss_metadata, f)
    print(f"✓ Saved index: {faiss_index.ntotal} vectors")


async def get_embedding(text: str) -> List[float]:
    """Get embedding for text using OpenAI"""
    response = await openai_client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding


async def add_document(text: str, metadata: dict = None) -> str:
    """Add a single document to FAISS"""
    global faiss_index, faiss_metadata
    
    if not text.strip():
        print("Skipping empty document")
        return None
    
    try:
        # Get embedding
        embedding = await get_embedding(text)
        
        # Convert to numpy array
        vector = np.array([embedding], dtype='float32')
        
        # Add to FAISS index
        faiss_index.add(vector)
        
        # Add metadata
        doc_metadata = {
            "id": str(uuid.uuid4()),
            "text": text,
            **(metadata or {})
        }
        faiss_metadata.append(doc_metadata)
        
        print(f"✓ Added document: {text[:100]}...")
        return doc_metadata["id"]
    except Exception as e:
        print(f"✗ Error adding document: {e}")
        return None


async def add_documents_from_text(text: str, chunk_size: int = 500):
    """Split text into chunks and add to database"""
    # Simple chunking by paragraphs
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    print(f"Found {len(paragraphs)} paragraphs to process")
    
    for i, paragraph in enumerate(paragraphs, 1):
        print(f"\nProcessing paragraph {i}/{len(paragraphs)}")
        await add_document(paragraph)
        
        # Small delay to avoid rate limits
        await asyncio.sleep(0.5)


async def add_sample_documents():
    """Add sample documents about Serbian history"""
    
    sample_documents = [
        {
            "text": "Стефан Немања (око 1113-1199) био је велики жупан Србије од 1166. до 1196. године. Он је оснивач dinastије Немањић која је владала Србијом више од два века. Немања је уједнио српске земље и створио јаку и независну средњовековну државу.",
            "metadata": {"category": "владари", "period": "средњи век", "person": "Стефан Немања"}
        },
        {
            "text": "Косовска битка вођена је 15. јуна 1389. године (по јулијанском календару, што одговара 28. јуну по грегоријанском календару) између српске војске кнеза Лазара Хребељановића и османске армије султана Мурата I. Битка се водила на Косову пољу.",
            "metadata": {"category": "битке", "period": "средњи век", "year": "1389"}
        },
        {
            "text": "Свети Сава, световно име Растко Немањић (око 1174-1236), био је први српски архиепископ и један од најважнијих личности српске историје. Син великог жупана Стефана Немање, Растко је напустио световни живот и постао монах.",
            "metadata": {"category": "црква", "period": "средњи век", "person": "Свети Сава"}
        },
        {
            "text": "Душанов законик је средњовековни српски правни кодекс који је донео цар Стефан Душан 1349. године на сабору у Скопљу. Законик представља врхунац српске средњовековне законодавне мисли и садржи 201 члан.",
            "metadata": {"category": "законодавство", "period": "средњи век", "year": "1349"}
        },
        {
            "text": "Први српски устанак је национално-ослободилачки устанак Срба против Османског царства који је трајао од 1804. до 1813. године. Предводио га је Ђорђе Петровић, познат као Карађорђе. Устанак је означио почетак ослобађања Србије од турске власти.",
            "metadata": {"category": "устанци", "period": "нови век", "year": "1804"}
        },
        {
            "text": "Други српски устанак избио је 1815. године под вођством Милоша Обреновића. За разлику од Првог српског устанка, овај устанак је био мање ратнички и више дипломатски оријентисан. Резултирао је признавањем аутономије Србије.",
            "metadata": {"category": "устанци", "period": "нови век", "year": "1815"}
        },
    ]
    
    print("Adding sample documents to FAISS database...\n")
    
    for i, doc in enumerate(sample_documents, 1):
        print(f"\n[{i}/{len(sample_documents)}]")
        await add_document(doc["text"], doc.get("metadata"))
        await asyncio.sleep(0.5)  # Rate limit protection
    
    # Save after adding documents
    save_faiss_index()
    
    print("\n" + "="*60)
    print("✓ Successfully added all sample documents!")
    print("="*60)


async def load_from_file(file_path: str):
    """Load documents from a text file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"Loading documents from: {file_path}")
        await add_documents_from_text(content)
        
        # Save after loading from file
        save_faiss_index()
        
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error loading file: {e}")


async def check_database_status():
    """Check current database status"""
    global faiss_index, faiss_metadata
    
    try:
        vectors_count = faiss_index.ntotal if faiss_index else 0
        
        print("\n" + "="*60)
        print(f"Database Status:")
        print(f"  Collection: {COLLECTION_NAME}")
        print(f"  Total documents: {vectors_count}")
        print(f"  Index file: {FAISS_INDEX_PATH}")
        print(f"  Metadata file: {FAISS_METADATA_PATH}")
        print("="*60 + "\n")
        
        return vectors_count
    except Exception as e:
        print(f"Error checking database: {e}")
        return 0


async def main():
    """Main function"""
    print("="*60)
    print("Српски историчар - Database Population Tool")
    print("="*60 + "\n")
    
    # Load existing index
    load_faiss_index()
    
    # Check current status
    await check_database_status()
    
    print("Options:")
    print("1. Add sample documents (6 documents about Serbian history)")
    print("2. Load documents from srpski-istoricar.rs.txt")
    print("3. Check database status only")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        await add_sample_documents()
        await check_database_status()
    elif choice == "2":
        file_path = "srpski-istoricar.rs.txt"
        if os.path.exists(file_path):
            await load_from_file(file_path)
            await check_database_status()
        else:
            print(f"\nFile not found: {file_path}")
            print("Please make sure the file exists in the current directory.")
    elif choice == "3":
        await check_database_status()
    elif choice == "4":
        print("Exiting...")
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    asyncio.run(main())
