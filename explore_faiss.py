"""
Script to explore and search FAISS index
"""
import os
import asyncio
import faiss
import pickle
import numpy as np
from pathlib import Path
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./faiss_data/serbian_history.index")
FAISS_METADATA_PATH = os.getenv("FAISS_METADATA_PATH", "./faiss_data/metadata.pkl")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# Initialize OpenAI client
openai_client = AsyncOpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)


def load_index():
    """Load FAISS index and metadata"""
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(FAISS_METADATA_PATH):
        print("❌ FAISS index not found!")
        print(f"   Expected: {FAISS_INDEX_PATH}")
        return None, None
    
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(FAISS_METADATA_PATH, 'rb') as f:
        metadata = pickle.load(f)
    
    return index, metadata


def show_index_stats(index, metadata):
    """Display statistics about the FAISS index"""
    print("\n" + "="*70)
    print("📊 FAISS INDEX STATISTICS")
    print("="*70)
    print(f"Total vectors: {index.ntotal}")
    print(f"Vector dimension: {index.d}")
    print(f"Metadata entries: {len(metadata)}")
    print(f"Index file: {FAISS_INDEX_PATH}")
    print(f"Metadata file: {FAISS_METADATA_PATH}")
    
    # Analyze metadata
    if metadata:
        periods = {}
        sources = {}
        
        for doc in metadata:
            # Count by period
            period = doc.get('period', 'Unknown')
            periods[period] = periods.get(period, 0) + 1
            
            # Count by source file
            source = doc.get('source_file', 'Unknown')
            sources[source] = sources.get(source, 0) + 1
        
        print(f"\n📂 Documents by period:")
        for period, count in sorted(periods.items()):
            period_name = doc.get('period_name', period)
            print(f"   {period_name:30} - {count:4} chunks")
        
        print(f"\n📄 Documents by source file (top 10):")
        for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"   {source[:50]:50} - {count:4} chunks")
    
    print("="*70 + "\n")


def show_sample_documents(metadata, n=5):
    """Display sample documents from the index"""
    print("\n" + "="*70)
    print(f"📄 SAMPLE DOCUMENTS (first {n})")
    print("="*70)
    
    for i, doc in enumerate(metadata[:n], 1):
        print(f"\n[{i}] ID: {doc.get('id', 'N/A')}")
        print(f"    Period: {doc.get('period_name', doc.get('period', 'N/A'))}")
        print(f"    Source: {doc.get('source_file', 'N/A')}")
        print(f"    Chunk: {doc.get('chunk_index', 'N/A')}/{doc.get('total_chunks', 'N/A')}")
        text = doc.get('text', '')
        preview = text[:200] + "..." if len(text) > 200 else text
        print(f"    Text: {preview}")
    
    print("="*70 + "\n")


async def get_embedding(text: str):
    """Get embedding for search query"""
    response = await openai_client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding


async def search_index(index, metadata, query: str, k: int = 5):
    """Search the FAISS index with a query"""
    print(f"\n🔍 Searching for: '{query}'")
    print("   Creating embedding...")
    
    # Get query embedding
    query_embedding = await get_embedding(query)
    query_vector = np.array([query_embedding], dtype='float32')
    
    # Search
    k = min(k, index.ntotal)  # Don't search for more than we have
    distances, indices = index.search(query_vector, k)
    
    print(f"\n📋 SEARCH RESULTS (Top {k}):")
    print("="*70)
    
    for i, (idx, distance) in enumerate(zip(indices[0], distances[0]), 1):
        if idx < len(metadata):
            doc = metadata[idx]
            print(f"\n[{i}] Similarity score: {1/(1+distance):.4f} (distance: {distance:.4f})")
            print(f"    Period: {doc.get('period_name', doc.get('period', 'N/A'))}")
            print(f"    Source: {doc.get('source_file', 'N/A')}")
            print(f"    Chunk: {doc.get('chunk_index', 'N/A')}/{doc.get('total_chunks', 'N/A')}")
            text = doc.get('text', '')
            # Show more text for search results
            preview = text[:400] + "..." if len(text) > 400 else text
            print(f"    Text:\n    {preview}\n")
    
    print("="*70 + "\n")


async def interactive_search(index, metadata):
    """Interactive search mode"""
    print("\n" + "="*70)
    print("🔍 INTERACTIVE SEARCH MODE")
    print("="*70)
    print("Type your search query (or 'quit' to exit)")
    print()
    
    while True:
        query = input("Search: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye! 👋")
            break
        
        if not query:
            continue
        
        await search_index(index, metadata, query, k=3)


async def main():
    """Main function"""
    print("\n" + "="*70)
    print("🇷🇸 FAISS INDEX EXPLORER")
    print("="*70)
    
    # Load index
    print("\n📂 Loading FAISS index...")
    index, metadata = load_index()
    
    if index is None:
        return
    
    print(f"✅ Loaded {index.ntotal} vectors")
    
    # Show statistics
    show_index_stats(index, metadata)
    
    # Show sample documents
    show_sample_documents(metadata, n=3)
    
    # Options menu
    while True:
        print("\nOptions:")
        print("1. Show more sample documents")
        print("2. Search by query (one-time)")
        print("3. Interactive search mode")
        print("4. Show statistics")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            n = input("How many documents? (default 5): ").strip()
            n = int(n) if n.isdigit() else 5
            show_sample_documents(metadata, n=min(n, len(metadata)))
        
        elif choice == "2":
            query = input("Enter search query: ").strip()
            if query:
                k = input("Number of results (default 5): ").strip()
                k = int(k) if k.isdigit() else 5
                await search_index(index, metadata, query, k=k)
        
        elif choice == "3":
            await interactive_search(index, metadata)
        
        elif choice == "4":
            show_index_stats(index, metadata)
        
        elif choice == "5":
            print("Goodbye! 👋")
            break
        
        else:
            print("❌ Invalid choice!")


if __name__ == "__main__":
    asyncio.run(main())
