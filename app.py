"""
Српски историчар - RAG Chat Agent
A RAG system using OpenAI and FAISS
"""
import os
import asyncio
from typing import Annotated
from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
from openai import AsyncOpenAI
import faiss
import numpy as np
import pickle
from pathlib import Path
import uuid
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__, static_folder='pictures', static_url_path='/static')
CORS(app)

# Configuration
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./faiss_data/serbian_history.index")
FAISS_METADATA_PATH = os.getenv("FAISS_METADATA_PATH", "./faiss_data/metadata.pkl")
COLLECTION_NAME = "serbian_history"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

# Initialize OpenAI client for embeddings
openai_client = AsyncOpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)

# Try to initialize FAISS index (with fallback)
FAISS_AVAILABLE = False
faiss_index = None
faiss_metadata = []

try:
    # Create data directory if it doesn't exist
    Path(FAISS_INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
    
    # Try to load existing index
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_METADATA_PATH):
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        with open(FAISS_METADATA_PATH, 'rb') as f:
            faiss_metadata = pickle.load(f)
        print(f"✓ FAISS index loaded: {faiss_index.ntotal} vectors")
        FAISS_AVAILABLE = True
    else:
        # Create new empty index (dimension 1536 for OpenAI embeddings)
        faiss_index = faiss.IndexFlatL2(1536)
        faiss_metadata = []
        print(f"✓ FAISS new index created (empty)")
        FAISS_AVAILABLE = True
except Exception as e:
    print(f"⚠️  FAISS NOT available: {e}")
    print(f"⚠️  Running in FALLBACK mode (without RAG)")
    FAISS_AVAILABLE = False


def save_faiss_index():
    """Save FAISS index and metadata to disk"""
    global faiss_index, faiss_metadata
    
    if not FAISS_AVAILABLE or faiss_index is None:
        print("Skipping FAISS save (not available)")
        return
    
    try:
        # Ensure directory exists
        Path(FAISS_INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
        
        # Save index and metadata
        faiss.write_index(faiss_index, FAISS_INDEX_PATH)
        with open(FAISS_METADATA_PATH, 'wb') as f:
            pickle.dump(faiss_metadata, f)
        print(f"✓ FAISS index saved: {faiss_index.ntotal} vectors")
    except Exception as e:
        print(f"✗ Error saving FAISS index: {e}")


async def get_embedding(text: str) -> list[float]:
    """Get embedding for text using OpenAI"""
    response = await openai_client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding


async def search_knowledge_base(query: str) -> str:
    """Search the Serbian history knowledge base for relevant information"""
    if not FAISS_AVAILABLE or faiss_index is None or faiss_index.ntotal == 0:
        return ""  # Return empty context in fallback mode
    
    try:
        # Get query embedding
        query_embedding = await get_embedding(query)
        query_vector = np.array([query_embedding], dtype='float32')
        
        # Search in FAISS (k=3 for top 3 results)
        k = min(3, faiss_index.ntotal)  # Don't search for more items than we have
        distances, indices = faiss_index.search(query_vector, k)
        
        if len(indices[0]) == 0:
            return "Нажалост, нисам пронашао релевантне информације у бази знања. База података можда још није попуњена."
        
        # Format results
        context = "Релевантне информације из базе знања:\n\n"
        for idx, i in enumerate(indices[0], 1):
            if i < len(faiss_metadata):
                context += f"{idx}. {faiss_metadata[i].get('text', '')}\n\n"
        
        return context
    except Exception as e:
        return f"Грешка при претраживању базе знања: {str(e)}"


async def create_agent_response(user_message: str) -> str:
    """Create response using OpenAI with RAG"""
    try:
        # Search knowledge base for relevant information (if available)
        context = await search_knowledge_base(user_message)
        
        # Create system message with context
        if FAISS_AVAILABLE and context:
            # Full RAG mode
            system_message = "Ти си виртуелни српски историчар. Одговарај на питања на основу докумената из базе знања. Буди прецизан, научан и неутралан."
            user_content = f"Контекст из базе знања:\n{context}\n\nКорисниково питање: {user_message}"
        else:
            # Fallback mode without RAG
            system_message = (
                "Ти си виртуелни српски историчар и научни асистент.\n\n"
                "⚠️ НАПОМЕНА: Тренутно радиш у режиму без приступа бази докумената (RAG систем није доступан).\n"
                "Одговарај на питања користећи своје опште знање о српској историји, али буди транспарентан о томе.\n\n"
                "Правила:\n"
                "1. Одговарај научно и прецизно\n"
                "2. Ако ниси сигуран, признај то\n"
                "3. На почетку одговора напомени да је база докумената тренутно недоступна\n"
                "4. Тон је научан, неутралан и поуздан\n"
            )
            user_content = user_message
        
        # Create messages for the chat
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content}
        ]
        
        # Get response from OpenAI
        response = await openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.01,
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Извините, дошло је до грешке: {str(e)}"


@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.json
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Run async function in event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response = loop.run_until_complete(create_agent_response(user_message))
        loop.close()
        
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'faiss_available': FAISS_AVAILABLE,
        'faiss_index_path': FAISS_INDEX_PATH,
        'vectors_count': faiss_index.ntotal if faiss_index else 0,
        'collection': COLLECTION_NAME,
        'mode': 'RAG' if FAISS_AVAILABLE else 'FALLBACK (no RAG)'
    })


if __name__ == '__main__':
    print("="*70)
    print("🇷🇸 СРПСКИ ИСТОРИЧАР - RAG Chat System")
    print("="*70)
    
    # Show status
    print(f"\nFAISS Index: {FAISS_INDEX_PATH}")
    print(f"Metadata: {FAISS_METADATA_PATH}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Vectors: {faiss_index.ntotal if faiss_index else 0}")
    print(f"Mode: {'✓ RAG (with documents)' if FAISS_AVAILABLE and faiss_index and faiss_index.ntotal > 0 else '⚠️  FALLBACK (without documents)'}")
    
    if not FAISS_AVAILABLE or not faiss_index or faiss_index.ntotal == 0:
        print("\n" + "="*70)
        print("⚠️  WARNING: Running without FAISS / RAG functionality")
        print("The chatbot will work but without access to document database.")
        print("To enable full RAG mode, populate the database first.")
        print("Run: python populate_database.py")
        print("="*70)
    
    print("\nStarting Flask server...\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
