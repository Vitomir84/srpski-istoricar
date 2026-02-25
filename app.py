"""
Српски историчар - RAG Chat Agent
A RAG system using OpenAI and Qdrant
"""
import os
import asyncio
from typing import Annotated
from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
from openai import AsyncOpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL", ":memory:")  # Use in-memory by default
QDRANT_PATH = os.getenv("QDRANT_PATH", "./qdrant_data")  # Or persistent path  
COLLECTION_NAME = "serbian_history"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

# Initialize OpenAI client for embeddings
openai_client = AsyncOpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)

# Try to initialize Qdrant client (with fallback)
QDRANT_AVAILABLE = False
qdrant_client = None

try:
    # Try in-memory or local path first
    if QDRANT_URL == ":memory:":
        qdrant_client = QdrantClient(location=":memory:")
        print(f"✓ Qdrant in-memory mode initialized")
        QDRANT_AVAILABLE = True
    elif QDRANT_URL.startswith("http"):
        # Try server connection
        qdrant_client = QdrantClient(url=QDRANT_URL, timeout=5)
        qdrant_client.get_collections()
        print(f"✓ Qdrant connected: {QDRANT_URL}")
        QDRANT_AVAILABLE = True
    else:
        # Use local persistent storage
        qdrant_client = QdrantClient(path=QDRANT_URL)
        print(f"✓ Qdrant local storage: {QDRANT_URL}")
        QDRANT_AVAILABLE = True
except Exception as e:
    print(f"⚠️  Qdrant NOT available: {e}")
    print(f"⚠️  Running in FALLBACK mode (without RAG)")
    QDRANT_AVAILABLE = False


def initialize_qdrant_collection():
    """Initialize Qdrant collection if it doesn't exist"""
    global QDRANT_AVAILABLE
    
    if not QDRANT_AVAILABLE:
        print("Skipping Qdrant initialization (not available)")
        return
    
    try:
        collections = qdrant_client.get_collections().collections
        collection_names = [col.name for col in collections]
        
        if COLLECTION_NAME not in collection_names:
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )
            print(f"✓ Created collection: {COLLECTION_NAME}")
        else:
            print(f"✓ Collection {COLLECTION_NAME} already exists")
    except Exception as e:
        print(f"✗ Error initializing Qdrant: {e}")
        QDRANT_AVAILABLE = False


async def get_embedding(text: str) -> list[float]:
    """Get embedding for text using OpenAI"""
    response = await openai_client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding


async def search_knowledge_base(query: str) -> str:
    """Search the Serbian history knowledge base for relevant information"""
    if not QDRANT_AVAILABLE:
        return ""  # Return empty context in fallback mode
    
    try:
        # Get query embedding
        query_embedding = await get_embedding(query)
        
        # Search in Qdrant
        search_result = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=3
        )
        
        if not search_result:
            return "Нажалост, нисам пронашао релевантне информације у бази знања. База података можда још није попуњена."
        
        # Format results
        context = "Релевантне информације из базе знања:\n\n"
        for idx, hit in enumerate(search_result, 1):
            context += f"{idx}. {hit.payload.get('text', '')}\n\n"
        
        return context
    except Exception as e:
        return f"Грешка при претраживању базе знања: {str(e)}"


async def create_agent_response(user_message: str) -> str:
    """Create response using OpenAI with RAG"""
    try:
        # Search knowledge base for relevant information (if available)
        context = await search_knowledge_base(user_message)
        
        # Create system message with context
        if QDRANT_AVAILABLE and context:
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
        'qdrant_available': QDRANT_AVAILABLE,
        'qdrant_url': QDRANT_URL if QDRANT_URL.startswith("http") else "local storage",
        'collection': COLLECTION_NAME,
        'mode': 'RAG' if QDRANT_AVAILABLE else 'FALLBACK (no RAG)'
    })


if __name__ == '__main__':
    print("="*70)
    print("🇷🇸 СРПСКИ ИСТОРИЧАР - RAG Chat System")
    print("="*70)
    
    # Initialize Qdrant collection
    initialize_qdrant_collection()
    
    # Show status
    if QDRANT_URL == ":memory:":
        print(f"\nQdrant: In-Memory Mode")
    elif QDRANT_URL.startswith("http"):
        print(f"\nQdrant: Server Mode - {QDRANT_URL}")
    else:
        print(f"\nQdrant: Local Storage - {QDRANT_URL}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Mode: {'✓ RAG (with documents)' if QDRANT_AVAILABLE else '⚠️  FALLBACK (without documents)'}")
    
    if not QDRANT_AVAILABLE:
        print("\n" + "="*70)
        print("⚠️  WARNING: Running without Qdrant / RAG functionality")
        print("The chatbot will work but without access to document database.")
        print("To enable full RAG mode, start Qdrant server first.")
        print("="*70)
    
    print("\nStarting Flask server...\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
