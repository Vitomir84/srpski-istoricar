"""
Српски историчар - RAG Chat Agent
A RAG system using OpenAI and FAISS
"""
import os
import asyncio
import json
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
BIBLIOGRAPHY_FILE = Path("bibliografija.json")
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
bibliography = {}

# Load bibliography
def load_bibliography():
    """Load bibliography data from JSON file"""
    global bibliography
    
    if BIBLIOGRAPHY_FILE.exists():
        try:
            with open(BIBLIOGRAPHY_FILE, 'r', encoding='utf-8') as f:
                bibliography = json.load(f)
            print(f"✓ Bibliography loaded: {len(bibliography)} entries")
        except Exception as e:
            print(f"⚠️  Error loading bibliography: {e}")
            bibliography = {}
    else:
        print(f"⚠️  Bibliography not found: {BIBLIOGRAPHY_FILE}")
        bibliography = {}

# Load bibliography on startup
load_bibliography()

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


async def search_knowledge_base(query: str, period: str = None, selected_documents: list = None) -> str:
    """Search the Serbian history knowledge base for relevant information"""
    if not FAISS_AVAILABLE or faiss_index is None or faiss_index.ntotal == 0:
        return ""  # Return empty context in fallback mode
    
    try:
        # Get query embedding
        query_embedding = await get_embedding(query)
        query_vector = np.array([query_embedding], dtype='float32')
        
        # Search in FAISS (k=20 to have enough after filtering)
        k = min(20, faiss_index.ntotal)  # Don't search for more items than we have
        distances, indices = faiss_index.search(query_vector, k)
        
        if len(indices[0]) == 0:
            return "Нажалост, нисам пронашао релевантне информације у бази знања. База података можда још није попуњена."
        
        # Filter results by period and/or selected documents
        filtered_results = []
        for i in indices[0]:
            if i < len(faiss_metadata):
                doc = faiss_metadata[i]
                
                # Filter by period if specified
                if period is not None and doc.get('period') != period:
                    continue
                
                # Filter by selected documents if specified
                if selected_documents is not None and len(selected_documents) > 0:
                    doc_source = doc.get('source_file', '')
                    if doc_source not in selected_documents:
                        continue
                
                filtered_results.append(doc)
                if len(filtered_results) >= 5:  # Limit to top 5
                    break
        
        if len(filtered_results) == 0:
            if selected_documents and len(selected_documents) > 0:
                return f"Нажалост, нисам пронашао релевантне информације у изабраним документима."
            return f"Нажалост, нисам пронашао релевантне информације за изабрани период у бази знања."
        
        # Format results with citations
        context = "Релевантне информације из базе знања:\n\n"
        for idx, doc in enumerate(filtered_results, 1):
            text = doc.get('text', '')
            citation = doc.get('citation', doc.get('source_file', 'Непознат извор'))
            
            context += f"{idx}. {text}\n"
            context += f"   Извор: {citation}\n\n"
        
        return context
    except Exception as e:
        return f"Грешка при претраживању базе знања: {str(e)}"


async def create_agent_response(user_message: str, period: str = None, selected_documents: list = None) -> str:
    """Create response using OpenAI with RAG"""
    try:
        # Search knowledge base for relevant information (if available)
        context = await search_knowledge_base(user_message, period, selected_documents)
        
        # Create system message with context
        if FAISS_AVAILABLE and context:
            # Full RAG mode
            system_message = (
                "Ти си виртуелни српски историчар и научни асистент.\n\n"
                "Имаш опште знање о српској историји али имаш и базу докумената коју увек треба да претражиш и извучеш из ње релевантне информације које треба да сложиш у одговор.\n"
                "Увек треба да навадеш документе из којих црпиш информације.\n\n"
                "Правила:\n"
                "1. Одговарај научно и прецизно\n"
                "2. Ако ниси сигуран, признај то\n"
                "3. Увек наводи изворе и документе из којих црпиш информације\n"
                "4. Тон је научан, неутралан и поуздан\n"
                "5. Ако база знања није попуњена, обавести корисника да база можда још није попуњена и да ће одговор бити базиран на општем знању\n"
                "6. Сваки документ је катогиризован по теми и периоду: нови_век, рани_век, средњи_век и остало.\n"
                "7. Ако је корисник навео период, користи само документе из тог периода.\n"
                "8. Увек одговарај на српском језику и само на ћирилици.\n"
                "9. Ако је корисник изабрао само један или више докумената, одговорај само на основу тих докумената, без коришћења општег знања.\n"
            )
            user_content = f"Контекст из базе знања:\n{context}\n\nКорисниково питање: {user_message}"
        else:
            # Fallback mode without RAG
            system_message = (
                "Ти си виртуелни српски историчар и научни асистент.\n\n"
                "Имаш опште знање о српској историји \n"
                "Правила:\n"
                "1. Одговарај научно и прецизно\n"
                "2. Ако ниси сигуран, признај то\n"
                "3. Тон је научан, неутралан и поуздан\n"
                "4. Ако база знања није попуњена, обавести корисника да база можда још није попуњена и да ће одговор бити базиран на општем знању\n"
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
        period = data.get('period', None)
        selected_documents = data.get('selected_documents', None)
        
        if not user_message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Run async function in event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response = loop.run_until_complete(create_agent_response(user_message, period, selected_documents))
        loop.close()
        
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/documents', methods=['GET'])
def get_documents():
    """Get list of all unique documents in the database"""
    try:
        if not FAISS_AVAILABLE or not faiss_metadata:
            return jsonify({'documents': []})
        
        # Extract unique documents with their metadata
        documents_dict = {}
        
        for doc in faiss_metadata:
            source_file = doc.get('source_file', '')
            if source_file and source_file not in documents_dict:
                # Try to get citation from metadata first
                citation = doc.get('citation', '')
                
                # If no citation in metadata, format from bibliography
                if not citation and source_file in bibliography:
                    bib = bibliography[source_file]
                    parts = []
                    if bib.get('autor'):
                        parts.append(bib['autor'])
                    if bib.get('godina'):
                        if parts:
                            parts[-1] += f" ({bib['godina']})"
                        else:
                            parts.append(f"({bib['godina']})")
                    if bib.get('naslov'):
                        parts.append(bib['naslov'])
                    citation = '. '.join(parts) if parts else source_file
                elif not citation:
                    citation = source_file
                
                # Get bibliographic data
                bib_data = bibliography.get(source_file, {})
                
                documents_dict[source_file] = {
                    'filename': source_file,
                    'citation': citation,
                    'autor': doc.get('autor') or bib_data.get('autor', ''),
                    'godina': doc.get('godina') or bib_data.get('godina', ''),
                    'naslov': doc.get('naslov') or bib_data.get('naslov', ''),
                    'izdavac': bib_data.get('izdavac', ''),
                    'strane': bib_data.get('strane', ''),
                    'period': doc.get('period', ''),
                    'period_name': doc.get('period_name', '')
                }
        
        # Convert to list and sort by author/title
        documents_list = sorted(documents_dict.values(), 
                               key=lambda x: (x.get('autor', ''), x.get('naslov', x['filename'])))
        
        return jsonify({'documents': documents_list})
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
