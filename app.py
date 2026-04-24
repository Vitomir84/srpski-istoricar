"""
Српски историчар - RAG Chat Agent
A RAG system using OpenAI and FAISS
"""
import os
import asyncio
import json
import logging
from datetime import datetime
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from openai import AsyncOpenAI
import httpx
import faiss
import numpy as np
import pickle
from pathlib import Path
import uuid
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Fix proxy issue - remove problematic proxy settings that cause "Invalid port: ':*'" error
# This is needed because system-level proxy settings with wildcards confuse httpx
if 'no_proxy' in os.environ:
    logger.warning(f"⚠️  Уклањам проблематичну no_proxy променљиву: {os.environ.get('no_proxy', '')[:50]}...")
    del os.environ['no_proxy']
if 'http_proxy' in os.environ:
    logger.warning(f"⚠️  Уклањам http_proxy променљиву за локални развој")
    del os.environ['http_proxy']
if 'https_proxy' in os.environ:
    logger.warning(f"⚠️  Уклањам https_proxy променљиву за локални развој")
    del os.environ['https_proxy']

logger.info("="*70)
logger.info("🇷🇸 СРПСКИ ИСТОРИЧАР - покретање апликације...")
logger.info("="*70)

# Configuration - Load before creating FastAPI app
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./faiss_data/serbian_history.index")
FAISS_METADATA_PATH = os.getenv("FAISS_METADATA_PATH", "./faiss_data/metadata.pkl")
BIBLIOGRAPHY_FILE = Path("bibliografija.json")
COLLECTION_NAME = "serbian_history"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

# Server Configuration
BACKEND_HOST = os.getenv("BACKEND_HOST", "127.0.0.1")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "5000"))
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS", 
    "http://localhost:3000,http://127.0.0.1:3000"
).split(",")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

# Create FastAPI app
app = FastAPI(
    title="Српски историчар API",
    description="RAG Chat Agent for Serbian History",
    version="2.0"
)

# Configure CORS - origins from environment variable
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str
    period: Optional[str] = None
    selected_documents: Optional[List[str]] = None

class ChatResponse(BaseModel):
    response: str

class HealthResponse(BaseModel):
    status: str
    faiss_available: bool
    faiss_index_path: str
    vectors_count: int
    collection: str
    mode: str

# Validate OpenAI configuration
if not OPENAI_API_KEY:
    logger.warning("⚠️  WARNING: OPENAI_API_KEY not set in environment!")
    logger.warning("   The chatbot will not work without a valid API key.")
    logger.warning("   Please set OPENAI_API_KEY in your .env file")
elif not OPENAI_API_KEY.startswith("sk-"):
    logger.warning("⚠️  WARNING: OPENAI_API_KEY doesn't look valid (should start with 'sk-')")
else:
    logger.info(f"✓ OpenAI API ključ учитан (модел: {MODEL_NAME})")

# Initialize OpenAI client for embeddings
try:
    # Create httpx client with trust_env=False to ignore system proxy settings
    http_client = httpx.AsyncClient(
        trust_env=False,  # Don't use environment proxy settings
        timeout=60.0,
        limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
    )
    
    openai_client = AsyncOpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        http_client=http_client,
        max_retries=2  # Retry failed requests up to 2 times
    )
    logger.info(f"✓ OpenAI клијент иницијализован (модел: {MODEL_NAME})")
except Exception as e:
    logger.error(f"⚠️  Грешка при иницијализацији OpenAI клијента: {e}")
    openai_client = None

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
            logger.info(f"✓ Библиографија учитана: {len(bibliography)} ставки")
        except Exception as e:
            logger.error(f"⚠️  Грешка при учитавању библиографије: {e}")
            bibliography = {}
    else:
        logger.warning(f"⚠️  Библиографија није пронађена: {BIBLIOGRAPHY_FILE}")
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
        parts.append(f"стр. {bib['strane']}")
    
    return '. '.join(parts) if parts else filename

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
        logger.info(f"✓ FAISS индекс учитан: {faiss_index.ntotal} вектора")
        FAISS_AVAILABLE = True
    else:
        # Create new empty index (dimension 1536 for OpenAI embeddings)
        faiss_index = faiss.IndexFlatL2(1536)
        faiss_metadata = []
        logger.info(f"✓ FAISS нови индекс креиран (празан)")
        FAISS_AVAILABLE = True
except Exception as e:
    logger.error(f"⚠️  FAISS НИЈЕ доступан: {e}")
    logger.warning(f"⚠️  Покретање у FALLBACK режиму (без RAG)")
    FAISS_AVAILABLE = False


def save_faiss_index():
    """Save FAISS index and metadata to disk"""
    global faiss_index, faiss_metadata
    
    if not FAISS_AVAILABLE or faiss_index is None:
        logger.warning("Прескачем FAISS снимање (није доступан)")
        return
    
    try:
        # Ensure directory exists
        Path(FAISS_INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
        
        # Save index and metadata
        faiss.write_index(faiss_index, FAISS_INDEX_PATH)
        with open(FAISS_METADATA_PATH, 'wb') as f:
            pickle.dump(faiss_metadata, f)
        logger.info(f"✓ FAISS индекс снимљен: {faiss_index.ntotal} вектора")
    except Exception as e:
        logger.error(f"✗ Грешка при снимању FAISS индекса: {e}")


async def get_embedding(text: str, max_retries: int = 3) -> list[float]:
    """Get embedding for text using OpenAI with retry logic"""
    text_preview = text[:100] + "..." if len(text) > 100 else text
    logger.debug(f"🔍 Тражим embedding за текст: {text_preview}")
    
    for attempt in range(max_retries):
        try:
            response = await openai_client.embeddings.create(
                input=text,
                model="text-embedding-ada-002",
                timeout=30.0  # 30 second timeout
            )
            logger.debug(f"✓ Embedding добијен (покушај {attempt + 1}/{max_retries})")
            return response.data[0].embedding
        except Exception as e:
            logger.warning(f"⚠️  Грешка при добијању embedding-а (покушај {attempt + 1}/{max_retries}): {str(e)}")
            if attempt == max_retries - 1:  # Last attempt
                raise Exception(f"Грешка при добијању embeddings након {max_retries} покушаја: {str(e)}")
            await asyncio.sleep(1)  # Wait 1 second before retry


async def search_knowledge_base(query: str, period: str = None, selected_documents: list = None) -> str:
    """Search the Serbian history knowledge base for relevant information"""
    logger.info(f"📚 Претраживање базе знања...")
    logger.info(f"   Упит: {query[:150]}..." if len(query) > 150 else f"   Упит: {query}")
    if period:
        logger.info(f"   Период филтер: {period}")
    if selected_documents:
        logger.info(f"   Изабрани документи: {len(selected_documents)} докумената")
        logger.info(f"   → {selected_documents}")
    
    if not FAISS_AVAILABLE or faiss_index is None or faiss_index.ntotal == 0:
        logger.warning("   ⚠️  FAISS није доступан, враћам празан контекст")
        return ""  # Return empty context in fallback mode
    
    # Check if specific documents are selected
    documents_selected = selected_documents is not None and len(selected_documents) > 0
    single_document_mode = documents_selected and len(selected_documents) == 1
    
    if documents_selected:
        if single_document_mode:
            logger.info(f"   🎯 Режим једног документа: {selected_documents[0]}")
            logger.info(f"   → Враћам више релевантних делова без филтрирања по периоду")
        else:
            logger.info(f"   🎯 Изабрано {len(selected_documents)} докумената")
            logger.info(f"   → Филтрирање по периоду је искључено")
    
    try:
        # Get query embedding with retry logic
        try:
            query_embedding = await get_embedding(query)
        except Exception as e:
            # If embedding fails, return informative message
            logger.error(f"   ✗ Грешка при добијању embedding-а: {str(e)}")
            return f"⚠️ Не могу да претражим базу знања због проблема са embeddings сервисом: {str(e)}"
        
        query_vector = np.array([query_embedding], dtype='float32')
        
        # Determine result limit based on mode
        result_limit = 15 if single_document_mode else 5
        
        # Special handling when documents are selected:
        # Search with large k, then filter by documents
        if documents_selected:
            logger.info(f"   🔍 Претражујем са великим k да бих пронашао чанкове из изабраних докумената...")
            
            # Use larger k to ensure we find chunks from selected documents
            k = min(faiss_index.ntotal, 1000)  # Search up to 1000 results
            distances, indices = faiss_index.search(query_vector, k)
            
            if len(indices[0]) == 0:
                logger.warning("   ⚠️  Нема пронађених резултата у FAISS индексу")
                return "Нажалост, нисам пронашао релевантне информације у бази знања."
            
            logger.info(f"   ✓ Пронађено {len(indices[0])} резултата у FAISS индексу")
            
            # Filter to only selected documents and keep order by relevance
            filtered_results = []
            for i in indices[0]:
                if i < len(faiss_metadata):
                    doc = faiss_metadata[i]
                    doc_source = doc.get('source_file', '')
                    
                    if doc_source in selected_documents:
                        filtered_results.append(doc)
                        logger.debug(f"      ✓ Пронашао: '{doc_source}'")
                        
                        if len(filtered_results) >= result_limit:
                            break
            
            logger.info(f"   ✓ Након филтрирања: {len(filtered_results)} чанкова из изабраних докумената")
            
        else:
            # Standard search without document filter
            k = min(20, faiss_index.ntotal)
            distances, indices = faiss_index.search(query_vector, k)
            
            if len(indices[0]) == 0:
                logger.warning("   ⚠️  Нема пронађених резултата у FAISS индексу")
                return "Нажалост, нисам пронашао релевантне информације у бази знања. База података можда још није попуњена."
            
            logger.info(f"   ✓ Пронађено {len(indices[0])} резултата у FAISS индексу")
            
            # Filter results by period
            filtered_results = []
            
            for i in indices[0]:
                if i < len(faiss_metadata):
                    doc = faiss_metadata[i]
                    
                    # Filter by period if specified
                    if period is not None and doc.get('period') != period:
                        continue
                    
                    filtered_results.append(doc)
                    if len(filtered_results) >= 5:
                        break
        
        # Check if we found any results
        if len(filtered_results) == 0:
            logger.warning(f"   ⚠️  Након филтрирања, није пронађен ниједан резултат")
            if documents_selected:
                return f"Нажалост, нисам пронашао релевантне информације у изабраним документима. Документ можда није у бази."
            if period:
                return f"Нажалост, нисам пронашао релевантне информације за изабрани период у бази знања."
            return "Нажалост, нисам пронашао релевантне информације у бази знања."
        
        if single_document_mode:
            logger.info(f"   ✓ Финално: {len(filtered_results)} релевантних делова из изабраног документа")
        else:
            logger.info(f"   ✓ Финално: {len(filtered_results)} релевантних резултата")
        
        # Format results with citations
        context = "Релевантне информације из базе знања:\n\n"
        for idx, doc in enumerate(filtered_results, 1):
            text = doc.get('text', '')
            source_file = doc.get('source_file', '')
            
            # Get formatted citation from bibliography
            citation = doc.get('citation', '')
            if not citation and source_file:
                citation = format_citation(source_file)
            if not citation:
                citation = 'Непознат извор'
            
            context += f"{idx}. {text}\n"
            context += f"   📚 Извор: {citation}\n\n"
            logger.debug(f"      Извор {idx}: {citation}")
        
        logger.info(f"   ✓ Контекст припремљен ({len(context)} карактера)")
        return context
    except Exception as e:
        logger.error(f"   ✗ Грешка при претраживању базе знања: {str(e)}")
        return f"Грешка при претраживању базе знања: {str(e)}"


async def create_agent_response(user_message: str, period: str = None, selected_documents: list = None) -> str:
    """Create response using OpenAI with RAG"""
    logger.info("="*70)
    logger.info(f"💬 НОВИ УПИТ ОД КОРИСНИКА")
    logger.info(f"   Питање: {user_message}")
    if period:
        logger.info(f"   Филтерperiода: {period}")
    if selected_documents:
        logger.info(f"   Изабрани документи: {selected_documents}")
    logger.info("="*70)
    
    # Check if OpenAI client is initialized
    if openai_client is None:
        logger.error("✗ OpenAI клијент није иницијализован!")
        return (
            "Извините, OpenAI клијент није исправно иницијализован. "
            "Проверите да ли је API кључ правилно подешен у .env фајлу."
        )
    
    try:
        # Search knowledge base for relevant information (if available)
        context = await search_knowledge_base(user_message, period, selected_documents)
        
        # Create system message with context
        if FAISS_AVAILABLE and context:
            logger.info("📝 Припремам PROMPT са RAG контекстом...")
            # Full RAG mode
            system_message = (
                "Ти си виртуелни српски историчар и научни асистент.\n\n"
                "Имаш опште знање о српској историји али имаш и базу докумената коју увек треба да претражиш и извучеш из ње релевантне информације које треба да сложиш у одговор.\n"
                "Увек треба да навадеш документе из којих црпиш информације користећи тачан формат навођења извора.\n\n"
                "Правила:\n"
                "1. Одговарај научно и прецизно\n"
                "2. Ако ниси сигуран, признај то\n"
                "3. Увек наводи изворе користећи ТАЧАН ФОРМАТ који је наведен уз сваки извор (нпр. 'Аутор (година). Наслов. Издавач. стр.')\n"
                "4. Када наводиш извор, користи тачно онај назив који је наведен после '📚 Извор:' у контексту\n"
                "5. Тон је научан, неутралан и поуздан\n"
                "6. Ако база знања није попуњена, обавести корисника да база можда још није попуњена и да ће одговор бити базиран на општем знању\n"
                "7. Сваки документ је катогиризован по теми и периоду: нови_век, рани_век, средњи_век и остало\n"
                "8. Ако је корисник навео период, користи само документе из тог периода\n"
                "9. Увек одговарај на српском језику и само на ћирилици\n"
                "10. Ако је корисник изабрао само један или више докумената, одговорај само на основу тих докумената, без коришћења општег знања\n"
                "11. На крају одговора, наведи све коришћене изворе у посебном одељку 'Извори:' користећи тачан формат из контекста\n"
            )
            user_content = f"Контекст из базе знања:\n{context}\n\nКорисниково питање: {user_message}"
            logger.debug(f"\n{'='*70}")
            logger.debug(f"PROMPT - СИСТЕМ:\n{system_message[:500]}...")
            logger.debug(f"\nPROMPT - КОРИСНИК:\n{user_content[:500]}...")
            logger.debug(f"{'='*70}\n")
        else:
            logger.warning("⚠️  Радим у FALLBACK режиму (без RAG)")
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
            logger.debug(f"\n{'='*70}")
            logger.debug(f"PROMPT - СИСТЕМ (FALLBACK):\n{system_message[:500]}...")
            logger.debug(f"\nPROMPT - КОРИСНИК:\n{user_content}")
            logger.debug(f"{'='*70}\n")
        
        # Create messages for the chat
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content}
        ]
        
        # Get response from OpenAI with timeout and retry
        logger.info(f"🤖 Шаљем захтев OpenAI-ју (модел: {MODEL_NAME})...")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    logger.info(f"   Покушај {attempt + 1}/{max_retries}...")
                response = await openai_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.01,
                    timeout=60.0  # 60 second timeout
                )
                
                answer = response.choices[0].message.content
                logger.info(f"✓ Одговор примљен од OpenAI-ја ({len(answer)} карактера)")
                logger.info(f"\n{'='*70}")
                logger.info(f"ОДГОВОР:\n{answer[:500]}..." if len(answer) > 500 else f"ОДГОВОР:\n{answer}")
                logger.info(f"{'='*70}\n")
                return answer
            except Exception as e:
                error_message = str(e).lower()
                
                # Check for specific error types
                logger.error(f"✗ Грешка при OpenAI захтеву: {str(e)}")
                if "connection" in error_message or "timeout" in error_message:
                    if attempt < max_retries - 1:
                        logger.warning(f"   ⏳ Чекам 2 секунде пре поновног покушаја...")
                        await asyncio.sleep(2)  # Wait 2 seconds before retry
                        continue
                    return (
                        "Извините, тренутно не могу да се повежем са OpenAI сервисом. "
                        "Могући узроци:\n"
                        "• Проблем са интернет везом\n"
                        "• OpenAI сервис је тренутно недоступан\n"
                        "• API кључ можда није важећи или је истекао\n\n"
                        f"Детаљи: {str(e)}"
                    )
                elif "api_key" in error_message or "authentication" in error_message or "unauthorized" in error_message:
                    return (
                        "Извините, дошло је до проблема са аутентификацијом. "
                        "API кључ можда није важећи или је истекао.\n\n"
                        f"Детаљи: {str(e)}"
                    )
                elif "rate_limit" in error_message or "quota" in error_message:
                    return (
                        "Извините, достигнут је лимит захтева или квота за API.\n\n"
                        f"Детаљи: {str(e)}"
                    )
                else:
                    logger.error(f"✗ Непозната грешка: {str(e)}")
                    return f"Извините, дошло је до грешке: {str(e)}"
                    
    except Exception as e:
        logger.error(f"✗✗✗ Неочекивана грешка у create_agent_response: {str(e)}")
        return f"Извините, дошло је до неочекиване грешке: {str(e)}"


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main application interface"""
    logger.info(f"🌐 GET / захтев - служи HTML интерфејс")
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api")
async def api_info():
    """API information endpoint"""
    logger.info(f"🌐 GET /api захтев")
    return {
        'message': 'Српски историчар API',
        'version': '2.0',
        'framework': 'FastAPI',
        'endpoints': {
            'chat': '/api/chat',
            'documents': '/api/documents',
            'health': '/api/health',
            'static': '/static/{filename}',
            'docs': '/docs'
        }
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle chat messages"""
    try:
        logger.info(f"\n🌐 API /chat захтев")
        
        if not request.message:
            logger.warning(f"   ⚠️  Празна порука")
            raise HTTPException(status_code=400, detail="Message is required")
        
        # Call async function directly (no need for event loop management in FastAPI)
        response = await create_agent_response(
            request.message, 
            request.period, 
            request.selected_documents
        )
        
        logger.info(f"✓ Одговор послат кориснику\n")
        return ChatResponse(response=response)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"✗ Грешка у /api/chat endpoint-у: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents")
async def get_documents():
    """Get list of all unique documents in the database"""
    logger.info(f"🌐 API /documents захтев")
    try:
        if not FAISS_AVAILABLE or not faiss_metadata:
            logger.warning("   ⚠️  FAISS није доступан, враћам празну листу")
            return {'documents': []}
        
        # Extract unique documents with their metadata
        documents_dict = {}
        
        for doc in faiss_metadata:
            source_file = doc.get('source_file', '')
            if source_file and source_file not in documents_dict:
                # Format citation from bibliography
                citation = format_citation(source_file)
                
                # Get bibliographic data
                bib_data = bibliography.get(source_file, {})
                
                documents_dict[source_file] = {
                    'filename': source_file,
                    'citation': citation,
                    'display_name': citation,  # Use citation as display name
                    'autor': doc.get('autor') or bib_data.get('autor', ''),
                    'godina': doc.get('godina') or bib_data.get('godina', ''),
                    'naslov': doc.get('naslov') or bib_data.get('naslov', ''),
                    'izdavac': bib_data.get('izdavac', ''),
                    'strane': bib_data.get('strane', ''),
                    'period': doc.get('period', ''),
                    'period_name': doc.get('period_name', '')
                }
        
        # Convert to list and sort by citation
        documents_list = sorted(documents_dict.values(), 
                               key=lambda x: x.get('citation', x['filename']))
        
        logger.info(f"   ✓ Враћам листу од {len(documents_list)} докумената")
        return {'documents': documents_list}
    except Exception as e:
        logger.error(f"✗ Грешка у /api/documents endpoint-у: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    logger.info(f"🌐 API /health захтев")
    
    health_info = HealthResponse(
        status='ok',
        faiss_available=FAISS_AVAILABLE,
        faiss_index_path=FAISS_INDEX_PATH,
        vectors_count=faiss_index.ntotal if faiss_index else 0,
        collection=COLLECTION_NAME,
        mode='RAG' if FAISS_AVAILABLE else 'FALLBACK (no RAG)'
    )
    
    logger.info(f"   ✓ Health check: {health_info.mode}, {health_info.vectors_count} вектора")
    return health_info


if __name__ == '__main__':
    import uvicorn
    
    logger.info("\n" + "="*70)
    logger.info("🇷🇸 СРПСКИ ИСТОРИЧАР - Покретање FastAPI сервера")
    logger.info("="*70)
    
    # Show status
    logger.info(f"\n📊 СТАТУС СИСТЕМА:")
    logger.info(f"   FAISS Index: {FAISS_INDEX_PATH}")
    logger.info(f"   Metadata: {FAISS_METADATA_PATH}")
    logger.info(f"   Collection: {COLLECTION_NAME}")
    logger.info(f"   Vectors: {faiss_index.ntotal if faiss_index else 0}")
    logger.info(f"   Mode: {'✓ RAG (са документима)' if FAISS_AVAILABLE and faiss_index and faiss_index.ntotal > 0 else '⚠️  FALLBACK (без докумената)'}")
    
    if not FAISS_AVAILABLE or not faiss_index or faiss_index.ntotal == 0:
        logger.warning("\n" + "="*70)
        logger.warning("⚠️  УПОЗОРЕЊЕ: Покретање без FAISS / RAG функционалности")
        logger.warning("Chatbot ће радити али без приступа бази докумената.")
        logger.warning("Да омогућите пун RAG режим, прво попуните базу.")
        logger.warning("Покрените: python populate_database.py")
        logger.warning("="*70)
    
    logger.info(f"\n🚀 Покретам FastAPI сервер на http://{BACKEND_HOST}:{BACKEND_PORT}")
    logger.info(f"📚 API документација доступна на: http://{BACKEND_HOST}:{BACKEND_PORT}/docs")
    logger.info(f"🌐 Дозвољени CORS извори: {', '.join(ALLOWED_ORIGINS)}\n")
    
    uvicorn.run(app, host=BACKEND_HOST, port=BACKEND_PORT, log_level="info")
