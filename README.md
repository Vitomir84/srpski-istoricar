# Српски историчар - RAG Chat Agent

A Retrieval-Augmented Generation (RAG) system for Serbian history built with:
- **FAISS** - Local vector database for document storage and retrieval
- **OpenAI** - For LLM and embeddings
- **Flask** - Web backend
- **Poetry** - Dependency management

## Features

- 🏛️ Chat interface in Serbian language
- 🔍 RAG-based responses using FAISS vector store
- 📚 PDF document parsing and indexing
- 💾 Fully local vector storage (no external database needed)
- 🎨 Beautiful, responsive UI
- 📖 OCR support for scanned documents (via Tesseract)

## Prerequisites

- Python 3.11 or higher
- Poetry (for dependency management)
- OpenAI API key
- Tesseract OCR (optional, for scanned PDFs)

## Installation

### 1. Clone or Download

Download this project to your machine.

### 2. Install Poetry

If you don't have Poetry installed:

```bash
pip install poetry
```

### 3. Install Dependencies

Using Poetry (recommended):

```bash
poetry install
```

Or using pip:

```bash
pip install -r requirements.txt
```

### 4. Install Tesseract OCR (Optional)

For OCR support on scanned PDFs:

**Windows:**
- Download from: https://github.com/UB-Mannheim/tesseract/wiki
- Make sure to install Serbian language pack (`srp.traineddata`)
- Add to PATH or set `TESSERACT_CMD` in `.env`

**Linux:**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-srp
```

**macOS:**
```bash
brew install tesseract tesseract-lang
```

### 5. Configure Environment Variables

Copy `.env.example` to `.env`:

```bash
copy .env.example .env
```

Edit `.env` and add your credentials:

```env
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o-mini
FAISS_INDEX_PATH=./faiss_data/serbian_history.index
FAISS_METADATA_PATH=./faiss_data/metadata.pkl
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe  # Windows only, if not in PATH
```

## Usage

### 1. Test the System

Verify everything is set up correctly:

```bash
poetry run python test_system.py
```

This checks:
- Python libraries
- Tesseract OCR
- FAISS database
- OpenAI API
- PDF files in `docs/` folder

### 2. Add Documents to the Database

#### Option A: Add Sample Documents (Quick Start)

```bash
poetry run python populate_database.py
# Choose option 1 for sample Serbian history documents
```

#### Option B: Parse All PDFs from docs/ Folder

```bash
poetry run python pdf_parser.py
```

This will:
- Scan all PDFs in `docs/novi_vek/`, `docs/rani_vek/`, `docs/srednji_vek/`, `docs/ostalo/`
- Extract text (with OCR fallback for scanned docs)
- Split into chunks (~1000 characters)
- Create embeddings via OpenAI
- Store in FAISS index
- Save progress after each file

#### Option C: Add a Single New PDF

```bash
poetry run python add_pdf_to_faiss.py "path/to/document.pdf" --period novi_vek
```

Periods: `novi_vek`, `rani_vek`, `srednji_vek`, `ostalo`

**Force re-indexing:**
```bash
poetry run python add_pdf_to_faiss.py "document.pdf" --period novi_vek --force
```

### 3. Start the Application

```bash
poetry run python app.py
```

The application will:
- Load the FAISS index from disk
- Start the Flask server on http://localhost:5000

### 4. Open the Chat Interface

Open your browser and go to:
```
http://localhost:5000
```

### 5. Explore the FAISS Index

View indexed documents and search:

```bash
poetry run python explore_faiss.py
```

This lets you:
- View index statistics
- Browse sample documents
- Search by query
- Interactive search mode

## Document Management

### Parsing PDF Files

The system supports automatic PDF parsing with the following features:

#### `pdf_parser.py` - Batch PDF Processing

Processes all PDFs from organized folders:

```bash
poetry run python pdf_parser.py
```

**Features:**
- Automatic text extraction from text-based PDFs
- OCR fallback for scanned documents (requires Tesseract)
- Smart text chunking (1000 chars with 200 char overlap)
- Metadata preservation (period, source file, chunk index)
- Progress tracking and automatic saving
- Handles large documents efficiently

**Folder Structure:**
```
docs/
├── novi_vek/        # Modern era documents
├── rani_vek/        # Early medieval
├── srednji_vek/     # Middle ages
└── ostalo/          # Other/miscellaneous
```

#### `add_pdf_to_faiss.py` - Single PDF Addition

Add one PDF at a time with duplicate detection:

```bash
# Add a new PDF
poetry run python add_pdf_to_faiss.py "novo-delo.pdf" --period novi_vek

# From specific location
poetry run python add_pdf_to_faiss.py "C:\Downloads\knjiga.pdf" --period srednji_vek

# Force re-index existing file
poetry run python add_pdf_to_faiss.py "dokument.pdf" --period novi_vek --force
```

**Arguments:**
- `pdf_file` - Path to PDF file (required)
- `--period` - Historical period: `novi_vek`, `rani_vek`, `srednji_vek`, `ostalo` (default: `ostalo`)
- `--force` - Re-index even if already in database

#### `populate_database.py` - Manual Document Addition

For text-based content and testing:

```bash
poetry run python populate_database.py
```

**Options:**
1. Add sample documents (6 pre-defined Serbian history texts)
2. Load from text file (`srpski-istoricar.rs.txt`)
3. Check database status

### Exploring the Index

#### `explore_faiss.py` - Index Explorer

```bash
poetry run python explore_faiss.py
```

**Features:**
- View index statistics (total vectors, by period, by source)
- Browse sample documents
- Search by semantic query
- Interactive search mode

## Project Structure

```
.
├── app.py                      # Main Flask application
├── populate_database.py        # Add documents manually
├── pdf_parser.py              # Parse all PDFs from docs/
├── add_pdf_to_faiss.py        # Add single PDF to FAISS
├── explore_faiss.py           # Browse and search FAISS index
├── test_system.py             # System verification
├── requirements.txt           # Python dependencies (pip)
├── pyproject.toml             # Poetry configuration
├── .env.example              # Environment variables template
├── .env                      # Your configuration (create this)
├── faiss_data/               # FAISS index storage (created automatically)
│   ├── serbian_history.index # Vector index
│   └── metadata.pkl          # Document metadata
├── docs/                     # PDF documents to index
│   ├── novi_vek/            # Modern era PDFs
│   ├── rani_vek/            # Early medieval PDFs
│   ├── srednji_vek/         # Middle ages PDFs
│   └── ostalo/              # Other PDFs
├── templates/
│   └── index.html           # Chat UI
├── pictures/                # Static images
└── README.md                # This file
```

## How It Works

1. **User Query**: User asks a question in the chat interface
2. **Embedding**: The query is converted to a vector embedding (OpenAI text-embedding-ada-002)
3. **Retrieval**: FAISS searches for relevant documents using L2 distance similarity
4. **Augmentation**: Retrieved documents are provided as context to the LLM
5. **Generation**: OpenAI GPT generates a response using the context
6. **Response**: The answer is displayed in the chat interface

### Document Indexing Process

1. **PDF Extraction**: Text extracted from PDFs (pypdf) or OCR (Tesseract)
2. **Chunking**: Documents split into ~1000 character chunks with overlap
3. **Embedding**: Each chunk converted to 1536-dim vector (OpenAI)
4. **Storage**: Vectors stored in FAISS index, metadata in pickle file
5. **Persistence**: Index saved to disk after each document

## API Endpoints

- `GET /` - Chat interface
- `POST /api/chat` - Send a message and get a response
  - Request: `{"message": "Ко је био Стефан Немања?"}`
  - Response: `{"response": "..."}`
- `GET /api/health` - Health check
  - Returns: `{"status": "ok", "faiss_available": true, "vectors_count": 1234, ...}`

## Customization

### Change the Agent's Instructions

Edit the `instructions` parameter in `app.py`:

```python
async with ChatAgent(
    chat_client=chat_client,
    model=MODEL_NAME,
    instructions="Your custom instructions here...",
    tools=[search_knowledge_base],
) as agent:
```

### Use Different Models

Update the `MODEL_NAME` in `.env`:
- `gpt-4o-mini` - Fast and cost-effective
- `gpt-4o` - More capable, higher cost
- `gpt-4-turbo` - Previous generation

### Use Azure OpenAI

Update `.env`:
```env
OPENAI_BASE_URL=https://your-resource.openai.azure.com/
OPENAI_API_KEY=your-azure-openai-key
MODEL_NAME=your-deployment-name
```

## Deployment

### Deploy to Production

For production deployment:

1. **Use environment variables** for all secrets
2. **Disable debug mode** in `app.py`:
   ```python
   app.run(host='0.0.0.0', port=5000, debug=False)
   ```
3. **Use a production WSGI server** like Gunicorn:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

### Deploy to Azure

1. Create an Azure App Service
2. Configure environment variables in Azure Portal
3. Deploy using Git or Azure CLI

### Deploy with Docker

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt --pre
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

Build and run:
```bash
docker build -t srpski-istoricar .
docker run -p 5000:5000 --env-file .env srpski-istoricar
```

## Troubleshooting

### "FAISS index not found" or empty database
The FAISS index is created automatically. To populate:
```bash
poetry run python populate_database.py  # Option 1 for samples
# OR
poetry run python pdf_parser.py         # Parse all PDFs
```

### "No relevant information found"
The database is empty or query doesn't match indexed content. Check:
```bash
poetry run python explore_faiss.py  # View what's indexed
```

### PDF parsing fails
- **Text-based PDFs**: Should work out of the box
- **Scanned PDFs**: Requires Tesseract OCR installation
- Check: `poetry run python test_system.py`

### Tesseract not found
Install Tesseract and add to PATH, or set in `.env`:
```env
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

### OpenAI API errors
- Check API key is valid
- Check you have credits
- Monitor rate limits (0.3s delay between chunk embeddings)

### Import errors
Make sure to use Poetry:
```bash
poetry install
poetry run python app.py
```

## Next Steps

1. **Add more documents**: Use `populate_database.py` to populate the knowledge base
2. **Improve chunking**: Split large documents into smaller, meaningful chunks
3. **Add metadata**: Include dates, sources, and categories in document metadata
4. **Enhance UI**: Add features like conversation history, citations, etc.
5. **Add authentication**: Secure the application with user authentication

## Resources

- [FAISS Documentation](https://github.com/facebookresearch/faiss/wiki)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [Poetry Documentation](https://python-poetry.org/docs/)

## License

MIT License

## Support

For issues and questions, please check the documentation or create an issue in the project repository.
