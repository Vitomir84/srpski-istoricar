# Српски историчар - RAG Chat Agent

A Retrieval-Augmented Generation (RAG) system for Serbian history built with:
- **Microsoft Agent Framework** - For building the AI agent
- **Qdrant** - Vector database for document storage and retrieval
- **OpenAI** - For LLM and embeddings
- **Flask** - Web backend

## Features

- 🏛️ Chat interface in Serbian language
- 🔍 RAG-based responses using Qdrant vector database
- 🤖 Built with Microsoft Agent Framework
- 📚 Ready for document ingestion
- 🎨 Beautiful, responsive UI

## Prerequisites

- Python 3.9 or higher
- Qdrant database (local or cloud)
- OpenAI API key

## Installation

### 1. Clone or Download

Download this project to your machine.

### 2. Install Dependencies

**IMPORTANT**: The `--pre` flag is required while Agent Framework is in preview.

```bash
pip install -r requirements.txt --pre
```

Or install packages individually:

```bash
pip install agent-framework-azure-ai --pre
pip install flask flask-cors openai qdrant-client python-dotenv
```

### 3. Set Up Qdrant

#### Option A: Local Qdrant (Recommended for Development)

Using Docker:
```bash
docker run -p 6333:6333 qdrant/qdrant
```

Or download from: https://qdrant.tech/documentation/quick-start/

#### Option B: Qdrant Cloud

1. Sign up at https://cloud.qdrant.io/
2. Create a cluster
3. Get your cluster URL and API key

### 4. Configure Environment Variables

Copy `.env.example` to `.env`:

```bash
copy .env.example .env
```

Edit `.env` and add your credentials:

```env
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o-mini
QDRANT_URL=http://localhost:6333
```

For Qdrant Cloud, update:
```env
QDRANT_URL=https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key
```

## Usage

### 1. Start the Application

```bash
python app.py
```

The application will:
- Initialize the Qdrant collection automatically
- Start the Flask server on http://localhost:5000

### 2. Open the Chat Interface

Open your browser and go to:
```
http://localhost:5000
```

### 3. Add Documents to the Database

Use the `populate_database.py` script to add documents:

```bash
python populate_database.py
```

See the "Adding Documents" section below for details.

## Adding Documents

The `populate_database.py` script helps you add documents to Qdrant:

```python
python populate_database.py
```

You can modify this script to:
- Load documents from text files
- Process PDFs or other formats
- Chunk large documents appropriately

Example usage in the script:
```python
# Add a single document
await add_document("Стефан Немања је био велики жупан...")

# Add multiple documents from a file
with open('documents/serbian_history.txt', 'r', encoding='utf-8') as f:
    for paragraph in f.read().split('\n\n'):
        if paragraph.strip():
            await add_document(paragraph)
```

## Project Structure

```
.
├── app.py                      # Main Flask application
├── populate_database.py        # Script to add documents to Qdrant
├── requirements.txt            # Python dependencies
├── .env.example               # Environment variables template
├── .env                       # Your configuration (create this)
├── templates/
│   └── index.html            # Chat UI
└── README.md                 # This file
```

## How It Works

1. **User Query**: User asks a question in the chat interface
2. **Embedding**: The query is converted to a vector embedding
3. **Retrieval**: Qdrant searches for relevant documents using vector similarity
4. **Augmentation**: Retrieved documents are provided as context to the LLM
5. **Generation**: Microsoft Agent Framework generates a response using the context
6. **Response**: The answer is displayed in the chat interface

## API Endpoints

- `GET /` - Chat interface
- `POST /api/chat` - Send a message and get a response
- `GET /api/health` - Health check

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

### "Collection not found" error
The collection is created automatically on first run. If you see this error, restart the application.

### "No relevant information found"
The database is empty. Run `populate_database.py` to add documents.

### Connection errors
- Check Qdrant is running: Visit http://localhost:6333/dashboard
- Check OpenAI API key is valid
- Ensure environment variables are loaded

## Next Steps

1. **Add more documents**: Use `populate_database.py` to populate the knowledge base
2. **Improve chunking**: Split large documents into smaller, meaningful chunks
3. **Add metadata**: Include dates, sources, and categories in document metadata
4. **Enhance UI**: Add features like conversation history, citations, etc.
5. **Add authentication**: Secure the application with user authentication

## Resources

- [Microsoft Agent Framework](https://github.com/microsoft/agent-framework)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [OpenAI API Documentation](https://platform.openai.com/docs)

## License

MIT License

## Support

For issues and questions, please check the documentation or create an issue in the project repository.
