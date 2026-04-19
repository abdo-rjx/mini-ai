# Unified AI Agent - mini-ai

A sophisticated multi-provider AI agent system that seamlessly integrates **Google Gemini 2.5 Flash** and **Anthropic Claude** APIs, with **Kimi (Moonshot AI)** serving as the intelligent orchestrator.

![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🌟 Features

### Multi-Provider AI Integration
- **Gemini 2.5 Flash**: Fast, efficient responses for creative and general tasks
- **Claude (Sonnet/Opus/Haiku)**: Superior analytical and coding capabilities
- **Intelligent Routing**: Kimi orchestrator automatically selects the best agent for each task

### Retrieval-Augmented Generation (RAG)
- **PDF Document Processing**: Automatically indexes PDF files from a local folder
- **FAISS Vector Database**: Efficient similarity search with HuggingFace embeddings
- **Context-Aware Responses**: Agents retrieve relevant document context automatically

### Task Specialization
| Task Type | Preferred Agent | Why |
|-----------|-----------------|-----|
| Code Generation | Claude | Superior code quality and reasoning |
| Analysis | Claude | Deep analytical capabilities |
| RAG Queries | Gemini | Fast retrieval and synthesis |
| General Chat | Configurable | User-defined preference |
| Summarization | Gemini | Efficient processing |

### User Interfaces
- **Web UI**: Beautiful Gradio interface with chat, status panels, and controls
- **CLI Mode**: Full-featured command-line interface for terminal users
- **API-Ready**: Modular design allows easy integration into other applications

## 📁 Project Structure

```
.
├── unified_ai_agent.py    # Main application (single-file deployment)
├── requirements.txt       # Python dependencies
├── README.md             # This documentation
├── books/                # Place PDF documents here for RAG
└── vector_db/            # Auto-generated FAISS vector database
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the files
git clone <repository-url>
cd unified-ai-agent

# Install dependencies
pip install -r requirements.txt
```

### 2. API Key Configuration

Set your API keys as environment variables:

```bash
# For Google Gemini
export GEMINI_API_KEY="your-gemini-api-key"

# For Anthropic Claude
export ANTHROPIC_API_KEY="sk-ant-your-anthropic-key"
```

**Get your API keys:**
- Gemini: https://makersuite.google.com/app/apikey
- Claude: https://console.anthropic.com/settings/keys

### 3. Run the Application

```bash
# Start web UI (default)
python unified_ai_agent.py

# Start CLI mode
python unified_ai_agent.py --cli
```

### 4. Add Documents for RAG (Optional)

Place PDF files in the `books/` folder:

```bash
mkdir -p books
cp your_documents/*.pdf books/
```

The vector database will be built automatically on first run.

## 📖 Usage Guide

### Web Interface

The web UI provides an intuitive chat interface with:
- **Main Chat**: Conversational interface with the AI
- **Status Panel**: View system status and available agents
- **Controls**: Clear history, refresh status

Access at `http://127.0.0.1:7860` after starting.

### CLI Commands

When running in CLI mode (`--cli`):

```
/status  - Show system status
/clear   - Clear conversation history
/agents  - List available agents
/rag     - Toggle RAG mode on/off
/quit    - Exit the application
```

### Command-Line Options

```bash
python unified_ai_agent.py [OPTIONS]

Options:
  --cli                   Run in command-line mode
  --books PATH            Path to PDF documents folder (default: ./books)
  --db PATH               Path to vector database folder (default: ./vector_db)
  --gemini-model MODEL    Gemini model to use (default: gemini-2.5-flash)
  --claude-model MODEL    Claude model to use (default: claude-sonnet-4-6)
  --default-provider      Default provider: gemini or claude
  --no-rag                Disable RAG functionality
  --host HOST             Host for web UI (default: 127.0.0.1)
  --port PORT             Port for web UI (default: 7860)
  --share                 Create public shareable link
  -h, --help              Show help message
```

### Examples

```bash
# Run with custom document path
python unified_ai_agent.py --books ~/Documents/PDFs --db ~/vector_stores/main

# Use Claude as default provider
python unified_ai_agent.py --default-provider claude

# Run CLI with specific models
python unified_ai_agent.py --cli --gemini-model gemini-1.5-pro --claude-model claude-opus-4-6

# Create public link (Gradio sharing)
python unified_ai_agent.py --share
```

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                     mini-ai                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Task Router → Agent Selector → Response Synthesizer │   │
│  └─────────────────────────────────────────────────────┘   │
└──────────────────────┬────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
   ┌────▼─────┐  ┌────▼─────┐  ┌─────▼────┐
   │  Gemini  │  │  Claude  │  │   RAG    │
   │  Agent   │  │  Agent   │  │  System  │
   └──────────┘  └──────────┘  └──────────┘
```

### Agent Selection Logic

1. **Task Classification**: Query is analyzed to determine task type
2. **Capability Matching**: Best agent is selected based on task requirements
3. **Fallback Chain**: If primary agent unavailable, fallback is used
4. **Context Enrichment**: RAG context is added when relevant

### RAG Pipeline

1. **Document Ingestion**: PDFs are loaded and split into chunks
2. **Embedding**: HuggingFace `all-MiniLM-L6-v2` creates embeddings
3. **Indexing**: FAISS stores vectors for fast retrieval
4. **Retrieval**: Top-k relevant chunks are fetched per query
5. **Augmentation**: Context is prepended to the LLM prompt

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GEMINI_API_KEY` | Google Gemini API key | Optional* |
| `ANTHROPIC_API_KEY` | Anthropic Claude API key | Optional* |

*At least one API key is required for the system to function.

### AgentConfig Parameters

```python
@dataclass
class AgentConfig:
    gemini_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    gemini_model: str = 'gemini-2.5-flash'
    claude_model: str = 'claude-sonnet-4-6'
    books_path: str = "./books"
    db_path: str = "./vector_db"
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 700
    chunk_overlap: int = 100
    search_k: int = 4
    default_provider: ProviderType = ProviderType.GEMINI
    enable_rag: bool = True
```

## 🔧 Advanced Usage

### Programmatic API

```python
from unified_ai_agent import KimiOrchestrator, AgentConfig, ProviderType

# Configure
config = AgentConfig(
    gemini_api_key="your-key",
    claude_api_key="your-key",
    default_provider=ProviderType.CLAUDE
)

# Initialize
orchestrator = KimiOrchestrator(config)

# Process queries
import asyncio
response = asyncio.run(orchestrator.process("Your question here"))
print(response)
```

### Custom Tool Registration

```python
# Register a custom tool
def my_custom_tool(param: str) -> str:
    return f"Processed: {param}"

orchestrator.tools["my_tool"] = my_custom_tool
```

### Adding Documents Programmatically

```python
# Add new PDFs to the vector store
orchestrator.rag_system.add_documents([
    "/path/to/new_document.pdf"
])
```

## 🐛 Troubleshooting

### Common Issues

#### No agents available
```
Warning: No AI agents are available!
```
**Solution**: Set at least one API key:
```bash
export GEMINI_API_KEY="your-key"
# or
export ANTHROPIC_API_KEY="your-key"
```

#### RAG not working
```
RAG system is not initialized.
```
**Solution**: 
1. Create the `books/` folder: `mkdir books`
2. Add PDF files: `cp *.pdf books/`
3. Restart the application

#### Import errors
```
ModuleNotFoundError: No module named 'gradio'
```
**Solution**: Install dependencies:
```bash
pip install -r requirements.txt
```

### Performance Tips

1. **Use GPU for embeddings**: Set `device='cuda'` in HuggingFaceEmbeddings for faster indexing
2. **Adjust chunk size**: Larger chunks = more context, smaller chunks = more precise retrieval
3. **Pre-build vector DB**: Build once, reuse across sessions
4. **Use release builds**: For production, ensure optimized Python environment

## 📊 Model Comparison

| Feature | Gemini 2.5 Flash | Claude Sonnet |
|---------|------------------|---------------|
| Speed | ⚡ Very Fast | 🚀 Fast |
| Code Quality | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Analysis | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Creativity | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Context Window | 1M tokens | 200K tokens |
| Cost | $ | $$ |

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional LLM providers (OpenAI, Cohere, etc.)
- Enhanced tool ecosystem
- Multi-modal capabilities
- Distributed agent coordination
- Performance optimizations

## 📜 License

MIT License - See LICENSE file for details.

## 🙏 Acknowledgments

- **Google** for Gemini API
- **Anthropic** for Claude API
- **LangChain** for the RAG framework
- **HuggingFace** for embeddings and transformers
- **Gradio** for the web interface

---

**Built with ❤️ by the Unified AI Agent Team**
