#!/usr/bin/env python3
"""
Unified AI Agent - Kimi Orchestrator
=====================================
A multi-provider AI agent system that integrates Gemini and Claude APIs
with Kimi (Moonshot AI) as the primary orchestrator.

Features:
- Multi-provider LLM support (Gemini 2.5 Flash, Claude Sonnet/Opus/Haiku)
- RAG capabilities with FAISS vector database
- PDF document processing and embedding
- Intelligent agent routing and orchestration
- Unified chat interface via Gradio
- Tool execution framework

Author: Unified Integration System
Version: 1.0.0
"""

import os
import sys
import json
import time
import asyncio
import logging
from typing import Optional, List, Dict, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# DEPENDENCY CHECK AND INSTALLATION HELPERS
# =============================================================================

def check_dependencies():
    """Check and report missing dependencies with installation instructions."""
    missing = []
    optional_missing = []
    
    required = {
        'google.generativeai': 'google-generativeai>=0.8.0',
        'anthropic': 'anthropic>=0.30.0',
        'langchain': 'langchain>=0.2.0',
        'langchain_community': 'langchain-community>=0.2.0',
        'langchain_text_splitters': 'langchain-text-splitters>=0.2.0',
        'langchain_huggingface': 'langchain-huggingface>=0.0.3',
        'faiss': 'faiss-cpu>=1.8.0',
        'gradio': 'gradio>=4.0.0',
        'sentence_transformers': 'sentence-transformers>=2.7.0',
    }
    
    for module, package in required.items():
        try:
            __import__(module.replace('.', '_') if module == 'faiss' else module)
        except ImportError:
            if module in ['faiss', 'sentence_transformers']:
                optional_missing.append(package)
            else:
                missing.append(package)
    
    if missing or optional_missing:
        print("=" * 70)
        print("MISSING DEPENDENCIES DETECTED")
        print("=" * 70)
        if missing:
            print("\nRequired packages (install with):")
            print(f"  pip install {' '.join(missing)}")
        if optional_missing:
            print("\nOptional packages for enhanced functionality:")
            print(f"  pip install {' '.join(optional_missing)}")
        print("\nOr install all at once:")
        print(f"  pip install {' '.join(missing + optional_missing)}")
        print("=" * 70)
        return False
    return True

# =============================================================================
# IMPORTS (after dependency check)
# =============================================================================

try:
    import google.generativeai as genai
    from google.generativeai.types import GenerationConfig
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("Google Generative AI not available. Gemini features disabled.")

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("Anthropic SDK not available. Claude features disabled.")

try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain components not available. RAG features disabled.")

try:
    from ui import GradioInterface, GRADIO_AVAILABLE
except ImportError:
    GRADIO_AVAILABLE = False
    GradioInterface = None  # type: ignore
    logger.warning("ui.py not found or Gradio not available. Web UI disabled.")

# =============================================================================
# CONFIGURATION AND DATA CLASSES
# =============================================================================

class ProviderType(Enum):
    """Supported LLM providers."""
    GEMINI = auto()
    CLAUDE = auto()
    KIMI = auto()  # Placeholder for future Kimi API integration

class TaskType(Enum):
    """Types of tasks the agent can handle."""
    CHAT = auto()
    RAG_QUERY = auto()
    CODE_GENERATION = auto()
    ANALYSIS = auto()
    SUMMARIZATION = auto()
    TOOL_EXECUTION = auto()

@dataclass
class AgentConfig:
    """Configuration for the unified agent system."""
    # API Keys (loaded from environment or passed directly)
    gemini_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    # Default models
    gemini_model: str = 'gemini-2.5-flash'
    claude_model: str = 'claude-sonnet-4-6'
    
    # RAG Configuration
    books_path: str = "./books"
    db_path: str = "./vector_db"
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 700
    chunk_overlap: int = 100
    search_k: int = 4
    
    # Agent Behavior
    default_provider: ProviderType = ProviderType.GEMINI
    enable_rag: bool = True
    verbose: bool = True
    
    # System prompts
    orchestrator_prompt: str = field(default_factory=lambda: """You are Kimi, the orchestrator of a multi-agent AI system. Your role is to:
1. Analyze user requests and determine the best approach
2. Route tasks to the appropriate specialized agent (Gemini for creative tasks, Claude for analytical tasks)
3. Synthesize responses from multiple agents when beneficial
4. Maintain context across the conversation
5. Execute tools when necessary

Be helpful, accurate, and efficient in coordinating the AI agents.""")

@dataclass
class Message:
    """A chat message."""
    role: str  # 'user', 'assistant', 'system', 'tool'
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

@dataclass
class AgentResponse:
    """Response from an agent."""
    content: str
    provider: ProviderType
    task_type: TaskType
    metadata: Dict[str, Any] = field(default_factory=dict)
    tool_calls: List[Dict] = field(default_factory=list)

# =============================================================================
# BASE AGENT CLASS
# =============================================================================

class BaseAgent(ABC):
    """Abstract base class for all AI agents."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.message_history: List[Message] = []
    
    @abstractmethod
    async def generate(self, prompt: str, context: Optional[str] = None, 
                       task_type: TaskType = TaskType.CHAT) -> AgentResponse:
        """Generate a response from the agent."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this agent is properly configured and available."""
        pass
    
    def add_to_history(self, message: Message):
        """Add a message to the conversation history."""
        self.message_history.append(message)
        # Keep last 20 messages for context
        if len(self.message_history) > 20:
            self.message_history = self.message_history[-20:]
    
    def clear_history(self):
        """Clear conversation history."""
        self.message_history = []

# =============================================================================
# GEMINI AGENT IMPLEMENTATION
# =============================================================================

class GeminiAgent(BaseAgent):
    """Gemini 2.5 Flash agent implementation."""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.client = None
        self.model = None
        self._initialize()
    
    def _initialize(self):
        """Initialize the Gemini client."""
        if not GEMINI_AVAILABLE:
            return
        
        api_key = self.config.gemini_api_key or os.getenv('GEMINI_API_KEY')
        if not api_key:
            logger.warning("Gemini API key not found. Set GEMINI_API_KEY environment variable.")
            return
        
        try:
            genai.configure(api_key=api_key)
            self.client = genai
            self.model = genai.GenerativeModel(self.config.gemini_model)
            logger.info(f"Gemini agent initialized with model: {self.config.gemini_model}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
    
    def is_available(self) -> bool:
        """Check if Gemini is available."""
        return GEMINI_AVAILABLE and self.model is not None
    
    async def generate(self, prompt: str, context: Optional[str] = None,
                       task_type: TaskType = TaskType.CHAT) -> AgentResponse:
        """Generate a response using Gemini."""
        if not self.is_available():
            return AgentResponse(
                content="Gemini agent is not available. Please check your API key.",
                provider=ProviderType.GEMINI,
                task_type=task_type,
                metadata={"error": "not_available"}
            )
        
        try:
            # Build the full prompt with context
            full_prompt = self._build_prompt(prompt, context, task_type)
            
            # Configure generation parameters based on task type
            generation_config = self._get_generation_config(task_type)
            
            # Generate response
            response = await asyncio.to_thread(
                self.model.generate_content,
                full_prompt,
                generation_config=generation_config
            )
            
            return AgentResponse(
                content=response.text,
                provider=ProviderType.GEMINI,
                task_type=task_type,
                metadata={
                    "model": self.config.gemini_model,
                    "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0),
                    "completion_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0),
                }
            )
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            return AgentResponse(
                content=f"Error generating response: {str(e)}",
                provider=ProviderType.GEMINI,
                task_type=task_type,
                metadata={"error": str(e), "traceback": traceback.format_exc()}
            )
    
    def _build_prompt(self, prompt: str, context: Optional[str], task_type: TaskType) -> str:
        """Build the full prompt with system instructions and context."""
        system_instruction = self._get_system_instruction(task_type)
        
        parts = [system_instruction]
        
        if context:
            parts.append(f"\n{'='*50}\nRETRIEVED CONTEXT:\n{'='*50}\n{context}\n")
        
        parts.append(f"\n{'='*50}\nUSER QUERY:\n{'='*50}\n{prompt}\n")
        parts.append(f"\n{'='*50}\nYOUR RESPONSE:\n{'='*50}")
        
        return "\n".join(parts)
    
    def _get_system_instruction(self, task_type: TaskType) -> str:
        """Get system instruction based on task type."""
        instructions = {
            TaskType.CHAT: "You are a helpful AI assistant. Provide clear, accurate, and helpful responses.",
            TaskType.RAG_QUERY: "You are a knowledgeable research assistant. Answer based on the provided context. If the context doesn't contain the answer, say so clearly.",
            TaskType.CODE_GENERATION: "You are an expert programmer. Write clean, well-documented, and efficient code. Include comments explaining key logic.",
            TaskType.ANALYSIS: "You are an analytical expert. Provide thorough analysis with clear reasoning and conclusions.",
            TaskType.SUMMARIZATION: "You are a summarization expert. Create concise, accurate summaries that capture the key points.",
            TaskType.TOOL_EXECUTION: "You are a tool execution agent. Execute the requested tool and report results clearly.",
        }
        return instructions.get(task_type, instructions[TaskType.CHAT])
    
    def _get_generation_config(self, task_type: TaskType) -> GenerationConfig:
        """Get generation configuration based on task type."""
        configs = {
            TaskType.CODE_GENERATION: GenerationConfig(
                temperature=0.2,
                max_output_tokens=8192,
            ),
            TaskType.ANALYSIS: GenerationConfig(
                temperature=0.3,
                max_output_tokens=4096,
            ),
            TaskType.SUMMARIZATION: GenerationConfig(
                temperature=0.5,
                max_output_tokens=2048,
            ),
        }
        return configs.get(task_type, GenerationConfig(
            temperature=0.7,
            max_output_tokens=4096,
        ))

# =============================================================================
# CLAUDE AGENT IMPLEMENTATION
# =============================================================================

class ClaudeAgent(BaseAgent):
    """Claude agent implementation."""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.client = None
        self._initialize()
    
    def _initialize(self):
        """Initialize the Anthropic client."""
        if not ANTHROPIC_AVAILABLE:
            return
        
        api_key = self.config.anthropic_api_key or os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            logger.warning("Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable.")
            return
        
        try:
            self.client = Anthropic(api_key=api_key)
            logger.info(f"Claude agent initialized with model: {self.config.claude_model}")
        except Exception as e:
            logger.error(f"Failed to initialize Claude: {e}")
    
    def is_available(self) -> bool:
        """Check if Claude is available."""
        return ANTHROPIC_AVAILABLE and self.client is not None
    
    async def generate(self, prompt: str, context: Optional[str] = None,
                       task_type: TaskType = TaskType.CHAT) -> AgentResponse:
        """Generate a response using Claude."""
        if not self.is_available():
            return AgentResponse(
                content="Claude agent is not available. Please check your API key.",
                provider=ProviderType.CLAUDE,
                task_type=task_type,
                metadata={"error": "not_available"}
            )
        
        try:
            # Build messages
            messages = self._build_messages(prompt, context, task_type)
            
            # Get system prompt
            system = self._get_system_instruction(task_type)
            
            # Generate response
            response = await asyncio.to_thread(
                self.client.messages.create,
                model=self.config.claude_model,
                max_tokens=4096,
                system=system,
                messages=messages
            )
            
            return AgentResponse(
                content=response.content[0].text,
                provider=ProviderType.CLAUDE,
                task_type=task_type,
                metadata={
                    "model": self.config.claude_model,
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                }
            )
        except Exception as e:
            logger.error(f"Claude generation error: {e}")
            return AgentResponse(
                content=f"Error generating response: {str(e)}",
                provider=ProviderType.CLAUDE,
                task_type=task_type,
                metadata={"error": str(e), "traceback": traceback.format_exc()}
            )
    
    def _build_messages(self, prompt: str, context: Optional[str], 
                        task_type: TaskType) -> List[Dict]:
        """Build message list for Claude API."""
        messages = []
        
        # Add conversation history (last 5 exchanges)
        for msg in self.message_history[-10:]:
            role = "user" if msg.role == "user" else "assistant"
            messages.append({"role": role, "content": msg.content})
        
        # Build current prompt with context
        content = prompt
        if context:
            content = f"Context:\n{context}\n\nQuestion: {prompt}"
        
        messages.append({"role": "user", "content": content})
        return messages
    
    def _get_system_instruction(self, task_type: TaskType) -> str:
        """Get system instruction based on task type."""
        instructions = {
            TaskType.CHAT: "You are Claude, a helpful AI assistant. Provide clear, accurate, and thoughtful responses.",
            TaskType.RAG_QUERY: "You are a research assistant. Answer questions based on the provided context. Be precise and cite information from the context when possible.",
            TaskType.CODE_GENERATION: "You are Claude, an expert software engineer. Write production-quality code with proper error handling, documentation, and best practices.",
            TaskType.ANALYSIS: "You are an analytical expert. Provide thorough, well-reasoned analysis with clear conclusions.",
            TaskType.SUMMARIZATION: "You are a summarization expert. Create accurate, concise summaries that preserve key information.",
            TaskType.TOOL_EXECUTION: "You are a tool execution agent. Execute tools precisely and report results clearly.",
        }
        return instructions.get(task_type, instructions[TaskType.CHAT])

# =============================================================================
# RAG SYSTEM
# =============================================================================

class RAGSystem:
    """Retrieval-Augmented Generation system using FAISS."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.embeddings = None
        self.vector_store = None
        self.retriever = None
        self.is_initialized = False
    
    def initialize(self) -> bool:
        """Initialize the RAG system."""
        if not LANGCHAIN_AVAILABLE:
            logger.warning("LangChain not available. RAG features disabled.")
            return False
        
        try:
            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.config.embedding_model
            )
            
            # Check for existing vector database
            if os.path.exists(self.config.db_path):
                logger.info("Loading existing vector database...")
                self.vector_store = FAISS.load_local(
                    self.config.db_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                self.retriever = self.vector_store.as_retriever(
                    search_kwargs={"k": self.config.search_k}
                )
                self.is_initialized = True
                return True
            
            # Build new vector database if books exist
            if os.path.isdir(self.config.books_path):
                return self._build_vector_db()
            
            logger.info("No existing vector DB and no books folder found. RAG ready but empty.")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            return False
    
    def _build_vector_db(self) -> bool:
        """Build vector database from PDF files."""
        try:
            logger.info("Building vector database from books...")
            start_time = time.time()
            
            # Find PDF files
            pdf_files = [
                f for f in os.listdir(self.config.books_path) 
                if f.lower().endswith('.pdf')
            ]
            
            if not pdf_files:
                logger.warning(f"No PDF files found in {self.config.books_path}")
                return True
            
            logger.info(f"Found {len(pdf_files)} PDF file(s)")
            
            # Load documents
            docs = []
            for i, file in enumerate(pdf_files, 1):
                file_path = os.path.join(self.config.books_path, file)
                logger.info(f"[{i}/{len(pdf_files)}] Loading: {file}")
                try:
                    loader = PyPDFLoader(file_path)
                    file_docs = loader.load()
                    docs.extend(file_docs)
                    logger.info(f"  Loaded {len(file_docs)} pages")
                except Exception as e:
                    logger.error(f"  Failed to load {file}: {e}")
            
            if not docs:
                logger.warning("No documents could be loaded")
                return True
            
            # Split documents
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
            chunks = splitter.split_documents(docs)
            logger.info(f"Split into {len(chunks)} chunks")
            
            # Create vector store
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
            self.vector_store.save_local(self.config.db_path)
            
            self.retriever = self.vector_store.as_retriever(
                search_kwargs={"k": self.config.search_k}
            )
            
            logger.info(f"Vector DB created in {time.time() - start_time:.1f}s")
            self.is_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to build vector DB: {e}")
            return False
    
    def query(self, question: str) -> str:
        """Query the RAG system."""
        if not self.is_initialized or not self.retriever:
            return "RAG system is not initialized."
        
        try:
            docs = self.retriever.invoke(question)
            if not docs:
                return "No relevant documents found."
            
            context_parts = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get('source', 'Unknown')
                page = doc.metadata.get('page', 'N/A')
                context_parts.append(
                    f"[Document {i}] Source: {source}, Page: {page}\n{doc.page_content}"
                )
            
            return "\n\n---\n\n".join(context_parts)
        except Exception as e:
            logger.error(f"RAG query error: {e}")
            return f"Error querying documents: {str(e)}"
    
    def add_documents(self, file_paths: List[str]) -> bool:
        """Add new documents to the vector store."""
        if not self.is_initialized:
            logger.error("RAG system not initialized")
            return False
        
        try:
            new_docs = []
            for file_path in file_paths:
                if file_path.lower().endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                    new_docs.extend(loader.load())
            
            if new_docs:
                self.vector_store.add_documents(new_docs)
                self.vector_store.save_local(self.config.db_path)
                logger.info(f"Added {len(new_docs)} documents to vector store")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False

# =============================================================================
# KIMI ORCHESTRATOR
# =============================================================================

class KimiOrchestrator:
    """
    Kimi Orchestrator - The central coordinator for the multi-agent system.
    
    Responsibilities:
    - Route tasks to the appropriate agent based on task type and availability
    - Manage conversation context and state
    - Coordinate between multiple agents for complex tasks
    - Execute tools when needed
    - Provide unified interface to users
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.gemini_agent = GeminiAgent(config)
        self.claude_agent = ClaudeAgent(config)
        self.rag_system = RAGSystem(config) if config.enable_rag else None
        self.conversation_history: List[Message] = []
        self.tools: Dict[str, Callable] = {}
        
        # Initialize RAG if enabled
        if self.rag_system:
            self.rag_system.initialize()
        
        # Register default tools
        self._register_default_tools()
        
        logger.info("Kimi Orchestrator initialized")
    
    def _register_default_tools(self):
        """Register default available tools."""
        self.tools = {
            "search_documents": self._tool_search_documents,
            "list_agents": self._tool_list_agents,
            "get_status": self._tool_get_status,
        }
    
    def _tool_search_documents(self, query: str) -> str:
        """Tool: Search documents in the RAG system."""
        if self.rag_system and self.rag_system.is_initialized:
            return self.rag_system.query(query)
        return "Document search is not available."
    
    def _tool_list_agents(self) -> str:
        """Tool: List available agents."""
        agents = []
        if self.gemini_agent.is_available():
            agents.append(f"- Gemini ({self.config.gemini_model}): Available")
        else:
            agents.append(f"- Gemini: Not available (check API key)")
        
        if self.claude_agent.is_available():
            agents.append(f"- Claude ({self.config.claude_model}): Available")
        else:
            agents.append(f"- Claude: Not available (check API key)")
        
        return "Available Agents:\n" + "\n".join(agents)
    
    def _tool_get_status(self) -> str:
        """Tool: Get system status."""
        status = []
        status.append(f"Kimi Orchestrator Status:")
        status.append(f"  - Gemini Agent: {'✓' if self.gemini_agent.is_available() else '✗'}")
        status.append(f"  - Claude Agent: {'✓' if self.claude_agent.is_available() else '✗'}")
        status.append(f"  - RAG System: {'✓' if self.rag_system and self.rag_system.is_initialized else '✗'}")
        status.append(f"  - Conversation History: {len(self.conversation_history)} messages")
        return "\n".join(status)
    
    def _determine_task_type(self, query: str) -> TaskType:
        """Determine the type of task based on the query."""
        query_lower = query.lower()
        
        # Code-related keywords
        code_keywords = ['code', 'program', 'function', 'script', 'implement', 'write a', 'debug']
        if any(kw in query_lower for kw in code_keywords):
            return TaskType.CODE_GENERATION
        
        # Analysis keywords
        analysis_keywords = ['analyze', 'analysis', 'compare', 'evaluate', 'assess', 'review']
        if any(kw in query_lower for kw in analysis_keywords):
            return TaskType.ANALYSIS
        
        # Summarization keywords
        summary_keywords = ['summarize', 'summary', 'tl;dr', 'brief', 'overview']
        if any(kw in query_lower for kw in summary_keywords):
            return TaskType.SUMMARIZATION
        
        # Document query keywords
        doc_keywords = ['document', 'book', 'pdf', 'according to', 'from the', 'in the text']
        if any(kw in query_lower for kw in doc_keywords):
            return TaskType.RAG_QUERY
        
        return TaskType.CHAT
    
    def _select_agent(self, task_type: TaskType) -> Optional[BaseAgent]:
        """Select the best agent for the task."""
        # Routing logic
        if task_type == TaskType.CODE_GENERATION:
            # Claude is generally better at code
            if self.claude_agent.is_available():
                return self.claude_agent
            return self.gemini_agent if self.gemini_agent.is_available() else None
        
        elif task_type == TaskType.ANALYSIS:
            # Both are good at analysis, prefer Claude for complex analysis
            if self.claude_agent.is_available():
                return self.claude_agent
            return self.gemini_agent if self.gemini_agent.is_available() else None
        
        elif task_type == TaskType.RAG_QUERY:
            # Gemini is faster for RAG queries
            if self.gemini_agent.is_available():
                return self.gemini_agent
            return self.claude_agent if self.claude_agent.is_available() else None
        
        # Default: use configured default or first available
        if self.config.default_provider == ProviderType.GEMINI:
            if self.gemini_agent.is_available():
                return self.gemini_agent
            return self.claude_agent if self.claude_agent.is_available() else None
        else:
            if self.claude_agent.is_available():
                return self.claude_agent
            return self.gemini_agent if self.gemini_agent.is_available() else None
    
    async def process(self, query: str, use_rag: bool = True) -> str:
        """
        Process a user query through the orchestrator.
        
        Args:
            query: The user's query
            use_rag: Whether to use RAG for context retrieval
        
        Returns:
            The agent's response as a string
        """
        # Add user message to history
        user_msg = Message(role="user", content=query)
        self.conversation_history.append(user_msg)
        
        # Determine task type
        task_type = self._determine_task_type(query)
        logger.info(f"Task type determined: {task_type.name}")
        
        # Get RAG context if enabled and relevant
        context = None
        if use_rag and self.rag_system and self.rag_system.is_initialized:
            if task_type in [TaskType.RAG_QUERY, TaskType.ANALYSIS, TaskType.CHAT]:
                context = self.rag_system.query(query)
                logger.info(f"Retrieved context length: {len(context) if context else 0} chars")
        
        # Select agent
        agent = self._select_agent(task_type)
        if not agent:
            return "Error: No AI agents are available. Please check your API keys."
        
        logger.info(f"Selected agent: {agent.__class__.__name__}")
        
        # Generate response
        response = await agent.generate(query, context=context, task_type=task_type)
        
        # Add assistant message to history
        assistant_msg = Message(
            role="assistant", 
            content=response.content,
            metadata={
                "provider": response.provider.name,
                "task_type": response.task_type.name,
                **response.metadata
            }
        )
        self.conversation_history.append(assistant_msg)
        
        return response.content
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        self.gemini_agent.clear_history()
        self.claude_agent.clear_history()
        logger.info("Conversation history cleared")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            "gemini_available": self.gemini_agent.is_available(),
            "claude_available": self.claude_agent.is_available(),
            "rag_initialized": self.rag_system.is_initialized if self.rag_system else False,
            "conversation_length": len(self.conversation_history),
            "default_provider": self.config.default_provider.name,
        }

# =============================================================================
# GRADIO UI  →  see ui.py
# =============================================================================
# GradioInterface has been moved to ui.py.
# Import is handled at the top of this file.

# =============================================================================
# CLI INTERFACE
# =============================================================================

def run_cli(orchestrator: KimiOrchestrator):
    """Run the command-line interface."""
    print("=" * 70)
    print("🤖 Unified AI Agent - Kimi Orchestrator (CLI Mode)")
    print("=" * 70)
    print("\nCommands:")
    print("  /status  - Show system status")
    print("  /clear   - Clear conversation history")
    print("  /agents  - List available agents")
    print("  /rag     - Toggle RAG mode")
    print("  /quit    - Exit")
    print("=" * 70)
    
    use_rag = True
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith('/'):
                cmd = user_input[1:].lower()
                
                if cmd == 'quit' or cmd == 'exit':
                    print("Goodbye!")
                    break
                
                elif cmd == 'status':
                    print("\n" + orchestrator._tool_get_status())
                
                elif cmd == 'clear':
                    orchestrator.clear_history()
                    print("Conversation history cleared.")
                
                elif cmd == 'agents':
                    print("\n" + orchestrator._tool_list_agents())
                
                elif cmd == 'rag':
                    use_rag = not use_rag
                    print(f"RAG mode: {'ON' if use_rag else 'OFF'}")
                
                else:
                    print(f"Unknown command: /{cmd}")
                
                continue
            
            # Process query
            print("\nThinking...", end='', flush=True)
            response = asyncio.run(orchestrator.process(user_input, use_rag=use_rag))
            print("\r          \r", end='')  # Clear "Thinking..."
            print(f"\nAgent: {response}")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point for the unified AI agent."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Unified AI Agent - Kimi Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables:
  GEMINI_API_KEY      API key for Google Gemini
  ANTHROPIC_API_KEY   API key for Anthropic Claude

Examples:
  # Run web UI (default)
  python unified_ai_agent.py
  
  # Run CLI mode
  python unified_ai_agent.py --cli
  
  # Specify custom paths
  python unified_ai_agent.py --books ./my_docs --db ./my_vectors
        """
    )
    
    parser.add_argument('--cli', action='store_true',
                        help='Run in command-line mode instead of web UI')
    parser.add_argument('--books', default='./books',
                        help='Path to PDF documents folder (default: ./books)')
    parser.add_argument('--db', default='./vector_db',
                        help='Path to vector database folder (default: ./vector_db)')
    parser.add_argument('--gemini-model', default='gemini-2.5-flash',
                        help='Gemini model to use (default: gemini-2.5-flash)')
    parser.add_argument('--claude-model', default='claude-sonnet-4-6',
                        help='Claude model to use (default: claude-sonnet-4-6)')
    parser.add_argument('--default-provider', choices=['gemini', 'claude'], default='gemini',
                        help='Default provider for general queries (default: gemini)')
    parser.add_argument('--no-rag', action='store_true',
                        help='Disable RAG functionality')
    parser.add_argument('--host', default='127.0.0.1',
                        help='Host to bind the web UI (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=7860,
                        help='Port to bind the web UI (default: 7860)')
    parser.add_argument('--share', action='store_true',
                        help='Create a public shareable link for the web UI')
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        print("\nPlease install missing dependencies and try again.")
        sys.exit(1)
    
    # Create configuration
    config = AgentConfig(
        gemini_api_key=os.getenv('GEMINI_API_KEY'),
        anthropic_api_key=os.getenv('ANTHROPIC_API_KEY'),
        gemini_model=args.gemini_model,
        claude_model=args.claude_model,
        books_path=args.books,
        db_path=args.db,
        default_provider=ProviderType.GEMINI if args.default_provider == 'gemini' else ProviderType.CLAUDE,
        enable_rag=not args.no_rag,
    )
    
    # Initialize orchestrator
    print("🚀 Initializing Kimi Orchestrator...")
    orchestrator = KimiOrchestrator(config)
    
    # Check if any agents are available
    status = orchestrator.get_status()
    if not status['gemini_available'] and not status['claude_available']:
        print("\n⚠️  Warning: No AI agents are available!")
        print("Please set at least one of these environment variables:")
        print("  - GEMINI_API_KEY (for Google Gemini)")
        print("  - ANTHROPIC_API_KEY (for Anthropic Claude)")
        print("\nThe system will start but responses will be limited.")
    
    # Run in appropriate mode
    if args.cli:
        run_cli(orchestrator)
    else:
        if not GRADIO_AVAILABLE:
            print("\n⚠️  Gradio not available. Falling back to CLI mode.")
            print("Install with: pip install gradio")
            run_cli(orchestrator)
        else:
            print(f"\n🌐 Starting web UI at http://{args.host}:{args.port}")
            print("Press Ctrl+C to stop\n")
            
            ui = GradioInterface(orchestrator)
            ui.launch(
                server_name=args.host,
                server_port=args.port,
                share=args.share,
                inbrowser=True
            )

if __name__ == "__main__":
    main()
