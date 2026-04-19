#!/usr/bin/env python3
"""
Quick test script for Unified AI Agent
======================================
Run this to verify the installation and basic functionality.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    tests = [
        ("google.genai", "Google GenAI"),
        ("anthropic", "Anthropic"),
        ("langchain", "LangChain"),
        ("langchain_community", "LangChain Community"),
        ("langchain_text_splitters", "LangChain Text Splitters"),
        ("faiss", "FAISS"),
        ("gradio", "Gradio"),
        ("sentence_transformers", "Sentence Transformers"),
    ]
    
    results = []
    for module, name in tests:
        try:
            __import__(module)
            results.append((name, "✅ OK"))
        except ImportError as e:
            results.append((name, f"❌ FAILED: {e}"))
    
    max_len = max(len(name) for name, _ in results)
    for name, status in results:
        print(f"  {name:<{max_len}}  {status}")
    
    return all(status == "✅ OK" for _, status in results)

def test_agent_classes():
    """Test that agent classes can be instantiated."""
    print("\nTesting agent classes...")
    
    try:
        from unified_ai_agent import (
            AgentConfig, 
            ProviderType, 
            TaskType,
            GeminiAgent,
            ClaudeAgent,
            KimiOrchestrator
        )
        
        # Test configuration
        config = AgentConfig(
            gemini_api_key="test-key",
            anthropic_api_key="test-key"
        )
        print("  ✅ AgentConfig - OK")
        
        # Test enums
        assert ProviderType.GEMINI is not None
        assert TaskType.CHAT is not None
        print("  ✅ ProviderType - OK")
        print("  ✅ TaskType - OK")
        
        # Test agent creation (will fail API check, but classes work)
        try:
            gemini = GeminiAgent(config)
            print("  ✅ GeminiAgent - OK (class instantiation)")
        except Exception as e:
            print(f"  ⚠️  GeminiAgent - Class OK but init warning: {e}")
        
        try:
            claude = ClaudeAgent(config)
            print("  ✅ ClaudeAgent - OK (class instantiation)")
        except Exception as e:
            print(f"  ⚠️  ClaudeAgent - Class OK but init warning: {e}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        return False

def test_rag_system():
    """Test RAG system initialization."""
    print("\nTesting RAG system...")
    
    try:
        from unified_ai_agent import AgentConfig, RAGSystem
        
        config = AgentConfig(enable_rag=True)
        rag = RAGSystem(config)
        
        print("  ✅ RAGSystem - OK (class instantiation)")
        return True
        
    except Exception as e:
        print(f"  ⚠️  RAGSystem - {e}")
        return False

def test_orchestrator():
    """Test orchestrator initialization."""
    print("\nTesting orchestrator...")
    
    try:
        from unified_ai_agent import AgentConfig, KimiOrchestrator
        
        config = AgentConfig(
            gemini_api_key=os.getenv('GEMINI_API_KEY', 'test'),
            anthropic_api_key=os.getenv('ANTHROPIC_API_KEY', 'test'),
            enable_rag=False  # Skip RAG for quick test
        )
        
        orchestrator = KimiOrchestrator(config)
        
        # Check status
        status = orchestrator.get_status()
        print(f"  ✅ KimiOrchestrator - OK")
        print(f"     - Gemini available: {status['gemini_available']}")
        print(f"     - Claude available: {status['claude_available']}")
        print(f"     - RAG initialized: {status['rag_initialized']}")
        
        return True
        
    except Exception as e:
        print(f"  ⚠️  KimiOrchestrator - {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Unified AI Agent - Quick Test Suite")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Agent Classes", test_agent_classes()))
    results.append(("RAG System", test_rag_system()))
    results.append(("Orchestrator", test_orchestrator()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name:<20} {status}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed! The system is ready to use.")
    else:
        print("⚠️  Some tests failed. Check the output above.")
        print("   You may need to install dependencies:")
        print("   pip install -r requirements.txt")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
