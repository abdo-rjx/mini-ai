#!/usr/bin/env python3
"""
Setup script for Unified AI Agent
==================================
Handles installation, dependency checking, and initial configuration.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def print_banner():
    """Print the setup banner."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║           🤖 UNIFIED AI AGENT - KIMI ORCHESTRATOR                   ║
║                                                                      ║
║              Setup and Configuration Utility                         ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print(f"❌ Python {version.major}.{version.minor} is not supported.")
        print("   Please use Python 3.9 or higher.")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def install_dependencies(upgrade=False):
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")
    
    cmd = [sys.executable, "-m", "pip", "install"]
    if upgrade:
        cmd.append("--upgrade")
    cmd.extend(["-r", "requirements.txt"])
    
    try:
        subprocess.check_call(cmd)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    
    dirs = ["books", "vector_db", "logs"]
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"   📂 {dir_name}/")
    
    print("✅ Directories created")

def setup_environment_file():
    """Create .env file template if it doesn't exist."""
    env_file = Path(".env")
    
    if env_file.exists():
        print("\n⚠️  .env file already exists. Skipping creation.")
        return
    
    print("\n📝 Creating .env template...")
    
    env_content = """# Unified AI Agent - Environment Configuration
# =============================================
# Copy this file to .env and fill in your API keys

# Google Gemini API Key
# Get yours at: https://makersuite.google.com/app/apikey
GEMINI_API_KEY=your_gemini_api_key_here

# Anthropic Claude API Key
# Get yours at: https://console.anthropic.com/settings/keys
ANTHROPIC_API_KEY=sk-ant-your_anthropic_key_here

# Optional: Custom paths
# BOOKS_PATH=./books
# VECTOR_DB_PATH=./vector_db

# Optional: Logging level
# LOG_LEVEL=INFO
"""
    
    env_file.write_text(env_content)
    print("✅ Created .env template")
    print("   🔔 Please edit .env and add your API keys")

def check_api_keys():
    """Check if API keys are configured."""
    print("\n🔑 Checking API keys...")
    
    gemini_key = os.getenv('GEMINI_API_KEY')
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    
    if gemini_key and gemini_key != 'your_gemini_api_key_here':
        print("   ✅ GEMINI_API_KEY is set")
    else:
        print("   ⚠️  GEMINI_API_KEY not configured")
    
    if anthropic_key and anthropic_key != 'sk-ant-your_anthropic_key_here':
        print("   ✅ ANTHROPIC_API_KEY is set")
    else:
        print("   ⚠️  ANTHROPIC_API_KEY not configured")
    
    if not gemini_key and not anthropic_key:
        print("\n   ❌ No API keys configured!")
        print("      The application will not function without at least one API key.")
        print("      Please set GEMINI_API_KEY or ANTHROPIC_API_KEY in your environment.")
        return False
    
    return True

def test_imports():
    """Test if key imports work."""
    print("\n🧪 Testing imports...")
    
    imports = [
        ("google.genai", "Gemini API"),
        ("anthropic", "Claude API"),
        ("langchain", "LangChain"),
        ("faiss", "FAISS"),
        ("gradio", "Gradio"),
    ]
    
    all_ok = True
    for module, name in imports:
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} - NOT INSTALLED")
            all_ok = False
    
    return all_ok

def print_next_steps():
    """Print next steps for the user."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                         🎉 SETUP COMPLETE!                           ║
╚══════════════════════════════════════════════════════════════════════╝

Next steps:
-----------

1. 🔑 Configure your API keys:
   - Edit the .env file and add your API keys
   - Or set environment variables:
     export GEMINI_API_KEY="your-key"
     export ANTHROPIC_API_KEY="your-key"

2. 📚 Add documents for RAG (optional):
   - Place PDF files in the ./books/ folder
   - They will be indexed automatically on first run

3. 🚀 Run the application:
   - Web UI:  python unified_ai_agent.py
   - CLI:     python unified_ai_agent.py --cli

4. 📖 View documentation:
   - See README.md for detailed usage instructions

Helpful commands:
-----------------
   python unified_ai_agent.py --help
   python unified_ai_agent.py --status

╔══════════════════════════════════════════════════════════════════════╗
║  Need help? Check README.md or run with --help flag                  ║
╚══════════════════════════════════════════════════════════════════════╝
""")

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(
        description="Setup utility for Unified AI Agent"
    )
    parser.add_argument(
        '--upgrade', 
        action='store_true',
        help='Upgrade all dependencies to latest versions'
    )
    parser.add_argument(
        '--skip-deps',
        action='store_true',
        help='Skip dependency installation'
    )
    parser.add_argument(
        '--env-only',
        action='store_true',
        help='Only create environment file and directories'
    )
    
    args = parser.parse_args()
    
    print_banner()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Environment-only mode
    if args.env_only:
        setup_environment_file()
        create_directories()
        print("\n✅ Environment setup complete!")
        sys.exit(0)
    
    # Install dependencies
    if not args.skip_deps:
        if not install_dependencies(upgrade=args.upgrade):
            print("\n❌ Dependency installation failed.")
            print("   Try running: pip install -r requirements.txt")
            sys.exit(1)
    
    # Test imports
    if not test_imports():
        print("\n⚠️  Some imports failed. The application may not work correctly.")
    
    # Create directories
    create_directories()
    
    # Setup environment file
    setup_environment_file()
    
    # Check API keys
    check_api_keys()
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main()
