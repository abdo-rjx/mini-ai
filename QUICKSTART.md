# Quick Start Guide - Unified AI Agent

Get up and running in 5 minutes!

## Step 1: Install (1 minute)

```bash
# Navigate to the project folder
cd unified-ai-agent

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Configure API Keys (1 minute)

Choose **at least one** API key:

### Option A: Environment Variables
```bash
export GEMINI_API_KEY="your-gemini-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

### Option B: .env File
```bash
cp .env.example .env
# Edit .env and add your keys
```

**Get your free API keys:**
- Gemini: https://makersuite.google.com/app/apikey
- Claude: https://console.anthropic.com/settings/keys

## Step 3: Run (1 minute)

```bash
# Web Interface (recommended)
python unified_ai_agent.py

# Or CLI mode
python unified_ai_agent.py --cli
```

Open your browser to `http://127.0.0.1:7860`

## Step 4: Add Documents (Optional, 2 minutes)

```bash
# Create books folder
mkdir books

# Copy your PDFs
cp ~/Documents/*.pdf books/

# Restart the app - documents will be indexed automatically
```

## That's It! 🎉

Try these example queries:
- "What agents are available?"
- "Explain Python async/await"
- "Summarize the key concepts from my documents"
- "Write a function to calculate fibonacci numbers"

## Common Commands

```bash
# Get help
python unified_ai_agent.py --help

# Run with custom settings
python unified_ai_agent.py --default-provider claude --books ./my_docs

# Create public link
python unified_ai_agent.py --share

# Run tests
python test_agent.py
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "No agents available" | Set GEMINI_API_KEY or ANTHROPIC_API_KEY |
| "Module not found" | Run `pip install -r requirements.txt` |
| "RAG not working" | Create `books/` folder and add PDFs |
| Port already in use | Use `--port 7861` to change port |

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Explore CLI commands with `/help` in CLI mode
- Check system status with `/status` command
