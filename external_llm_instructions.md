# External LLM Configuration Guide

Database Copilot supports multiple external LLM providers that can provide better quality reviews and recommendations, especially for complex database migrations.

## Supported Providers

- OpenAI (GPT-4, GPT-3.5)
- Anthropic Claude
- Google Gemini
- Mistral AI
- DeepSeek

## Configuration Methods

### 1. Using Streamlit Secrets (Recommended)

1. Edit the `.streamlit/secrets.toml` file in the project root directory
2. Uncomment and set the appropriate values for your chosen LLM provider
3. Save the file and restart the application

Example `.streamlit/secrets.toml` for OpenAI:

```toml
# External LLM Configuration
LLM_TYPE = "openai"
OPENAI_API_KEY = "your-openai-api-key"
```

Example for Claude:

```toml
# External LLM Configuration
LLM_TYPE = "claude"
ANTHROPIC_API_KEY = "your-anthropic-api-key"
```

### 2. Using Environment Variables

Set environment variables before running the application:

#### OpenAI

```bash
export LLM_TYPE=openai
export OPENAI_API_KEY=your-api-key
export OPENAI_MODEL=gpt-4o  # Optional, defaults to gpt-4o
```

#### Anthropic (Claude)

```bash
export LLM_TYPE=claude
export ANTHROPIC_API_KEY=your-api-key
export CLAUDE_MODEL=claude-3-opus-20240229  # Optional
```

#### Google (Gemini)

```bash
export LLM_TYPE=gemini
export GOOGLE_API_KEY=your-api-key
export GEMINI_MODEL=gemini-1.5-pro  # Optional
```

#### Mistral AI

```bash
export LLM_TYPE=mistral
export MISTRAL_API_KEY=your-api-key
export MISTRAL_MODEL=mistral-medium  # Optional
```

#### DeepSeek

```bash
export LLM_TYPE=deepseek
export DEEPSEEK_API_KEY=your-api-key
export DEEPSEEK_MODEL=deepseek-chat  # Optional
```

## Required Packages

Each external LLM provider requires specific Python packages:

```bash
# Install all supported providers
pip install openai anthropic google-generativeai mistralai
```

These packages are included in the project's requirements files and should be automatically installed when you run the setup script:

```bash
python setup.py
```

If you're experiencing issues with missing packages, you can manually install them using the command above.

For more detailed information, refer to the full documentation in `docs/external_llm_setup.md`.
