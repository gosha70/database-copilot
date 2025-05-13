# External LLM Setup Guide

This guide explains how to set up and use external LLMs with Database Copilot.

## Overview

Database Copilot supports multiple external LLM providers:

- OpenAI (GPT-4, GPT-3.5)
- Anthropic Claude
- Google Gemini
- Mistral AI
- DeepSeek

Using an external LLM can provide better quality reviews and recommendations, especially for complex database migrations. It also helps avoid dependency issues with local LLMs.

## Configuration

There are two ways to configure external LLMs with Database Copilot:

### 1. Using Streamlit Secrets (Recommended)

The recommended way to configure external LLMs is by using Streamlit's secrets management:

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

Alternatively, you can set environment variables before running the application:

1. Set the `LLM_TYPE` environment variable to specify which provider to use
2. Set the provider-specific API key environment variable
3. Optionally set the model name environment variable

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
export CLAUDE_MODEL=claude-3-opus-20240229  # Optional, defaults to claude-3-opus-20240229
```

#### Google (Gemini)

```bash
export LLM_TYPE=gemini
export GOOGLE_API_KEY=your-api-key
export GEMINI_MODEL=gemini-1.5-pro  # Optional, defaults to gemini-1.5-pro
```

#### Mistral AI

```bash
export LLM_TYPE=mistral
export MISTRAL_API_KEY=your-api-key
export MISTRAL_MODEL=mistral-medium  # Optional, defaults to mistral-medium
```

#### DeepSeek

```bash
export LLM_TYPE=deepseek
export DEEPSEEK_API_KEY=your-api-key
export DEEPSEEK_MODEL=deepseek-chat  # Optional, defaults to deepseek-chat
```

## Required Packages

Each external LLM provider requires specific Python packages:

- OpenAI: `pip install openai`
- Anthropic: `pip install anthropic`
- Google: `pip install google-generativeai`
- Mistral: `pip install mistralai`
- DeepSeek: `pip install openai` (DeepSeek uses OpenAI-compatible API)

You can install all supported providers with:

```bash
pip install openai anthropic google-generativeai mistralai
```

**Important**: If you're seeing errors about missing packages when trying to use external LLMs, make sure to install these packages first. The application will fail with a clear error message if the required packages are not installed.

## Verifying External LLM Usage

When an external LLM is successfully configured, you'll see a message in the logs:

```
USING EXTERNAL LLM: [provider] with model [model_name]
```

For example:

```
USING EXTERNAL LLM: openai with model gpt-4o
```

## Troubleshooting

If you encounter issues with external LLMs:

1. **API Key Issues**: Ensure your API key is valid and has not expired
2. **Network Issues**: Check your internet connection and firewall settings
3. **Package Issues**: Verify that you have installed the required packages for your chosen provider
4. **Rate Limiting**: Some providers have rate limits that may affect usage

Check the application logs for detailed error messages that can help diagnose the issue.

## Fallback Behavior

If an external LLM fails to initialize, the application will raise an error rather than falling back to a local LLM. This ensures that you're aware of configuration issues rather than getting potentially misleading results from a fallback model.
