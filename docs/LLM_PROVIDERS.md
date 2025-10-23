# External LLM Provider Configuration

Docs2DB supports OpenAI-compatible APIs and IBM WatsonX for contextual chunk generation. This guide shows how to configure different providers.

## Quick Start

The chunking system uses these key parameters:
- `--skip-context`: Disable contextual generation entirely (fastest, lower quality)
- `--context-model`: Model name/identifier for the LLM
- `--openai-url`: URL for OpenAI-compatible API (Ollama, OpenAI, etc.)
- `--watsonx-url`: URL for IBM WatsonX API (mutually exclusive with `--openai-url`)

## Local (Ollama)

**Default configuration** - no flags needed:

```bash
uv run docs2db chunk
```

This uses:
- Model: `qwen2.5:7b-instruct`
- URL: `http://localhost:11434` (default Ollama)

### Faster local models:

```bash
# 3B model (2-3x faster)
uv run docs2db chunk --context-model qwen2.5:3b-instruct

# 1.5B model (4-5x faster, may be lower quality)
uv run docs2db chunk --context-model qwen2.5:1.5b-instruct

# Alternative fast models
uv run docs2db chunk --context-model llama3.2:3b-instruct
uv run docs2db chunk --context-model gemma2:2b-instruct

# Custom Ollama URL
uv run docs2db chunk --openai-url "http://localhost:11434" --context-model qwen2.5:7b-instruct
```

## WatsonX

WatsonX provides IBM's Granite and other models. Authentication requires an API key and project ID.

### Setup:

1. Get your API key and project ID from IBM Cloud
2. Set environment variables:
   ```bash
   export WATSONX_API_KEY="your-api-key-here"
   export WATSONX_PROJECT_ID="your-project-id-here"
   ```

3. Run chunking:
   ```bash
   uv run docs2db chunk \
     --watsonx-url "https://us-south.ml.cloud.ibm.com" \
     --context-model "ibm/granite-13b-chat-v2"
   ```

### Available WatsonX models:
- `ibm/granite-13b-chat-v2` - Good balance
- `ibm/granite-20b-multilingual` - Larger, more capable
- `meta-llama/llama-3-70b-instruct` - Very capable, slower
- `mistralai/mixtral-8x7b-instruct-v01` - Fast and good quality

### Region URLs:
- US South: `https://us-south.ml.cloud.ibm.com/ml/v1`
- EU Germany: `https://eu-de.ml.cloud.ibm.com/ml/v1`
- Japan Tokyo: `https://jp-tok.ml.cloud.ibm.com/ml/v1`

## OpenAI

### Setup:

1. Get your API key from https://platform.openai.com/api-keys
2. Set environment variable:
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

3. Run chunking:
   ```bash
   uv run docs2db chunk \
     --openai-url "https://api.openai.com" \
     --context-model "gpt-4o-mini"
   ```

### Recommended OpenAI models:
- `gpt-4o-mini` - Fast, cost-effective (recommended)
- `gpt-4o` - More capable, slower, more expensive
- `gpt-3.5-turbo` - Cheapest, decent quality

## Authentication

Authentication is handled via environment variables:

**Required for WatsonX:**
- `WATSONX_API_KEY` - Your IBM Cloud API key
- `WATSONX_PROJECT_ID` - Your WatsonX project ID

## Performance Considerations

Contextual chunking speed depends on:
1. **Model size** - Smaller models (1.5B-3B) are 2-5x faster than 7B+ models
2. **API latency** - Local Ollama is fastest, external APIs add network overhead
3. **Document size** - Larger documents may hit context limits
4. **Cost** - External APIs charge per token (input + output)

### Recommendations:

**For development/testing:**
```bash
# Skip context generation entirely
uv run docs2db chunk --skip-context
```

**For production with speed priority:**
```bash
# Use local small model
uv run docs2db chunk --context-model qwen2.5:3b-instruct
```

**For production with quality priority:**
```bash
# Use capable cloud model
uv run docs2db chunk \
  --llm-base-url "https://api.openai.com" \
  --context-model "gpt-4o-mini"
```

## Troubleshooting

### "Connection refused" error
- For local: Ensure Ollama is running (`ollama serve`)
- For external: Check your internet connection and firewall

### "Unauthorized" or "Authentication failed"
- Verify your API key is set: `echo $OPENAI_API_KEY`
- Check the API key is valid in your provider's dashboard
- Ensure the environment variable is exported, not just set

### "Model not found"
- Verify the model name matches your provider's catalog
- For Ollama: Pull the model first (`ollama pull qwen2.5:3b-instruct`)

### Slow performance
- Try a smaller model (`--context-model qwen2.5:3b-instruct`)
- Use `--skip-context` for fastest processing
- Check if your provider has rate limits

### Context length exceeded
- This happens with very large documents
- Currently no automatic handling - document splitting will be added in a future update
- Workaround: Use `--skip-context` or process smaller documents

## Cost Estimation

For external APIs, contextual chunking costs depend on:
- **Input tokens**: Document size + chunk size (per chunk)
- **Output tokens**: ~50-100 tokens of context per chunk

Example for a 100-page document (~50K tokens) with 200 chunks:
- Total input: ~200 × (50K + 500) ≈ 10M tokens
- Total output: ~200 × 75 ≈ 15K tokens

At OpenAI gpt-4o-mini pricing ($0.15/$0.60 per 1M tokens):
- Input cost: 10M × $0.15 = $1.50
- Output cost: 0.015M × $0.60 = $0.01
- **Total: ~$1.51**

**Note**: Document context is cached per-document with Ollama, reducing effective input tokens significantly. External APIs may or may not support caching depending on the provider.
