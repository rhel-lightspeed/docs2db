# Docs2DB

Build a RAG database from documents. Docs2DB processes documents into chunks and embeddings, loads them into PostgreSQL with pgvector, and produces portable SQL dumps.

**What it does:**
- Ingests documents (PDF, DOCX, XLSX, HTML, MD, CSV, etc.) using Docling
- Generates contextual chunks with LLM assistance
- Creates embeddings (Granite 30M by default)
- Loads into PostgreSQL with pgvector
- Produces portable `ragdb_dump.sql` files

**What it's for:**
- Creating databases for RAG systems that use [docs2db-api](https://github.com/rhel-lightspeed/docs2db-api)

## Installation

```bash
uv tool install docs2db
```

**Requirements:** Docker or Podman (for database management)

## Quickstart

**One command:**
```bash
docs2db pipeline /path/to/your/documents
```

This starts a database, processes everything, and creates `ragdb_dump.sql`.

**Next steps:** See [docs2db-api](https://github.com/rhel-lightspeed/docs2db-api) to use your database for RAG search. Follow one of its demos to use it with Llama Stack or integrate it into your agent.

## Database Configuration

**Configuration precedence (highest to lowest):**
1. CLI arguments: `--host`, `--port`, `--db`, `--user`, `--password`
2. Environment variables: `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`
3. `DATABASE_URL`: `postgresql://user:pass@host:port/database`
4. `postgres-compose.yml` in current directory
5. Defaults: `localhost:5432`, user=`postgres`, password=`postgres`, db=`ragdb`

**Examples:**

```bash
# Use defaults (docs2db db-start creates everything)
docs2db load

# Environment variables
export POSTGRES_HOST=prod.example.com
export POSTGRES_DB=mydb
docs2db load

# DATABASE_URL (cloud providers)
export DATABASE_URL="postgresql://user:pass@host:5432/db"
docs2db load

# CLI arguments
docs2db load --host localhost --db mydb
```

**Note:** Don't mix `DATABASE_URL` with individual `POSTGRES_*` variables.

## Commands

### Database Lifecycle

```bash
docs2db db-start      # Start PostgreSQL (Docker/Podman)
docs2db db-stop       # Stop PostgreSQL
docs2db db-logs       # View logs (-f to follow)
docs2db db-destroy    # Delete all data (prompts for confirmation)
docs2db db-status     # Check connection and stats
```

### Pipeline

```bash
docs2db pipeline <path>              # Complete workflow
docs2db pipeline <path> \
  --output-file my-rag.sql \         # Custom output
  --skip-context \                   # Skip contextual chunks (faster)
  --model e5-small-v2                # Different embedding model
```

### Individual Steps

These are the same steps `pipeline` runs.

```bash
docs2db ingest <path>                # Ingest documents
docs2db chunk                        # Generate chunks
docs2db embed                        # Generate embeddings
docs2db load                         # Load into database
docs2db db-dump                      # Create SQL dump
docs2db db-restore <file>            # Restore from dump
docs2db audit                        # Check content directory
```

Each processing step (`ingest`, `chunk`, `embed`) creates files in `docs2db_content/` that the next step reads.

## Processing Options

### Chunking

```bash
# Fast (skip contextual generation)
docs2db chunk --skip-context

# Custom LLM provider
docs2db chunk --context-model qwen2.5:7b-instruct              # Ollama
docs2db chunk --openai-url https://api.openai.com \           # OpenAI
  --context-model gpt-4o-mini
docs2db chunk --watsonx-url https://us-south.ml.cloud.ibm.com # WatsonX

# Patterns and directories
docs2db chunk --pattern "docs/**/*.json"
docs2db chunk --content-dir my-content
```

Configuration via environment variables or `.env` file also supported. Run `docs2db chunk --help` for all options.

### Embedding

```bash
# Different model
docs2db embed --model granite-30m-english

# Patterns and directories
docs2db embed --pattern "docs/**/*.chunks.json"
docs2db embed --content-dir my-content
```

Run `docs2db embed --help` for all options.

## Content Directory

The content directory (default: `docs2db_content/`) stores:
- Ingested documents in Docling JSON format
- `.chunks.json` files with text chunks
- `.gran.json` files with embeddings

**Important:** Commit this directory to version control. It contains expensive preprocessing that can be reused across updates. Docs2DB automatically skips files that haven't changed.

## RAG Features

- **Contextual chunks** - LLM-generated context for each chunk ([Anthropic's approach](https://www.anthropic.com/engineering/contextual-retrieval))
- **Vector embeddings** - Multiple models: granite-30m, e5-small-v2, slate-125m, noinstruct-small
- **Full-text search** - PostgreSQL tsvector with GIN indexing for BM25
- **Vector similarity** - pgvector extension with HNSW indexes
- **Schema versioning** - Track metadata and schema changes
- **Portable dumps** - Self-contained SQL files that work anywhere with [docs2db-api](https://github.com/rhel-lightspeed/docs2db-api)
- **Incremental processing** - Automatically skips unchanged files

## Troubleshooting

### "Neither Docker nor Podman found"
Install Docker (https://docs.docker.com/get-docker/) or Podman (https://podman.io/getting-started/installation)

### "Database connection refused"
```bash
docs2db db-start      # Start the database
docs2db db-status     # Check connection
```

### "Module not found" errors
Use `uv tool install docs2db`

## Using as a Library

```bash
uv add docs2db
```

```python
from docs2db import ingest_file, ingest_from_content

# Your code here
```

See `docs2db --help` for the full Python API.

## Development

```bash
git clone https://github.com/rhel-lightspeed/docs2db
cd docs2db
uv sync
pre-commit install

# Run tests
make test

# Run all checks
pre-commit run --all-files
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## Serving Your Database

Use [docs2db-api](https://github.com/rhel-lightspeed/docs2db-api) to serve your RAG database with a REST API.

## License

See [LICENSE](LICENSE) for details.
