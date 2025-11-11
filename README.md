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

**Requirements:**
- Podman or Docker (for database management)
- Ollama (optional, needed for contextual chunking which is on by default; turn off with `--skip-context`)

## Quickstart

**One command:**
```bash
docs2db pipeline /path/to/your/documents
```

This starts a database, processes everything, and creates `ragdb_dump.sql`.

**Next steps:** See [docs2db-api](https://github.com/rhel-lightspeed/docs2db-api) to use your database for RAG search. Follow one of its demos to use it with Llama Stack or integrate it into your agent.

## Overview

**Processing time:** For large document sets (10,000+ documents), processing can take hours to days. Don't worry, the process is **resumable**. All intermediate files are saved locally in `docs2db_content/`, so if processing is interrupted, restarting will automatically skip files that are already complete.

**Database:** Docs2DB uses PostgreSQL with pgvector to build your RAG database. When you run the omnibus `pipeline` command it automatically creates a temporary database server using Podman/Docker, processes your documents, creates a `.sql` dump file, and then destroys the temporary database. You may never notice the database server running.

**Processing stages:** The `pipeline` command runs four stages:
1. **Ingest** - Convert documents to Docling JSON format
2. **Chunk** - Break documents into searchable sections (with optionally skippable LLM-generated context)
3. **Embed** - Generate vector embeddings for each chunk
4. **Load** - Insert everything into PostgreSQL

Each stage can also be run individually (`ingest`, `chunk`, `embed`, `load`). All stages are **idempotent** - they check file timestamps and hashes to skip work that's already been done.

**Performance tip:** The slowest stage is contextual chunking, where an LLM (via Ollama or WatsonX) generates context for each document section. This improves search quality but can be skipped for faster build-time processing using `--skip-context` with either `pipeline` or `chunk`. Skipping context also means you don't need Ollama installed. Note that skipping context may reduce search accuracy. Search time is unaffected whether contextual chunking is used or not.

**Storage:** Intermediate files in `docs2db_content/` can be substantial (roughly 2-3x your source document size). Keep this directory, it is valuable for incremental updates and should be committed to version control.

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
docs2db db-start      # Start PostgreSQL (Podman/Docker)
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
  --model intfloat/e5-small-v2       # Different embedding model
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
docs2db chunk --pattern "docs/**"
docs2db chunk --content-dir my-content
```

Configuration via environment variables or `.env` file also supported. Run `docs2db chunk --help` for all options.

### Embedding

```bash
# Different model
docs2db embed --model granite-30m-english

# Patterns and directories
docs2db embed --pattern "docs/**"
docs2db embed --content-dir my-content
```

Run `docs2db embed --help` for all options.

## Content Directory

The content directory (default: `docs2db_content/`) stores intermediate processing files. The directory structure mirrors your source document hierarchy - each source file gets its own subdirectory:

```
docs2db_content/
├── path/
│   └── to/
│       └── your/
│           └── document/
│               ├── source.json      # Docling ingested document
│               ├── chunks.json      # Text chunks with context
│               ├── gran.json        # Granite embeddings
│               └── meta.json        # Processing metadata
└── README.md
```

**Files per document:**
- `source.json` - Ingested document in Docling JSON format
- `chunks.json` - Text chunks (with optional LLM-generated context)
- `gran.json` - Vector embeddings (filename varies by model: `slate.json`, `e5sm.json`, etc.)
- `meta.json` - Processing metadata and timestamps

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

### "Neither Podman nor Docker found"
Install Podman (https://podman.io/getting-started/installation) or Docker (https://docs.docker.com/get-docker/)

### "Database connection refused"
```bash
docs2db db-start      # Start the database
docs2db db-status     # Check connection
```

## Using as a Library

```bash
uv add docs2db
```

```python
from pathlib import Path
from docs2db.ingest import ingest_file, ingest_from_content

# Ingest a file from disk
ingest_file(
    source_file=Path("document.pdf"),
    content_path=Path("docs2db_content/my_docs/document"),
    source_metadata={"source": "my_system", "retrieved_at": "2024-01-01"}  # optional
)

# Ingest content from memory (HTML, markdown, etc.)
ingest_from_content(
    content="<html>...</html>",
    content_path=Path("docs2db_content/my_docs/page"),
    stream_name="page.html",
    source_metadata={"url": "https://example.com"},  # optional
    content_encoding="utf-8"  # optional, defaults to "utf-8"
)
```

Both functions convert documents to Docling JSON format and save to `content_path/source.json`. Use the CLI commands (`chunk`, `embed`, `load`) to process the ingested documents.

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

Use [docs2db-api](https://github.com/rhel-lightspeed/docs2db-api) to query your RAG database with hybrid search (vector + BM25) and reranking.

**As a Python library:**

```python
from docs2db_api.rag.engine import UniversalRAGEngine, RAGConfig

config = RAGConfig(similarity_threshold=0.7, max_chunks=5)
engine = UniversalRAGEngine(config=config)
await engine.start()  # Auto-detects database and embedding model
results = await engine.search_documents("How do I configure authentication?")
```

**As a REST API or Llama Stack integration:**

See the [docs2db-api repository](https://github.com/rhel-lightspeed/docs2db-api) for FastAPI REST server and Llama Stack adapter examples.

## License

See [LICENSE](LICENSE) for details.
