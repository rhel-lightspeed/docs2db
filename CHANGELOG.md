# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- `db-status` now displays document paths instead of filenames in "Recent document activity" section for better identification in subdirectory structure

## [0.3.1] - 2025-11-06

### Changed
- **BREAKING**: `load` command now uses directory-based patterns (must end with `**` or `*`) instead of file patterns
  - Old: `--pattern "**/source.json"` or `--pattern "**/*.json"`
  - New: `--pattern "**"` or `--pattern "external/**"`
  - Patterns are automatically converted to match `source.json` files internally
- `load` command default pattern changed from `**/*.json` to `**` (directory-based)
- Pattern validation now checks for proper glob wildcard structure instead of file extensions

## [0.3.0] - 2025-11-06

### Added
- `document_needs_update()` function for external API callers to check if documents need updating without knowing internal storage details
- Enhanced audit reporting with separate tracking for stale chunks and stale embeddings
- Zero-chunk document detection in audit (documents that legitimately have no chunks)
- Orphan directory detection in audit (directories without `source.json`)
- Audit command now accepts directory patterns for targeted auditing (e.g., `--pattern "external/**"` or `--pattern "allowed_kcs/*"`)

### Changed
- **BREAKING**: File storage now uses subdirectory-based structure where each document gets its own directory:
  - Old: `doc_name.json`, `doc_name.chunks.json`, `doc_name.gran.json`
  - New: `doc_name/source.json`, `doc_name/chunks.json`, `doc_name/gran.json`
- **BREAKING**: `ingest_from_content()` now requires `stream_name` parameter (e.g., `"document.html"`) for proper format detection
- **BREAKING**: `ingest_file()` and `ingest_from_content()` now accept directory paths instead of file paths (e.g., `content_dir/doc_name` not `content_dir/doc_name/source.json`)
- **BREAKING**: Default glob patterns changed to `**/source.json` (for chunking) and `**/chunks.json` (for embedding)
- Audit command default pattern changed from `**/*.json` to `**` (directory-based pattern)
- Audit now validates that patterns don't include file extensions (must match directories only)
- Audit results now show separate counts for chunks, embeddings, and zero-chunk documents
- Moved routine document analysis logging to DEBUG level; summarization events still logged at INFO level
- **performance improvement**: LLM API clients (WatsonX, OpenAI) are now reused across documents in the same batch instead of creating new clients for each document
- `LLMSession.__init__()` no longer takes `doc_text` parameter; call `set_document(doc_text)` after initialization
- Added `LLM_PROVIDER` setting (defaults to `"openai"`) to explicitly choose between OpenAI-compatible and WatsonX providers; provider selection now respects Pydantic Settings precedence (CLI/env > .env file > defaults)
- Added `--llm-provider`, `--openai-url`, `--watsonx-url`, `--context-model`, and `--context-limit` CLI flags to `chunk` and `pipeline` commands to explicitly control LLM provider settings
- Provider is inferred from URL flags if not explicitly specified (e.g., `--watsonx-url` → `watsonx`, `--openai-url` → `openai`)
- Removed validation that prevented both `--openai-url` and `--watsonx-url` from being set; provider selection now explicit via `--llm-provider` or inferred from flags
- Settings access is now centralized in public API functions only (`generate_chunks()`, `generate_embeddings()`, `load_documents()`, `perform_audit()`); all internal functions require explicit parameters and validate inputs strictly (no global settings access)

### Fixed
- Addressed WatsonX rate limiting issues by reusing API clients across documents in worker batches
- Fixed race condition causing duplicate key errors when multiple workers insert the same embedding model concurrently (now uses `INSERT ... ON CONFLICT DO NOTHING`)
- Fixed `db-destroy` command to correctly parse project name from `postgres-compose.yml` instead of hardcoding "docs2db" (was failing to remove volumes for projects with different names)
- Fixed `ingest_file()` to enforce `.json` extension regardless of caller-provided path, ensuring downstream chunking/embedding tools always find correct files

## [0.2.1] - 2025-11-03

### Changed
- Improved warning messages: metadata parsing errors now include the file path for easier debugging
- Moved WatsonX dependencies from `dependency-groups` to `optional-dependencies` for proper PyPI extra support

### Fixed
- WatsonX extra (`docs2db[watsonx]`) now installable from PyPI

## [0.2.0] - 2025-11-03

### Added
- Document ingestion using Docling (`ingest` command) for PDF, DOCX, PPTX, and more
- Contextual chunking with LLM support (Ollama via OpenAI-compatible API, OpenAI, WatsonX)
- BM25 full-text search with PostgreSQL tsvector and GIN indexing for hybrid search
- Database lifecycle commands: `db-start`, `db-stop`, `db-destroy`, `db-logs`
- `db-restore` command for loading SQL dumps
- `pipeline` command for end-to-end workflow (ingest → chunk → embed → load → dump)
- Multi-tier PostgreSQL configuration precedence (CLI > Env Vars > DATABASE_URL > Compose > Defaults)
- Metadata arguments for pipeline and load commands (`--username`, `--title`, `--description`, `--note`)
- Metadata tracking for ingested documents and chunking operations
- `--skip-context` flag to bypass LLM contextual chunking
- `--context-model` and `--openai-url`/`--watsonx-url` flags for LLM provider configuration
- Persistent LLM sessions with KV cache reuse for improved performance
- Memory-efficient in-memory document ingestion
- Comprehensive database configuration tests
- Pre-commit hooks for code quality enforcement (ruff, pyright, gitleaks)

### Changed
- Default content directory changed from `content/` to `docs2db_content/`
- Commands now use settings defaults: `load`, `audit`, and `pipeline` fall back to `settings.content_base_dir` and `settings.embedding_model`
- Simplified database lifecycle: removed profile parameter (always uses "prod")
- Improved error messages: database connection errors now suggest `docs2db db-start` instead of `make db-up`
- Reduced logging verbosity: suppressed verbose docling library output, moved per-file conversion messages to DEBUG
- Updated `.gitignore` to exclude generated artifacts (`docs2db_content/`, `ragdb_dump.sql`)
- Improved CLI argument handling with explicit None checks and user-friendly error messages

### Fixed
- Typer required argument handling now provides clear error messages instead of TypeErrors
- Removed duplicate error logging in database operations
- Updated compose file password to match default settings (`postgres`)
- Corrected ingest command docstring to show `docs2db_content/` directory

## [0.1.0] - 2025-09-29

### Added
- Initial implementation of docs2db
- Basic document chunking using HybridChunker from docling_core
- Embedding generation with Granite 30M English model
- PostgreSQL database with pgvector for vector similarity search
- CLI commands: `chunk`, `embed`, `load`, `audit`, `db-status`, `db-dump`, `cleanup-workers`
- Multiprocessing support for chunking and embedding operations
- Comprehensive test suite
- Development tooling: Makefile, Docker Compose setup for PostgreSQL

[Unreleased]: https://github.com/rhel-lightspeed/docs2db/compare/v0.3.1...HEAD
[0.3.1]: https://github.com/rhel-lightspeed/docs2db/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/rhel-lightspeed/docs2db/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/rhel-lightspeed/docs2db/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/rhel-lightspeed/docs2db/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/rhel-lightspeed/docs2db/releases/tag/v0.1.0
