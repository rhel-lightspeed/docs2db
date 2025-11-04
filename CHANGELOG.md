# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Moved routine document analysis logging to DEBUG level; summarization events still logged at INFO level
- **performance improvement**: LLM API clients (WatsonX, OpenAI) are now reused across documents in the same batch instead of creating new clients for each document
- `LLMSession.__init__()` no longer takes `doc_text` parameter; call `set_document(doc_text)` after initialization
- **Breaking change**: `LLM_OPENAI_URL` now defaults to `None` instead of `http://localhost:11434`; users must explicitly set either `LLM_OPENAI_URL` or `LLM_WATSONX_URL` when using contextual chunking

### Fixed
- Fixed WatsonX rate limiting issues by reusing API clients across documents in worker batches

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

[Unreleased]: https://github.com/rhel-lightspeed/docs2db/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/rhel-lightspeed/docs2db/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/rhel-lightspeed/docs2db/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/rhel-lightspeed/docs2db/releases/tag/v0.1.0
