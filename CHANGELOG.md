# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial PyPI package preparation
- `pipeline` command for end-to-end workflow (ingest → chunk → embed → load → dump)
- Database lifecycle commands: `db-start`, `db-stop`, `db-destroy`, `db-logs`
- Multi-tier PostgreSQL configuration precedence (CLI > Env Vars > DATABASE_URL > Compose > Defaults)
- Metadata arguments for pipeline and load commands (`--username`, `--title`, `--description`, `--note`)
- Comprehensive database configuration tests
- Pre-commit hooks for code quality enforcement

### Changed
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
- Document ingestion using Docling for PDF, DOCX, PPTX, and more
- Contextual chunking with LLM support (Ollama, OpenAI, WatsonX)
- Embedding generation with multiple model support
- PostgreSQL database with pgvector for vector similarity search
- CLI interface with commands: ingest, chunk, embed, load, audit, db-status, db-dump, db-restore
- Comprehensive test suite
- Development tooling: Makefile, pre-commit hooks, Docker Compose setup

[Unreleased]: https://github.com/rhel-lightspeed/docs2db/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/rhel-lightspeed/docs2db/releases/tag/v0.1.0
