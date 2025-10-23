# Document Metadata

Docs2DB generates `.meta.json` files alongside Docling JSON documents to store metadata about the source document, its processing, and optional user-supplied provenance.

## Location and Naming

Metadata files are stored adjacent to their corresponding Docling JSON files:
```
content/
  docs/
    guide.json          # Docling document
    guide.meta.json     # Metadata for guide.json
```

## File Structure

The metadata file is **sparse** - only fields with actual data are included. Empty sections are omitted.

### Example: Full Metadata
```json
{
  "metadata_version": "1.0",

  "filesystem": {
    "original_path": "/sources/docs/guide.html",
    "size_bytes": 245680,
    "mtime": "2025-10-23T10:30:00Z",
    "detected_mime": "text/html"
  },

  "content": {
    "title": "RHEL 9.4 Administration Guide",
    "language": "en"
  },

  "source": {
    "source_type": "graphql",
    "source_url": "https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/9.4/html/system_administrators_guide/index",
    "source_etag": "abc123def456",
    "retrieved_at": "2025-10-23T10:30:00Z",
    "retriever": "codex-documentation-v1.0",
    "license": "CC-BY-SA-4.0"
  },

  "processing": {
    "source_hash": "xxh64:a1b2c3d4e5f6...",
    "ingested_at": "2025-10-23T10:31:00Z",
    "docling_version": "2.42.1"
  }
}
```

### Example: Minimal Metadata (Auto-detected only)
```json
{
  "metadata_version": "1.0",
  "filesystem": {
    "size_bytes": 12540
  },
  "processing": {
    "ingested_at": "2025-10-23T10:31:00Z",
    "docling_version": "2.42.1"
  }
}
```

## Field Descriptions

### Top Level
- `metadata_version`: Schema version for the metadata format (e.g., `"1.0"`)

### `filesystem` (Auto-detected)
Information about the source file on disk:
- `original_path`: Full path to the original source file
- `size_bytes`: File size in bytes
- `mtime`: Last modification time (ISO 8601 format)
- `detected_mime`: MIME type detected from file extension

### `content` (Auto-detected)
Information extracted from the document content:
- `title`: Document title (from Docling's `name` field)
- `language`: Document language code (e.g., `"en"`, `"es"`)

### `source` (User-supplied)
Provenance information supplied by external tools (e.g., Codex retrievers):
- `source_type`: Type of source (e.g., `"graphql"`, `"web"`, `"local"`)
- `source_url`: Original URL where the document was retrieved
- `source_etag`: ETag or version identifier from the source
- `retrieved_at`: Timestamp when the document was retrieved (ISO 8601)
- `retriever`: Tool/version that retrieved the document
- `license`: Known license for the content (e.g., `"CC-BY-SA-4.0"`, `"Apache-2.0"`)
- Custom fields as needed by retrievers

### `processing` (Auto-generated)
Information about how the document was processed:
- `source_hash`: xxHash (xxh64) of the source file (format: `"xxh64:..."`) - fast non-cryptographic hash
- `ingested_at`: Timestamp when ingestion occurred (ISO 8601)
- `docling_version`: Version of Docling used for conversion

## Usage

### During Ingestion

The `ingest` command automatically generates metadata for all processed documents:
```bash
uv run docs2db ingest /path/to/sources
```

This creates:
- `content/**/*.json` (Docling documents)
- `content/**/*.meta.json` (metadata)

### Supplying User Metadata

External tools can supply metadata by calling the `generate_metadata` function:

```python
from docs2db.ingest import generate_metadata

source_metadata = {
    "source_type": "graphql",
    "source_url": "https://example.com/doc",
    "source_etag": "abc123",
    "retrieved_at": "2025-10-23T10:30:00Z",
    "retriever": "my-retriever-v1.0",
    "license": "MIT"
}

generate_metadata(
    source_file=Path("/sources/doc.html"),
    content_path=Path("content/doc.json"),
    source_metadata=source_metadata
)
```

### Auditing Metadata

The `audit` command checks metadata files:
```bash
uv run docs2db audit
```

This reports:
- Total metadata files
- Orphaned metadata (no corresponding Docling JSON)
- Version mismatches (outdated schema)
- Invalid JSON files
