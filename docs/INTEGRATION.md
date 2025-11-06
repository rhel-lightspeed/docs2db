# Integration Guide: Using Docs2DB as a Library

Docs2DB can be used as a library by external tools to leverage its ingestion and metadata generation capabilities.

## Key Function: `ingest_from_content()`

The `ingest_from_content()` function allows you to ingest in-memory content without saving intermediate files.

### Function Signature

```python
def ingest_from_content(
    content: str | bytes,
    content_path: Path,
    stream_name: str,
    source_metadata: dict[str, Any] | None = None,
    content_encoding: str = "utf-8",
) -> bool:
    """Convert in-memory content to Docling JSON and generate metadata.

    Args:
        content: The content to convert (HTML, markdown, etc). Can be string or bytes.
        content_path: Directory path where the document should be stored (source.json will be created inside).
        stream_name: Stream name with extension for docling to detect format (e.g., "doc.html", "article.md").
        source_metadata: Optional metadata about the source (URL, etag, license, etc).
        content_encoding: Encoding to use for string content. Defaults to "utf-8".

    Returns:
        bool: True if successful, False otherwise
    """
```

### Usage Example

```python
from pathlib import Path
from docs2db.ingest import ingest_from_content

# Prepare your content
html_content = "<html><body><h1>My Document</h1></body></html>"

# Build source metadata (optional but recommended)
source_metadata = {
    "source_type": "graphql",
    "source_url": "https://docs.example.com/...",
    "source_etag": "abc123",
    "retrieved_at": "2025-10-23T10:30:00Z",
    "retriever": "example-graphql-v1.0",
    "license": "CC-BY-SA-4.0",
}

# Ingest the content
success = ingest_from_content(
    content=html_content,
    content_path=Path("content/documentation/exampletech/9/guide"),
    stream_name="guide.html",  # Extension tells docling this is HTML
    source_metadata=source_metadata,
)

if success:
    print("✅ Document ingested successfully!")
    # Files created:
    #   - content/documentation/exampletech/9/guide/source.json (Docling JSON)
    #   - content/documentation/exampletech/9/guide/meta.json (Metadata)
```

## What Gets Created

When you call `ingest_from_content()`, two files are created in the specified directory:

### 1. Docling JSON (`source.json`)
The structured document representation created by Docling, containing:
- Document text and structure
- Layout information
- Extracted metadata (title, language, etc.)

### 2. Metadata file (`meta.json`)
A sparse, versioned metadata file containing:

```json
{
  "metadata_version": "1.0",

  "filesystem": {
    "original_path": "documentation/example/9/guide.json",
    "size_bytes": 12540
  },

  "content": {
    "title": "ExampleTech Installation Guide",
    "language": "en"
  },

  "source": {
    "source_type": "graphql",
    "source_url": "https://docs.example.com/...",
    "source_etag": "abc123",
    "retrieved_at": "2025-10-23T10:30:00Z",
    "retriever": "example-graphql-v1.0",
    "license": "CC-BY-SA-4.0"
  },

  "processing": {
    "source_hash": "xxh64:a1b2c3d4e5f6...",
    "ingested_at": "2025-10-23T10:31:00Z",
    "docling_version": "2.44.0"
  }
}
```

## Source Metadata Fields

The `source_metadata` dict can contain any fields you want to track. Common fields:

| Field | Description | Example |
|-------|-------------|---------|
| `source_type` | Type of retrieval system | `"graphql"`, `"web_scrape"`, `"filesystem"` |
| `source_url` | Original URL of the document | `"https://docs.example.com/..."` |
| `source_etag` | HTTP ETag for change detection | `"abc123"` |
| `retrieved_at` | Timestamp when retrieved | `"2025-10-23T10:30:00Z"` |
| `retriever` | Tool/version that retrieved it | `"example-graphql-v1.0"` |
| `license` | Known license for the content | `"CC-BY-SA-4.0"`, `"MIT"` |

You can add custom fields as needed for your retriever.

## Setting Up as a Dependency

### Option 1: Local Development (editable)

In your `pyproject.toml`:

```toml
[project]
dependencies = [
    "docs2db",
    # ... other dependencies
]

[tool.uv.sources]
docs2db = { path = "../docs2db", editable = true }
```

### Option 2: Git Repository

```toml
[project]
dependencies = [
    "docs2db @ git+https://github.com/rhel-lightspeed/docs2db.git",
]
```

### Option 3: PyPI (when published)

```toml
[project]
dependencies = [
    "docs2db>=0.1.0",
]
```
