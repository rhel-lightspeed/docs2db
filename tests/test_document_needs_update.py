"""Tests for document_needs_update function."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from docs2db.ingest import document_needs_update, ingest_file, ingest_from_content


@pytest.fixture
def test_content_dir(tmp_path):
    """Create a test content directory."""
    content_dir = tmp_path / "test_content"
    content_dir.mkdir()
    return content_dir


@pytest.fixture
def sample_html_file(tmp_path):
    """Create a sample HTML file."""
    html_file = tmp_path / "test.html"
    html_file.write_text("<html><body><h1>Test</h1><p>Content</p></body></html>")
    return html_file


def test_document_needs_update_nonexistent(test_content_dir):
    """Test that nonexistent document needs update."""
    doc_path = test_content_dir / "nonexistent_doc"
    assert document_needs_update(doc_path) is True


def test_document_needs_update_exists_no_params(test_content_dir, sample_html_file):
    """Test that existing document with no comparison params doesn't need update."""
    doc_path = test_content_dir / "test_doc"

    # Ingest the document
    ingest_file(sample_html_file, doc_path)

    # Check without comparison params - should be False (exists, nothing to compare)
    assert document_needs_update(doc_path) is False


def test_document_needs_update_missing_metadata(test_content_dir):
    """Test that document with missing metadata needs update when comparison requested."""
    doc_path = test_content_dir / "test_doc"
    doc_path.mkdir()

    # Create source.json but no meta.json
    source_file = doc_path / "source.json"
    source_file.write_text('{"name": "test"}')

    # Should need update because metadata is missing (when checking timestamp)
    assert (
        document_needs_update(doc_path, source_timestamp="2024-01-15T10:30:00Z") is True
    )


def test_document_needs_update_same_file_hash(test_content_dir, sample_html_file):
    """Test that document with same file hash doesn't need update."""
    doc_path = test_content_dir / "test_doc"

    # Ingest the document
    ingest_file(sample_html_file, doc_path)

    # Check with same source file - should be False (hash matches)
    assert document_needs_update(doc_path, source_file=sample_html_file) is False


def test_document_needs_update_different_file_hash(
    test_content_dir, sample_html_file, tmp_path
):
    """Test that document with different file hash needs update."""
    doc_path = test_content_dir / "test_doc"

    # Ingest the document with original file
    ingest_file(sample_html_file, doc_path)

    # Create a modified version of the file
    modified_file = tmp_path / "modified.html"
    modified_file.write_text(
        "<html><body><h1>Modified</h1><p>Different content</p></body></html>"
    )

    # Check with modified source file - should be True (hash differs)
    assert document_needs_update(doc_path, source_file=modified_file) is True


def test_document_needs_update_same_content_hash(test_content_dir):
    """Test that document with same content hash doesn't need update."""
    doc_path = test_content_dir / "test_doc"
    content = "<html><body><h1>Test</h1></body></html>"

    # Ingest from content
    ingest_from_content(
        content, doc_path, "test.html", source_metadata={"source_type": "test"}
    )

    # Check with same content - should be False (hash matches)
    assert document_needs_update(doc_path, content=content) is False


def test_document_needs_update_different_content_hash(test_content_dir):
    """Test that document with different content hash needs update."""
    doc_path = test_content_dir / "test_doc"
    original_content = "<html><body><h1>Original</h1></body></html>"

    # Ingest from content
    ingest_from_content(
        original_content, doc_path, "test.html", source_metadata={"source_type": "test"}
    )

    # Check with different content - should be True (hash differs)
    modified_content = "<html><body><h1>Modified</h1></body></html>"
    assert document_needs_update(doc_path, content=modified_content) is True


def test_document_needs_update_same_timestamp(test_content_dir):
    """Test that document with same timestamp doesn't need update."""
    doc_path = test_content_dir / "test_doc"
    timestamp = "2024-01-15T10:30:00Z"

    # Ingest with timestamp
    ingest_from_content(
        "<html><body>Test</body></html>",
        doc_path,
        "test.html",
        source_metadata={"modified": timestamp, "source_type": "test"},
    )

    # Check with same timestamp - should be False (timestamp matches)
    assert document_needs_update(doc_path, source_timestamp=timestamp) is False


def test_document_needs_update_different_timestamp(test_content_dir):
    """Test that document with different timestamp needs update."""
    doc_path = test_content_dir / "test_doc"
    old_timestamp = "2024-01-15T10:30:00Z"

    # Ingest with old timestamp
    ingest_from_content(
        "<html><body>Test</body></html>",
        doc_path,
        "test.html",
        source_metadata={"modified": old_timestamp, "source_type": "test"},
    )

    # Check with newer timestamp - should be True (timestamp differs)
    new_timestamp = "2024-01-16T10:30:00Z"
    assert document_needs_update(doc_path, source_timestamp=new_timestamp) is True


def test_document_needs_update_timestamp_but_no_stored(test_content_dir):
    """Test that document needs update when we have timestamp but it wasn't stored."""
    doc_path = test_content_dir / "test_doc"

    # Ingest without timestamp in metadata
    ingest_from_content(
        "<html><body>Test</body></html>",
        doc_path,
        "test.html",
        source_metadata={"source_type": "test"},  # No modified field
    )

    # Check with timestamp - should be True (no stored timestamp to compare)
    assert (
        document_needs_update(doc_path, source_timestamp="2024-01-15T10:30:00Z") is True
    )


def test_document_needs_update_bytes_content(test_content_dir):
    """Test document_needs_update with bytes content."""
    doc_path = test_content_dir / "test_doc"
    content_bytes = b"<html><body>Test</body></html>"

    # Ingest from bytes content
    ingest_from_content(
        content_bytes, doc_path, "test.html", source_metadata={"source_type": "test"}
    )

    # Check with same bytes content - should be False
    assert document_needs_update(doc_path, content=content_bytes) is False

    # Check with different bytes content - should be True
    different_bytes = b"<html><body>Different</body></html>"
    assert document_needs_update(doc_path, content=different_bytes) is True


def test_document_needs_update_multiple_checks_all_pass(
    test_content_dir, sample_html_file
):
    """Test that document doesn't need update when all checks pass."""
    doc_path = test_content_dir / "test_doc"
    timestamp = "2024-01-15T10:30:00Z"

    # Ingest with timestamp
    ingest_file(sample_html_file, doc_path, source_metadata={"modified": timestamp})

    # Check with both matching file and timestamp - should be False
    assert (
        document_needs_update(
            doc_path, source_file=sample_html_file, source_timestamp=timestamp
        )
        is False
    )


def test_document_needs_update_multiple_checks_one_fails(
    test_content_dir, sample_html_file
):
    """Test that document needs update when any check fails."""
    doc_path = test_content_dir / "test_doc"
    old_timestamp = "2024-01-15T10:30:00Z"

    # Ingest with old timestamp
    ingest_file(sample_html_file, doc_path, source_metadata={"modified": old_timestamp})

    # Check with matching file but different timestamp - should be True
    new_timestamp = "2024-01-16T10:30:00Z"
    assert (
        document_needs_update(
            doc_path, source_file=sample_html_file, source_timestamp=new_timestamp
        )
        is True
    )


def test_document_needs_update_corrupted_metadata(test_content_dir):
    """Test that document with corrupted metadata needs update when comparison requested."""
    doc_path = test_content_dir / "test_doc"
    doc_path.mkdir()

    # Create source.json
    source_file = doc_path / "source.json"
    source_file.write_text('{"name": "test"}')

    # Create corrupted meta.json
    meta_file = doc_path / "meta.json"
    meta_file.write_text("{ invalid json")

    # Should need update because metadata is corrupted (when checking timestamp)
    assert (
        document_needs_update(doc_path, source_timestamp="2024-01-15T10:30:00Z") is True
    )
