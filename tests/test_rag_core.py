"""
Core RAG Functionality Tests
============================

Focused tests for the core RAG engine functionality.
These tests ensure the basic RAG search works correctly.
"""

import pytest
import pytest_asyncio


class TestRAGCore:
    """Test core RAG engine functionality"""

    @pytest.mark.asyncio
    async def test_basic_search_functionality(self):
        """Test that basic RAG search returns results"""
        from docs2db.rag.engine import search_documents

        result = await search_documents(
            "How to configure SSH?",
            model_name="granite-30m-english",
            max_chunks=3,
            similarity_threshold=0.5,
        )

        # Basic result validation
        assert result is not None
        assert hasattr(result, "query")
        assert hasattr(result, "documents")
        assert result.query == "How to configure SSH?"
        assert isinstance(result.documents, list)
        assert len(result.documents) <= 3

        # Validate document structure
        if result.documents:
            doc = result.documents[0]
            assert "text" in doc
            assert "similarity_score" in doc
            assert "document_path" in doc
            assert isinstance(doc["similarity_score"], float)
            assert 0.0 <= doc["similarity_score"] <= 1.0

    @pytest.mark.asyncio
    async def test_different_models(self):
        """Test that different embedding models work"""
        from docs2db.rag.engine import search_documents

        # Test with the default granite model
        result = await search_documents(
            "user management",
            model_name="granite-30m-english",
            max_chunks=2,
            similarity_threshold=0.6,
        )

        assert result is not None
        assert len(result.documents) <= 2

    @pytest.mark.asyncio
    async def test_similarity_threshold(self):
        """Test that similarity threshold filtering works"""
        from docs2db.rag.engine import search_documents

        # High threshold should return fewer results
        high_threshold_result = await search_documents(
            "test query",
            model_name="granite-30m-english",
            max_chunks=10,
            similarity_threshold=0.9,
        )

        # Low threshold should return more results
        low_threshold_result = await search_documents(
            "test query",
            model_name="granite-30m-english",
            max_chunks=10,
            similarity_threshold=0.3,
        )

        # Low threshold should return at least as many results as high threshold
        assert len(low_threshold_result.documents) >= len(
            high_threshold_result.documents
        )

    @pytest.mark.asyncio
    async def test_max_chunks_limit(self):
        """Test that max_chunks parameter limits results correctly"""
        from docs2db.rag.engine import search_documents

        # Test with limit of 1
        result_1 = await search_documents(
            "configuration",
            model_name="granite-30m-english",
            max_chunks=1,
            similarity_threshold=0.5,
        )

        # Test with limit of 5
        result_5 = await search_documents(
            "configuration",
            model_name="granite-30m-english",
            max_chunks=5,
            similarity_threshold=0.5,
        )

        assert len(result_1.documents) <= 1
        assert len(result_5.documents) <= 5
        assert len(result_5.documents) >= len(result_1.documents)
