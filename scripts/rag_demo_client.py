#!/usr/bin/env python3
"""
RAG functionality demo client for docs2db.

IMPORTANT: This script requires docs2db-api to be installed.
- docs2db creates the database (ingest, chunk, embed, load)
- docs2db-api provides the RAG search engine

This demo script uses docs2db-api's UniversalRAGEngine to demonstrate:
1. Hybrid search (vector similarity + BM25 keyword search)
2. Question refinement for better query expansion
3. Cross-encoder reranking
4. Reciprocal Rank Fusion (RRF) scoring

Usage:
    # Basic test
    uv run python scripts/rag_demo_client.py

    # Custom query
    uv run python scripts/rag_demo_client.py --query "solar panels"

    # Interactive mode
    uv run python scripts/rag_demo_client.py --interactive

    # Without question refinement
    uv run python scripts/rag_demo_client.py --no-refine
"""

import asyncio
import sys

# Try to import docs2db-api
try:
    from docs2db_api.rag.engine import RAGConfig, UniversalRAGEngine
except ImportError:
    print("=" * 80)
    print("⚠️  ERROR: docs2db-api is not installed")
    print("=" * 80)
    print()
    print("This script requires the docs2db-api package for RAG functionality.")
    print()
    print("To install it from a local checkout:")
    print()
    print("    cd /path/to/docs2db")
    print("    uv add --editable /path/to/docs2db-api --dev")
    print()
    # TODO: Update message once docs2db-api is published to PyPI
    print("Or install from PyPI (when available):")
    print()
    print("    uv add docs2db-api --dev")
    print()
    print("=" * 80)
    sys.exit(1)


async def test_rag_query(
    query: str = "What is renewable energy?",
    limit: int = 5,
    threshold: float = 0.7,
    model: str = "granite-30m-english",
    enable_refinement: bool = True,
):
    """Test RAG functionality using the Universal RAG Engine.

    Args:
        query: The test query to search for
        limit: Maximum number of results to return
        threshold: Similarity threshold (0.0-1.0)
        model: Embedding model to use
        enable_refinement: Enable question refinement

    Returns:
        List of similar documents found
    """

    print("=" * 80)
    print("Testing RAG functionality with docs2db-api")
    print("=" * 80)
    print()

    # Configure RAG engine
    config = RAGConfig(
        model_name=model,
        similarity_threshold=threshold,
        max_chunks=limit,
        enable_question_refinement=enable_refinement,
    )

    print(f"📊 Configuration:")
    print(f"   Model: {model}")
    print(f"   Threshold: {threshold}")
    print(f"   Max results: {limit}")
    print(f"   Question refinement: {'enabled' if enable_refinement else 'disabled'}")
    print()

    # Create RAG engine
    engine = UniversalRAGEngine(config)

    try:
        print(f"🔍 Query: '{query}'")
        print()

        # Perform RAG search
        result = await engine.search_documents(query)

        # Display refined questions if enabled
        if result.refined_questions:
            print("🎯 Refined Questions:")
            for i, q in enumerate(result.refined_questions, 1):
                print(f"   {i}. {q}")
            print()

        # Display metadata
        if result.metadata:
            print("📈 Metadata:")
            for key, value in result.metadata.items():
                print(f"   {key}: {value}")
            print()

        # Display results
        print(f"📄 Found {len(result.documents)} documents:")
        print("=" * 80)

        for i, doc in enumerate(result.documents, 1):
            print()
            print(f"Result {i}:")
            print(f"   Document: {doc['document_path']}")
            print(f"   Chunk: {doc.get('chunk_index', 'unknown')}")
            print(f"   Overall Score: {doc['similarity_score']:.3f}")

            # Show detailed scores from hybrid search
            if "rrf_score" in doc:
                print(f"   RRF Score: {doc['rrf_score']:.3f}")
            if doc.get("vector_similarity") is not None:
                print(f"   Vector Similarity: {doc['vector_similarity']:.3f}")
            if doc.get("bm25_rank") is not None:
                print(f"   BM25 Rank: {doc['bm25_rank']:.3f}")

            print(f"   Text preview:")
            text_preview = doc["text"][:200]
            print(f"      {text_preview}...")

        if result.documents:
            print()
            print("=" * 80)
            print(f"✅ Successfully found {len(result.documents)} relevant documents")
            print("=" * 80)
            return result.documents
        else:
            print()
            print("=" * 80)
            print("⚠️  No documents found. Try:")
            print("   - Lowering the similarity threshold (--threshold)")
            print("   - Using a different query")
            print("   - Verifying the database contains embeddings")
            print("=" * 80)
            return []

    except Exception as e:
        print()
        print("=" * 80)
        print(f"❌ Error during RAG test: {e}")
        print("=" * 80)
        raise
    finally:
        await engine.close()


async def interactive_rag_test(
    model: str = "granite-30m-english",
    threshold: float = 0.7,
    enable_refinement: bool = True,
):
    """Interactive RAG testing - allows custom queries."""
    print()
    print("=" * 80)
    print("Interactive RAG Testing")
    print("=" * 80)
    print()
    print("Enter queries to test similarity search (or 'quit' to exit)")
    print()

    while True:
        try:
            query = input("Query> ").strip()
            if query.lower() in ["quit", "exit", "q"]:
                break
            if not query:
                continue

            await test_rag_query(
                query,
                limit=3,
                threshold=threshold,
                model=model,
                enable_refinement=enable_refinement,
            )
            print()

        except KeyboardInterrupt:
            print()
            print("Exiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test RAG functionality using docs2db-api"
    )
    parser.add_argument(
        "--query",
        "-q",
        default="What is renewable energy?",
        help="Query to test (default: 'What is renewable energy?')",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=5,
        help="Maximum number of results (default: 5)",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.7,
        help="Similarity threshold 0.0-1.0 (default: 0.7)",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="granite-30m-english",
        help="Embedding model to use (default: granite-30m-english)",
    )
    parser.add_argument(
        "--no-refine",
        action="store_true",
        help="Disable question refinement",
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Run in interactive mode"
    )

    args = parser.parse_args()

    enable_refinement = not args.no_refine

    if args.interactive:
        asyncio.run(
            interactive_rag_test(
                model=args.model,
                threshold=args.threshold,
                enable_refinement=enable_refinement,
            )
        )
    else:
        results = asyncio.run(
            test_rag_query(
                args.query, args.limit, args.threshold, args.model, enable_refinement
            )
        )
        sys.exit(0 if results else 1)
