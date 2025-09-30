#!/usr/bin/env python3
"""
RAG functionality test script for docs2db.

This script verifies that the database is ready for RAG
(Retrieval-Augmented Generation) by testing similarity
search capabilities. It can be used to:

1. Verify that the database contains embeddings
2. Test query embedding generation
3. Perform similarity searches
4. Validate RAG readiness

Usage:
    # Basic test
    uv run python tests/test_rag_functionality.py

    # Custom query
    uv run python tests/test_rag_functionality.py --query "solar panels"

    # Interactive mode
    uv run python tests/test_rag_functionality.py --interactive
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to the path so we can import docs2db modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from docs2db.database import DatabaseManager, get_db_config
from docs2db.embeddings import Embedding


async def test_rag_query(query: str = "What is renewable energy?", limit: int = 5):
    """Test RAG functionality by performing a similarity search.

    Args:
        query: The test query to search for
        limit: Maximum number of results to return

    Returns:
        List of similar chunks found
    """

    print("Testing RAG functionality...")

    # Get database config
    config = get_db_config()
    print(
        f"Connecting to database: {config['user']}@{config['host']}:{config['port']}/{config['database']}"
    )

    # Create database manager
    db_manager = DatabaseManager(
        host=config["host"],
        port=int(config["port"]),
        database=config["database"],
        user=config["user"],
        password=config["password"],
    )

    # Create embedding model
    embedding_model = Embedding.from_name("granite-30m-english")

    # Test query
    print(f"Query: '{query}'")

    try:
        # Generate embedding for the query
        print("Generating query embedding...")
        provider = embedding_model._get_provider()
        query_embeddings = provider.generate_embeddings([query])
        query_embedding = query_embeddings[0]

        print(f"Query embedding generated: {len(query_embedding)} dimensions")

        # Perform similarity search
        print("Searching for similar chunks...")
        similar_chunks = await db_manager.search_similar(
            query_embedding=query_embedding,
            model_name="granite-30m-english",
            limit=limit,
            similarity_threshold=0.1,  # Lower threshold to get some results
        )

        print(f"Found {len(similar_chunks)} similar chunks:")
        print("=" * 80)

        for i, chunk in enumerate(similar_chunks, 1):
            print(f"\nResult {i}:")
            print(f"   Document: {chunk['document_filename']}")
            print(f"   Similarity: {chunk['similarity']:.3f}")
            print(f"   Text preview: {chunk['text'][:200]}...")
            print(f"   Distance: {chunk['distance']:.3f}")

        if similar_chunks:
            print(f"\nFound {len(similar_chunks)} relevant chunks.")
            return similar_chunks
        else:
            print("No similar chunks found. Try lowering the similarity threshold.")
            return []

    except Exception as e:
        print(f"Error during RAG test: {e}")
        raise


async def interactive_rag_test():
    """Interactive RAG testing - allows custom queries."""
    print("\nInteractive RAG Testing")
    print("Enter queries to test similarity search (or 'quit' to exit)")

    while True:
        try:
            query = input("\nEnter your query: ").strip()
            if query.lower() in ["quit", "exit", "q"]:
                break
            if not query:
                continue

            await test_rag_query(query, limit=3)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test RAG functionality")
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
        "--interactive", "-i", action="store_true", help="Run in interactive mode"
    )

    args = parser.parse_args()

    if args.interactive:
        asyncio.run(interactive_rag_test())
    else:
        results = asyncio.run(test_rag_query(args.query, args.limit))
        sys.exit(0 if results else 1)
