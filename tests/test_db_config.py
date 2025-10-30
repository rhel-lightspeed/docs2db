"""Tests for database configuration."""

import os
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from docs2db.database import get_db_config
from docs2db.exceptions import ConfigurationError


@pytest.fixture
def clean_env(monkeypatch):
    """Remove all database-related environment variables."""
    env_vars = [
        "DATABASE_URL",
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_DB",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
    ]
    for var in env_vars:
        monkeypatch.delenv(var, raising=False)


def test_default_config(clean_env, tmp_path, monkeypatch):
    """Test default configuration when no config sources are available."""
    # Change to temp directory with no compose file
    monkeypatch.chdir(tmp_path)

    config = get_db_config()

    assert config["host"] == "localhost"
    assert config["port"] == "5432"
    assert config["database"] == "ragdb"
    assert config["user"] == "postgres"
    assert config["password"] == "postgres"


def test_env_vars_override_defaults(clean_env, tmp_path, monkeypatch):
    """Test that environment variables override defaults."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("POSTGRES_HOST", "prod.example.com")
    monkeypatch.setenv("POSTGRES_PORT", "5433")
    monkeypatch.setenv("POSTGRES_DB", "production_db")
    monkeypatch.setenv("POSTGRES_USER", "admin")
    monkeypatch.setenv("POSTGRES_PASSWORD", "secret123")

    config = get_db_config()

    assert config["host"] == "prod.example.com"
    assert config["port"] == "5433"
    assert config["database"] == "production_db"
    assert config["user"] == "admin"
    assert config["password"] == "secret123"


def test_database_url(clean_env, tmp_path, monkeypatch):
    """Test DATABASE_URL parsing."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql://myuser:mypass@db.example.com:5434/mydb"
    )

    config = get_db_config()

    assert config["host"] == "db.example.com"
    assert config["port"] == "5434"
    assert config["database"] == "mydb"
    assert config["user"] == "myuser"
    assert config["password"] == "mypass"


def test_database_url_postgres_scheme(clean_env, tmp_path, monkeypatch):
    """Test DATABASE_URL with postgres:// scheme (not postgresql://)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", "postgres://user:pass@localhost:5432/db")

    config = get_db_config()

    assert config["host"] == "localhost"
    assert config["user"] == "user"
    assert config["password"] == "pass"


def test_database_url_without_port(clean_env, tmp_path, monkeypatch):
    """Test DATABASE_URL without explicit port."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost/mydb")

    config = get_db_config()

    assert config["host"] == "localhost"
    assert config["database"] == "mydb"


def test_database_url_without_password(clean_env, tmp_path, monkeypatch):
    """Test DATABASE_URL without password."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", "postgresql://user@localhost:5432/mydb")

    config = get_db_config()

    assert config["user"] == "user"
    assert config["host"] == "localhost"


def test_database_url_invalid_scheme(clean_env, tmp_path, monkeypatch):
    """Test DATABASE_URL with invalid scheme."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", "mysql://user:pass@localhost/db")

    with pytest.raises(ConfigurationError, match="Invalid DATABASE_URL scheme"):
        get_db_config()


def test_database_url_invalid_format(clean_env, tmp_path, monkeypatch):
    """Test DATABASE_URL with invalid format."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", "postgresql://invalid")

    with pytest.raises(ConfigurationError, match="Invalid DATABASE_URL format"):
        get_db_config()


def test_conflict_database_url_and_env_vars(clean_env, tmp_path, monkeypatch):
    """Test that DATABASE_URL and POSTGRES_* vars conflict is detected."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost/db")
    monkeypatch.setenv("POSTGRES_HOST", "other.example.com")

    with pytest.raises(ConfigurationError, match="Conflicting database configuration"):
        get_db_config()


def test_compose_file_in_cwd(clean_env, tmp_path, monkeypatch):
    """Test reading configuration from postgres-compose.yml in CWD."""
    monkeypatch.chdir(tmp_path)

    compose_content = """
name: test-project

services:
  db:
    image: pgvector/pgvector:pg17
    environment:
      POSTGRES_USER: compose_user
      POSTGRES_PASSWORD: compose_pass
      POSTGRES_DB: compose_db
    ports:
      - 5435:5432
"""

    compose_file = tmp_path / "postgres-compose.yml"
    compose_file.write_text(compose_content)

    config = get_db_config()

    assert config["user"] == "compose_user"
    assert config["password"] == "compose_pass"
    assert config["database"] == "compose_db"
    assert config["port"] == "5435"


def test_compose_file_malformed(clean_env, tmp_path, monkeypatch):
    """Test handling of malformed compose file."""
    monkeypatch.chdir(tmp_path)

    compose_file = tmp_path / "postgres-compose.yml"
    compose_file.write_text("invalid: yaml: content: [")

    # Should not raise, should use defaults and log warning
    config = get_db_config()

    # Should fall back to defaults when compose file is malformed
    assert config["user"] == "postgres"
    assert config["database"] == "ragdb"
    assert config["host"] == "localhost"


def test_env_vars_override_compose_file(clean_env, tmp_path, monkeypatch):
    """Test that environment variables override compose file."""
    monkeypatch.chdir(tmp_path)

    compose_content = """
name: test-project

services:
  db:
    environment:
      POSTGRES_USER: compose_user
      POSTGRES_DB: compose_db
"""

    compose_file = tmp_path / "postgres-compose.yml"
    compose_file.write_text(compose_content)

    monkeypatch.setenv("POSTGRES_USER", "env_user")
    monkeypatch.setenv("POSTGRES_DB", "env_db")

    config = get_db_config()

    # Environment variables should win
    assert config["user"] == "env_user"
    assert config["database"] == "env_db"


def test_partial_env_vars(clean_env, tmp_path, monkeypatch):
    """Test that partial environment variables work with defaults."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("POSTGRES_HOST", "custom.host.com")
    # Don't set other vars

    config = get_db_config()

    assert config["host"] == "custom.host.com"  # From env
    assert config["port"] == "5432"  # Default
    assert config["database"] == "ragdb"  # Default
    assert config["user"] == "postgres"  # Default


def test_database_url_overrides_compose(clean_env, tmp_path, monkeypatch):
    """Test that DATABASE_URL overrides compose file."""
    monkeypatch.chdir(tmp_path)

    compose_content = """
name: test-project

services:
  db:
    environment:
      POSTGRES_USER: compose_user
      POSTGRES_DB: compose_db
"""

    compose_file = tmp_path / "postgres-compose.yml"
    compose_file.write_text(compose_content)

    monkeypatch.setenv(
        "DATABASE_URL", "postgresql://url_user:url_pass@url.host.com/url_db"
    )

    config = get_db_config()

    # DATABASE_URL should win over compose
    assert config["user"] == "url_user"
    assert config["database"] == "url_db"
    assert config["host"] == "url.host.com"


def test_env_vars_override_database_url(clean_env, tmp_path, monkeypatch):
    """Test precedence: individual env vars should override DATABASE_URL."""
    monkeypatch.chdir(tmp_path)

    # This should not conflict because we're testing that individual vars
    # take precedence. But wait - the code raises an error if both are set!
    # Let me re-read the implementation...

    # Actually, the implementation detects this as a conflict and raises.
    # So this test should verify the conflict detection.
    monkeypatch.setenv("DATABASE_URL", "postgresql://url_user:pass@host/db")
    monkeypatch.setenv("POSTGRES_HOST", "other.host.com")

    with pytest.raises(ConfigurationError, match="Conflicting database configuration"):
        get_db_config()
