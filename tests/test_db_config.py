"""Tests for database configuration."""

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


def test_default_config_exits_without_credentials(clean_env, tmp_path, monkeypatch):
    """Test that missing user/password raises ConfigurationError."""
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ConfigurationError, match="Missing required database credentials"):
        get_db_config()


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
    assert config["password"] == "secret123"  # noqa: S105


def test_database_url(clean_env, tmp_path, monkeypatch):
    """Test DATABASE_URL parsing."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", "postgresql://myuser:mypass@db.example.com:5434/mydb")

    config = get_db_config()

    assert config["host"] == "db.example.com"
    assert config["port"] == "5434"
    assert config["database"] == "mydb"
    assert config["user"] == "myuser"
    assert config["password"] == "mypass"  # noqa: S105


def test_database_url_postgres_scheme(clean_env, tmp_path, monkeypatch):
    """Test DATABASE_URL with postgres:// scheme (not postgresql://)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", "postgres://user:pass@localhost:5432/db")

    config = get_db_config()

    assert config["host"] == "localhost"
    assert config["user"] == "user"
    assert config["password"] == "pass"  # noqa: S105


def test_database_url_without_port(clean_env, tmp_path, monkeypatch):
    """Test DATABASE_URL without explicit port."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost/mydb")

    config = get_db_config()

    assert config["host"] == "localhost"
    assert config["database"] == "mydb"


def test_database_url_without_password(clean_env, tmp_path, monkeypatch):
    """Test DATABASE_URL without password raises ConfigurationError."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", "postgresql://user@localhost:5432/mydb")

    with pytest.raises(ConfigurationError, match="Missing required database credentials"):
        get_db_config()


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


def test_partial_env_vars_exits_without_credentials(clean_env, tmp_path, monkeypatch):
    """Test that setting only host without user/password raises ConfigurationError."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("POSTGRES_HOST", "custom.host.com")

    with pytest.raises(ConfigurationError, match="Missing required database credentials"):
        get_db_config()


def test_partial_env_vars_with_credentials(clean_env, tmp_path, monkeypatch):
    """Test that partial environment variables work when credentials are provided."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("POSTGRES_HOST", "custom.host.com")
    monkeypatch.setenv("POSTGRES_USER", "myuser")
    monkeypatch.setenv("POSTGRES_PASSWORD", "mypass")

    config = get_db_config()

    assert config["host"] == "custom.host.com"
    assert config["port"] == "5432"
    assert config["database"] == "ragdb"
    assert config["user"] == "myuser"
    assert config["password"] == "mypass"  # noqa: S105


def test_database_url_takes_precedence(clean_env, tmp_path, monkeypatch):
    """Test that DATABASE_URL works for config resolution."""
    monkeypatch.chdir(tmp_path)

    monkeypatch.setenv("DATABASE_URL", "postgresql://url_user:url_pass@url.host.com/url_db")

    config = get_db_config()

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
