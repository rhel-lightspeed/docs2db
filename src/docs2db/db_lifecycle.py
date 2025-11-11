"""Database lifecycle management for Podman/Docker containers."""

import shutil
import subprocess
from pathlib import Path
from typing import Optional

import structlog

from docs2db.exceptions import ConfigurationError

logger = structlog.get_logger()


def detect_container_runtime() -> Optional[str]:
    """Detect available container runtime (podman or docker).

    Returns:
        "podman", "docker", or None if neither is available
    """
    if shutil.which("podman"):
        return "podman"
    elif shutil.which("docker"):
        return "docker"
    return None


def get_compose_file() -> Path:
    """Get the postgres-compose.yml file path.

    Looks in current working directory first, then falls back to
    creating a default one in the current directory.

    Returns:
        Path to compose file

    Raises:
        ConfigurationError: If compose file not found and can't create default
    """
    compose_file = Path.cwd() / "postgres-compose.yml"

    if compose_file.exists():
        return compose_file

    # Offer to create a default compose file
    logger.info(
        "No postgres-compose.yml found in current directory. "
        "Creating a default configuration..."
    )

    default_compose = """name: docs2db

services:
  db:
    container_name: docs2db-db
    profiles: ["prod"]
    image: docker.io/pgvector/pgvector:pg17
    restart: always
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: ragdb
    ports:
      - 5432:5432
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
"""

    try:
        with open(compose_file, "w") as f:
            f.write(default_compose)
        logger.info(f"Created postgres-compose.yml in {compose_file.parent}")
        return compose_file
    except Exception as e:
        raise ConfigurationError(
            f"Could not create postgres-compose.yml: {e}. "
            f"Please create one manually or ensure write permissions."
        ) from e


def get_project_name_from_compose(compose_file: Path) -> str:
    """Extract project name from compose file.

    Args:
        compose_file: Path to the compose file

    Returns:
        Project name

    Raises:
        OSError: If compose file cannot be read
        ConfigurationError: If compose file does not contain a "name:" field
    """
    with open(compose_file) as f:
        for line in f:
            line = line.strip()
            # Look for "name: <project_name>" at the top of the file
            if line.startswith("name:"):
                # Extract the project name (handle quoted and unquoted values)
                name = line.split(":", 1)[1].strip()
                # Remove quotes if present
                name = name.strip('"').strip("'")
                return name

    raise ConfigurationError(
        f"No 'name:' field found in compose file: {compose_file}. "
        f"Please add a project name to the compose file."
    )


def start_database(profile: str = "prod") -> bool:
    """Start PostgreSQL database using Podman/Docker compose.

    Args:
        profile: Container compose profile to use (default: "prod")

    Returns:
        True if successful, False otherwise

    Raises:
        ConfigurationError: If container runtime not available
    """
    runtime = detect_container_runtime()

    if not runtime:
        raise ConfigurationError(
            "Neither Podman nor Docker found. Please install one:\n"
            "  - Podman: https://podman.io/getting-started/installation\n"
            "  - Docker: https://docs.docker.com/get-docker/"
        )

    compose_file = get_compose_file()

    logger.info(f"Starting PostgreSQL database using {runtime}...")
    logger.info(f"Using compose file: {compose_file}")

    try:
        cmd = [
            runtime,
            "compose",
            "-f",
            str(compose_file),
            "--profile",
            profile,
            "up",
            "-d",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )

        if result.stdout:
            logger.info(result.stdout.strip())

        logger.info("Database started successfully")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start database: {e}")
        if e.stderr:
            logger.error(e.stderr)
        return False


def stop_database(profile: str = "prod") -> bool:
    """Stop PostgreSQL database using Podman/Docker compose.

    Data is preserved in volumes.

    Args:
        profile: Container compose profile to use (default: "prod")

    Returns:
        True if successful, False otherwise

    Raises:
        ConfigurationError: If container runtime not available
    """
    runtime = detect_container_runtime()

    if not runtime:
        raise ConfigurationError(
            "Neither Podman nor Docker found. Cannot stop database."
        )

    compose_file = get_compose_file()

    logger.info(f"Stopping PostgreSQL database using {runtime}...")

    try:
        cmd = [
            runtime,
            "compose",
            "-f",
            str(compose_file),
            "--profile",
            profile,
            "down",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )

        if result.stdout:
            logger.info(result.stdout.strip())

        logger.info("Database stopped successfully (data preserved)")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to stop database: {e}")
        if e.stderr:
            logger.error(e.stderr)
        return False


def destroy_database(profile: str = "prod") -> bool:
    """Stop PostgreSQL database and remove all data volumes.

    WARNING: This will delete all database data!

    Args:
        profile: Docker compose profile to use (default: "prod")

    Returns:
        True if successful, False otherwise

    Raises:
        ConfigurationError: If container runtime not available
    """
    runtime = detect_container_runtime()

    if not runtime:
        raise ConfigurationError(
            "Neither Podman nor Docker found. Cannot destroy database."
        )

    compose_file = get_compose_file()

    # First stop the database
    if not stop_database(profile):
        return False

    # Then remove volumes
    logger.info("Removing database volumes...")

    try:
        # Get the project name from compose file
        project_name = get_project_name_from_compose(compose_file)
        volume_name = f"{project_name}_pgdata"

        logger.info(f"Project name: {project_name}")
        logger.info(f"Attempting to remove volume: {volume_name}")

        cmd = [runtime, "volume", "rm", volume_name]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        # Don't fail if volume doesn't exist
        if result.returncode == 0:
            logger.info(f"Volume {volume_name} removed successfully")
        else:
            logger.info(f"Volume {volume_name} may not exist or already removed")

        logger.info("Database destroyed successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to remove volumes: {e}")
        return False


def get_database_logs(follow: bool = False) -> bool:
    """View PostgreSQL database logs.

    Args:
        follow: If True, follow logs in real-time (like tail -f)

    Returns:
        True if successful, False otherwise

    Raises:
        ConfigurationError: If container runtime not available
    """
    runtime = detect_container_runtime()

    if not runtime:
        raise ConfigurationError("Neither Podman nor Docker found. Cannot view logs.")

    compose_file = get_compose_file()

    logger.info("Viewing database logs...")

    try:
        cmd = [
            runtime,
            "compose",
            "-f",
            str(compose_file),
            "logs",
        ]

        if follow:
            cmd.append("-f")

        cmd.append("db")

        # For logs, we want to show output directly to user
        subprocess.run(cmd, check=True)

        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to view logs: {e}")
        return False
    except KeyboardInterrupt:
        # Normal exit when following logs
        logger.info("\nStopped viewing logs")
        return True
