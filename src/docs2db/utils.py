"""Shared utilities for Docs2DB"""

import os
import signal
import subprocess
import time
from pathlib import Path

import structlog
import xxhash
from transformers import AutoModel, AutoTokenizer

from docs2db.exceptions import ConfigurationError

logger = structlog.get_logger(__name__)


def hash_file(file_path: Path) -> str:
    """Generate xxHash hash of file content."""
    hash_obj = xxhash.xxh64()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)
    return f"xxh64:{hash_obj.hexdigest()}"


def hash_bytes(content_bytes: bytes) -> str:
    """Generate xxhash of bytes."""
    hash_obj = xxhash.xxh64()
    hash_obj.update(content_bytes)
    return f"xxh64:{hash_obj.hexdigest()}"


def ensure_model_available(model_id: str) -> None:
    """
    Ensure a model is available locally, downloading if necessary.

    Args:
        model_id: Hugging Face model identifier
            (e.g., "ibm-granite/granite-embedding-30m-english")
    """
    # Try to load from local cache only.
    try:
        AutoModel.from_pretrained(model_id, local_files_only=True)
        AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    except Exception:
        # An exception means the model is not in the local cache.
        try:
            # Download the model to the local cache.
            AutoModel.from_pretrained(model_id)
            AutoTokenizer.from_pretrained(model_id)
            # Verify the download.
            AutoModel.from_pretrained(model_id, local_files_only=True)
            AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        except Exception as download_error:
            logger.error(f"Failed to download {model_id}: {download_error}")
            raise ConfigurationError(
                f"Failed to download model {model_id}: {download_error}"
            ) from download_error


def cleanup_orphaned_workers() -> bool:
    """Clean up orphaned worker processes.

    Returns:
        True if successful, False if errors occurred

    Raises:
        ConfigurationError: If system commands are not available
    """
    try:
        result = subprocess.run(
            ["ps", "aux"], capture_output=True, text=True, check=True
        )

        orphaned_processes = []
        current_pid = os.getpid()
        current_ppid = os.getppid()

        for line in result.stdout.split("\n"):
            is_orphaned_worker = (
                ("docs2db-chunking-worker:" in line)
                or ("multiprocessing.spawn" in line and "spawn_main" in line)
                or ("docs2db" in line and "python" in line.lower())
            )

            if is_orphaned_worker:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        pid = int(parts[1])
                        # Don't kill current process or parent
                        if pid != current_pid and pid != current_ppid:
                            orphaned_processes.append((pid, line.strip()))
                    except ValueError:
                        continue

        if not orphaned_processes:
            logger.info("No orphaned docs2db workers found")
            return True

        logger.info(f"Found {len(orphaned_processes)} orphaned docs2db workers")
        for pid, description in orphaned_processes:
            logger.info(f"  PID {pid}: {description}")

        # Kill the processes gracefully

        # Step 1: Send SIGTERM to all processes
        pids_to_kill = [pid for pid, _ in orphaned_processes]
        logger.info(f"Sending SIGTERM to {len(pids_to_kill)} processes...")

        for pid in pids_to_kill:
            try:
                os.kill(pid, signal.SIGTERM)
                logger.info(f"  SIGTERM → PID {pid}")
            except (ProcessLookupError, PermissionError):
                # Process already gone or no permission - remove from list
                pids_to_kill.remove(pid)

        # Step 2: Wait 10 seconds for graceful shutdown
        if pids_to_kill:
            logger.info("Waiting 10 seconds for graceful shutdown...")
            time.sleep(10)

        # Step 3: Check which processes are still alive and SIGKILL them
        still_alive = []
        for pid in pids_to_kill:
            try:
                # Check if process still exists (os.kill with signal 0 doesn't kill, just checks)
                os.kill(pid, 0)
                still_alive.append(pid)
            except ProcessLookupError:
                # Process is gone - good!
                pass

        # Step 4: SIGKILL any remaining processes
        killed_count = len(pids_to_kill) - len(
            still_alive
        )  # Processes that responded to SIGTERM

        if still_alive:
            logger.info(f"Force-killing {len(still_alive)} unresponsive processes...")

            for pid in still_alive:
                try:
                    os.kill(pid, signal.SIGKILL)
                    killed_count += 1
                    logger.info(f"SIGKILL → PID {pid}")
                except (ProcessLookupError, PermissionError) as e:
                    logger.info(f"Could not kill PID {pid}: {e}")

        graceful_count = len(pids_to_kill) - len(still_alive)
        if graceful_count > 0:
            logger.info(f"{graceful_count} processes terminated gracefully")
        if len(still_alive) > 0:
            logger.info(f"{len(still_alive)} processes force-killed")
        logger.info(f"Total: {killed_count} orphaned processes cleaned up")

        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Error checking processes: {e}")
        raise ConfigurationError(f"Failed to check system processes: {e}") from e
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        raise ConfigurationError(f"Worker cleanup failed: {e}") from e
