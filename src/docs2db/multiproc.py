"""Multiprocessing utilities for Docs2DB."""

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import structlog
from rich.console import Console, Group
from rich.live import Live
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.text import Text

logger = structlog.get_logger(__name__)


def batch_generator(files: list[Path], batch_size: int):
    """Generate batches of file paths (str) for multiprocessing workers.

    Takes a list of Path objects and yields them in batches of the
    specified size. Converts Path objects to strings for JSON
    serialization compatibility with multiprocessing workers.

    Args:
        files: List of Path objects to be batched
        batch_size: Number of files to include in each batch

    Yields:
        list[str]: Batch of file paths as strings, with up to
                   batch_size items. The final batch may contain fewer
                   items if the total number of files is not evenly
                   divisible by batch_size.
    """
    for i in range(0, len(files), batch_size):
        batch = [str(f) for f in files[i : i + batch_size]]
        yield batch


def worker_count(
    total_files: int,
    max_workers: Optional[int] = None,
    max_workers_config: Optional[int] = None,
) -> int:
    """Determine optimal number of workers for processing.

    Args:
        total_files: Number of files to process
        max_workers: Explicit max workers override
        max_workers_config: Config-based max workers (optional limit)

    Returns:
        Optimal number of workers
    """
    if max_workers is not None:
        return min(max_workers, total_files)

    # Use CPU count - 1, but at least 1.
    cpu_workers = max(1, (os.cpu_count() or 1) - 1)

    # Apply config limit if provided.
    if max_workers_config is not None:
        cpu_workers = min(cpu_workers, max_workers_config)

    # Don't use more workers than files.
    return min(cpu_workers, total_files)


class ProgressDisplay(Live):
    """Generic Live display for processing progress with worker status.

    Args:
        console: Rich Console instance for output
        max_workers: Maximum number of worker processes
        total_files: Total number of files to process
        task_description: Description text shown in the progress bar
    """

    def __init__(
        self,
        console: Console,
        max_workers: int,
        total_files: int,
        task_description: str,
    ):
        self.console = console
        self.max_workers = max_workers

        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None),
            TextColumn("{task.percentage:>3.0f}%"),
            TextColumn("{task.completed:>6}/{task.total:<6}"),
            TextColumn("err:{task.fields[errors]:>6}"),
            TimeRemainingColumn(),
            console=console,
            expand=True,
        )

        # Task that shows overall progress from all workers.
        self.overall_task = self.progress.add_task(
            task_description,
            total=total_files,
            completed=0,
            errors=0,
        )

        # Create worker status lines.
        self.worker_lines = {}
        for worker_id in range(max_workers):
            self.set_worker_status(worker_id, "starting...")

        # Initialize Live with the display.
        super().__init__(self._make_display(), console=console)

    def _make_display(self) -> Group:
        """Create the combined display group."""
        return Group(self.progress, *self.worker_lines.values())

    def set_worker_status(self, worker_id: int, status: str):
        """Update a specific worker's status with terminal-width awareness."""
        terminal_width = self.console.size.width
        pre = f"Worker {worker_id:>2}: "
        available_width = terminal_width - len(pre) - 4

        if len(status) > available_width:
            trim_status = status[: available_width - 3] + "..."
        else:
            trim_status = status

        self.worker_lines[worker_id] = Text(f"{pre}{trim_status}", style="dim")

    def update_progress(self, completed: int, errors: int = 0):
        """Update the overall progress bar."""
        self.progress.update(
            self.overall_task,
            completed=completed,
            errors=errors,
        )

    def refresh_display(self):
        """Refresh the entire display."""
        self.update(self._make_display())


class BatchProcessor:
    """Multiprocess files with progress tracking and worker management.

    Args:
        worker_function: The worker function to execute for each batch.
        worker_args: Arguments to expand and pass to worker function.
        progress_message: Message to display in progress bar.
        batch_size: Number of files per batch.
        mem_threshold_mb: Restart workers if memory exceeds this (MB).
        max_workers: Max workers to use. If None, use optimal count.
    """

    def __init__(
        self,
        worker_function,
        worker_args: tuple,
        progress_message: str,
        batch_size: int,
        mem_threshold_mb: int,
        max_workers: Optional[int] = None,
    ):
        self.worker_function = worker_function
        self.worker_args = worker_args
        self.progress_message = progress_message
        self.batch_size = batch_size
        self.mem_threshold_mb = mem_threshold_mb
        self.max_workers = max_workers

        # Processing state
        self.processed = 0
        self.errors = 0
        self.error_data = []
        self.futures = {}
        self.executor: Optional[ProcessPoolExecutor] = None
        self.display: Optional[ProgressDisplay] = None
        self.console = Console()

    def process_files(
        self,
        to_process: list[Path],
    ) -> tuple[int, int]:
        """Process files using multiprocessing workers.

        Args:
            to_process: List of Path objects to process

        Returns:
            tuple[int, int]: (num_processed, num_failed)
        """
        count = len(to_process)
        self.max_workers = worker_count(count, max_workers=self.max_workers)

        # Use single-threaded processing when max_workers=1
        # This avoids fork issues on ARM Linux and other platforms
        if self.max_workers == 1:
            return self._process_single_threaded(to_process)

        batches = batch_generator(to_process, self.batch_size)

        self.console.print(
            f"[blue]Processing {count} files using {self.max_workers} workers[/blue]"
        )

        self._setup_logging()

        with ProgressDisplay(
            self.console, self.max_workers, count, self.progress_message
        ) as display:
            self.display = display
            self.executor = ProcessPoolExecutor(max_workers=self.max_workers)

            try:
                self._prime_worker_pool(batches)
                self._process_all_batches(batches)
            finally:
                self.executor.shutdown(wait=True)

        self._restore_logging()
        self._display_errors()

        return self.processed, self.errors

    def _process_single_threaded(
        self,
        to_process: list[Path],
    ) -> tuple[int, int]:
        """Process files in single-threaded mode (no forking).

        Used when max_workers=1 to avoid fork-related issues on ARM Linux
        and other platforms where forking after loading PyTorch causes deadlocks.

        Args:
            to_process: List of Path objects to process

        Returns:
            tuple[int, int]: (num_processed, num_failed)
        """
        count = len(to_process)
        self.console.print(
            f"[blue]Processing {count} files in single-threaded mode[/blue]"
        )

        batches = list(batch_generator(to_process, self.batch_size))

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None),
            TextColumn("{task.percentage:>3.0f}%"),
            TextColumn("{task.completed:>6}/{task.total:<6}"),
            TextColumn("err:{task.fields[errors]:>6}"),
            TimeRemainingColumn(),
            console=self.console,
            expand=True,
        ) as progress:
            task = progress.add_task(
                self.progress_message,
                total=count,
                completed=0,
                errors=0,
            )

            for batch in batches:
                result = self.worker_function(batch, *self.worker_args)

                self.processed += len(batch)
                self.errors += result["errors"]
                self.error_data.extend(result["error_data"])

                progress.update(
                    task,
                    completed=self.processed,
                    errors=self.errors,
                )

        self._display_errors()
        return self.processed, self.errors

    def _setup_logging(self):
        """Set up Rich logging to avoid progress bar interference."""
        self.original_handlers = logging.getLogger().handlers.copy()
        logging.getLogger().handlers.clear()

        rich_handler = RichHandler(
            console=self.console,
            show_time=False,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
        )
        logging.getLogger().addHandler(rich_handler)
        logging.getLogger().setLevel(logging.INFO)

    def _restore_logging(self):
        """Restore original logging handlers."""
        logging.getLogger().handlers.clear()
        logging.getLogger().handlers.extend(self.original_handlers)

    def _prime_worker_pool(self, batches):
        """Submit initial batches to all available workers."""
        assert isinstance(self.display, ProgressDisplay)
        assert isinstance(self.max_workers, int)
        for worker_id in range(self.max_workers):
            self._submit_batch_to_worker(worker_id, batches)
        self.display.refresh_display()

    def _submit_batch_to_worker(self, worker_id: int, batches) -> bool:
        """Submit a batch to a specific worker.

        Returns True if batch was submitted.

        """
        assert isinstance(self.display, ProgressDisplay)
        assert isinstance(self.executor, ProcessPoolExecutor)
        try:
            batch = next(batches)
            self.display.set_worker_status(worker_id, f"processing {batch[0]}")
            future = self.executor.submit(
                self.worker_function, batch, *self.worker_args
            )
            self.futures[future] = worker_id, batch
            return True
        except StopIteration:
            self.display.set_worker_status(worker_id, "idle")
            return False

    def _process_all_batches(self, batches):
        """Main processing loop that handles completed batches."""
        while self.futures:
            for future in as_completed(list(self.futures.keys())):
                worker_id, batch = self.futures.pop(future)

                if self._handle_batch_result(future, worker_id, batch):
                    # Worker restart was triggered
                    self._restart_workers(batches)
                    break
                else:
                    # Normal processing - submit next batch
                    self._submit_batch_to_worker(worker_id, batches)

                self._update_display()

            # Re-prime pool after restart if needed
            if not self.futures and batches:
                self._prime_worker_pool(batches)

    def _handle_batch_result(self, future, worker_id: int, batch) -> bool:
        """Handle a completed batch.

        Returns True if worker restart is needed.

        """
        try:
            result = future.result()
            self._process_successful_result(result, worker_id, batch)
            return self._check_memory_restart(result, worker_id)
        except Exception as e:
            # Log more detail for BrokenProcessPool errors
            from concurrent.futures.process import BrokenProcessPool

            if isinstance(e, BrokenProcessPool):
                logger.error(
                    f"Worker process terminated abruptly (BrokenProcessPool). "
                    f"This usually indicates: (1) Worker ran out of memory, "
                    f"(2) Segmentation fault, or (3) Fatal exception in worker initialization. "
                    f"Worker {worker_id} was processing batch: {batch[:3]}"
                )
            self._process_failed_result(e, worker_id, batch)
            return False

    def _process_successful_result(self, result, worker_id: int, batch):
        """Update logs and displays for a successful batch result."""
        assert isinstance(self.display, ProgressDisplay)
        # Replay worker logs
        for log_entry in result["worker_logs"]:
            logger.log(
                log_entry["level"],
                f"[Worker {worker_id}] {log_entry['message']}",
            )

        self.display.set_worker_status(
            worker_id,
            f"completed {result['last_file']}",
        )
        self.processed += len(batch)
        self.errors += result["errors"]
        self.error_data.extend(result["error_data"])

    def _process_failed_result(self, error, worker_id: int, batch):
        """Update error tracking and displays for a failed batch result."""
        assert isinstance(self.display, ProgressDisplay)

        # Log detailed error information
        logger.error(
            f"Worker {worker_id} failed processing batch: {error}",
            exc_info=True,
            extra={"batch_files": batch[:3], "error_type": type(error).__name__},
        )

        for file in batch:
            self.error_data.append({"file": file, "error": error})
            self.processed += 1
            self.errors += 1
        self.display.set_worker_status(worker_id, f"error: {error}")

    def _check_memory_restart(self, result, worker_id: int) -> bool:
        """Check if worker needs restart due to memory usage."""
        memory = result["memory"]
        if memory > self.mem_threshold_mb:
            logger.warning(f"Worker {worker_id} using {memory}MB, restarting")
            return True
        return False

    def _restart_workers(self, batches):
        """Restart all workers to clear memory."""
        assert isinstance(self.display, ProgressDisplay)
        assert isinstance(self.executor, ProcessPoolExecutor)
        # Complete remaining futures before restart
        remaining_futures = list(self.futures.keys())
        if remaining_futures:
            task_count = len(remaining_futures)
            logger.info(f"Waiting on {task_count} tasks before worker restart")
            self._complete_remaining_futures(remaining_futures)

        # Shutdown and create new executor
        self.executor.shutdown(wait=False)
        self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        self.futures = {}

        # Update worker statuses
        assert isinstance(self.max_workers, int)
        for worker_id in range(self.max_workers):
            self.display.set_worker_status(worker_id, "restarted")
        self.display.refresh_display()
        logger.info("Worker pool restarted to clear memory")

    def _complete_remaining_futures(self, remaining_futures):
        """Complete all remaining futures before worker restart."""
        assert isinstance(self.display, ProgressDisplay)
        for future in as_completed(remaining_futures):
            if future in self.futures:
                worker_id, batch = self.futures.pop(future)
                try:
                    result = future.result()
                    self._replay_worker_logs(result, worker_id)

                    self.processed += len(batch)
                    self.errors += result["errors"]
                    self.error_data.extend(result["error_data"])
                    self.display.set_worker_status(
                        worker_id, "completed before restart"
                    )
                except Exception as e:
                    for file in batch:
                        self.error_data.append({"file": file, "error": e})
                        self.processed += 1
                        self.errors += 1
                    self.display.set_worker_status(worker_id, f"error: {e}")

    def _replay_worker_logs(self, result, worker_id: int):
        """Replay worker logs from a result."""
        if "worker_logs" in result:
            for log_entry in result["worker_logs"]:
                logger.log(
                    log_entry["level"],
                    f"[Worker {worker_id}] {log_entry['message']}",
                )

    def _update_display(self):
        """Update the progress display."""
        assert isinstance(self.display, ProgressDisplay)
        self.display.update_progress(
            completed=self.processed,
            errors=self.errors,
        )
        self.display.refresh_display()

    def _display_errors(self):
        """Display any errors that occurred during processing."""
        if self.error_data:
            logging.error("\nErrors during processing:")
            for error in self.error_data:
                logging.error(f"{error['error']}\n{error['file']}\n")


class LogCollector:
    """Collects logs from worker processes."""

    def __init__(self):
        self.logs = []

    def handle(self, record):
        self.logs.append({
            "level": record.levelno,
            "levelname": record.levelname,
            "message": record.getMessage(),
            "name": record.name,
            "timestamp": record.created,
            "pathname": getattr(record, "pathname", ""),
            "lineno": getattr(record, "lineno", 0),
        })


class WorkerLogHandler(logging.Handler):
    """Custom logging handler that collects logs for replay in main process."""

    def __init__(self, collector):
        super().__init__()
        self.collector = collector

    def emit(self, record):
        self.collector.handle(record)


def setup_worker_logging(module_name: str) -> LogCollector:
    """Set up logging for a worker process to capture logs for replay.

    Args:
        module_name: The __name__ of the module calling this function

    Returns:
        log_collector
    """
    # Set up memory logging handler to capture logs from worker.
    log_collector = LogCollector()

    # Create a custom handler that collects logs.
    worker_handler = WorkerLogHandler(log_collector)
    worker_handler.setLevel(logging.DEBUG)

    # Get the root logger and replace its handlers with our handler.
    root_logger = logging.getLogger()
    root_logger.handlers = [worker_handler]
    root_logger.setLevel(logging.DEBUG)

    # Also capture the specific logger used by this module.
    # This ensures this module's 'logger' variable uses our custom handler.
    module_logger = logging.getLogger(module_name)
    module_logger.handlers = [worker_handler]
    module_logger.setLevel(logging.DEBUG)
    # Ensure it doesn't propagate to avoid double logging.
    module_logger.propagate = False

    # Configure structlog to route through standard logging in this worker.
    # Use a simple configuration that extracts just the message, no formatting.
    import structlog

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            # Just pass through the event as the message, no extra formatting.
            lambda logger, method_name, event_dict: event_dict["event"],
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    return log_collector
