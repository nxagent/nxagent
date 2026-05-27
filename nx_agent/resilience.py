"""Retry and timeout primitives for backend and tool operations."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, Type


class OperationTimeoutError(TimeoutError):
    """Raised internally when NxAgent stops waiting for an operation."""

    def __init__(self, timeout: float) -> None:
        self.timeout = timeout
        super().__init__(f"Operation exceeded timeout={timeout}s.")


class RetryExhaustedError(Exception):
    """Internal wrapper retaining the last operation failure and attempt count."""

    def __init__(self, last_error: Exception, attempts: int) -> None:
        self.last_error = last_error
        self.attempts = attempts
        super().__init__(str(last_error))


@dataclass(frozen=True)
class RetryPolicy:
    """Configuration for retrying a failed backend or tool operation.

    ``retries`` counts retries after the initial attempt. Timed-out operations
    are terminal by default because Python cannot safely terminate work that is
    already running.
    """

    retries: int = 0
    backoff: float = 0.0
    multiplier: float = 2.0
    retry_on: Tuple[Type[Exception], ...] = (Exception,)
    retry_timeouts: bool = False

    def __post_init__(self) -> None:
        if self.retries < 0:
            raise ValueError("retries must be >= 0")
        if self.backoff < 0:
            raise ValueError("backoff must be >= 0")
        if self.multiplier < 1:
            raise ValueError("multiplier must be >= 1")
        if not self.retry_on:
            raise ValueError("retry_on must contain at least one exception type")

    def should_retry(self, error: Exception) -> bool:
        """Return whether *error* is eligible for another attempt."""
        if isinstance(error, OperationTimeoutError) and not self.retry_timeouts:
            return False
        return isinstance(error, self.retry_on)


def _execute_with_timeout(operation: Callable[[], Any], timeout: Optional[float]) -> Any:
    """Run an operation, returning promptly when an optional timeout elapses."""
    if timeout is None:
        return operation()
    if timeout <= 0:
        raise ValueError("timeout must be > 0")

    outcomes: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def _target() -> None:
        try:
            outcomes.put((True, operation()))
        except Exception as exc:
            outcomes.put((False, exc))

    worker = threading.Thread(target=_target, daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise OperationTimeoutError(timeout)

    succeeded, value = outcomes.get()
    if succeeded:
        return value
    raise value


def execute_with_retry(
    operation: Callable[[], Any],
    policy: RetryPolicy,
    timeout: Optional[float] = None,
) -> tuple[Any, int]:
    """Execute an operation under retry and timeout controls."""
    for attempt in range(1, policy.retries + 2):
        try:
            return _execute_with_timeout(operation, timeout), attempt
        except Exception as exc:
            if attempt > policy.retries or not policy.should_retry(exc):
                raise RetryExhaustedError(exc, attempt) from exc
            if policy.backoff:
                time.sleep(policy.backoff * (policy.multiplier ** (attempt - 1)))

    raise RuntimeError("unreachable")
