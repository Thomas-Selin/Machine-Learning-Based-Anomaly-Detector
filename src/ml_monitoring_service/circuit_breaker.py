"""Circuit breaker pattern for graceful degradation of external services."""

import logging
import time
from collections.abc import Callable
from enum import Enum
from functools import wraps
from typing import Any

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Circuit breaker to prevent cascading failures.

    When a service fails repeatedly, the circuit opens and subsequent calls
    fail immediately without attempting the operation. After a timeout period,
    the circuit enters half-open state to test if the service recovered.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: int = 60,
        expected_exception: type[Exception] | tuple[type[Exception], ...] = Exception,
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            timeout: Seconds to wait before trying again (half-open state)
            expected_exception: Exception type(s) to catch and count as failure
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time: float | None = None
        self.state = CircuitState.CLOSED

    def call(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result of func if successful

        Raises:
            Exception: If circuit is open or func raises unexpected exception
        """
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info(
                    f"Circuit breaker for {func.__name__} entering HALF_OPEN state"
                )
            else:
                raise Exception(
                    f"Circuit breaker is OPEN for {func.__name__}. "
                    f"Will retry after {self.timeout}s timeout."
                )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        return (
            self.last_failure_time is not None
            and time.time() - self.last_failure_time >= self.timeout
        )

    def _on_success(self) -> None:
        """Handle successful operation."""
        if self.state == CircuitState.HALF_OPEN:
            logger.info("Circuit breaker reset to CLOSED state")
        self.failure_count = 0
        self.state = CircuitState.CLOSED

    def _on_failure(self) -> None:
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures. "
                f"Will retry after {self.timeout}s timeout."
            )


def circuit_breaker(
    failure_threshold: int = 5,
    timeout: int = 60,
    expected_exception: type[Exception] | tuple[type[Exception], ...] = Exception,
    fallback_return: Any = None,
) -> Callable:
    """Decorator for circuit breaker pattern with fallback.

    Args:
        failure_threshold: Number of failures before opening circuit
        timeout: Seconds to wait before trying again
        expected_exception: Exception type(s) to catch
        fallback_return: Value to return when circuit is open

    Returns:
        Decorated function with circuit breaker protection
    """
    cb = CircuitBreaker(failure_threshold, timeout, expected_exception)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return cb.call(func, *args, **kwargs)
            except Exception as e:
                logger.warning(
                    f"Circuit breaker caught exception in {func.__name__}: {e}"
                )
                if fallback_return is not None:
                    logger.info(
                        f"Returning fallback value for {func.__name__}: {fallback_return}"
                    )
                    return fallback_return
                raise

        return wrapper

    return decorator
