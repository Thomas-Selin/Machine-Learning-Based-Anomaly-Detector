"""Tests for circuit breaker functionality."""

import time

import pytest

from ml_monitoring_service.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    circuit_breaker,
)


class TestCircuitBreaker:
    """Test circuit breaker pattern."""

    def test_circuit_closed_on_success(self):
        """Test circuit stays closed on successful calls."""
        cb = CircuitBreaker(failure_threshold=3, timeout=1)

        def successful_func():
            return "success"

        result = cb.call(successful_func)
        assert result == "success"
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_circuit_opens_after_threshold(self):
        """Test circuit opens after failure threshold is reached."""
        cb = CircuitBreaker(failure_threshold=3, timeout=1)

        def failing_func():
            raise ValueError("Test error")

        # Fail 3 times to reach threshold
        for _ in range(3):
            with pytest.raises(ValueError):
                cb.call(failing_func)

        assert cb.state == CircuitState.OPEN
        assert cb.failure_count == 3

    def test_circuit_rejects_when_open(self):
        """Test circuit rejects calls when open."""
        cb = CircuitBreaker(failure_threshold=2, timeout=1)

        def failing_func():
            raise ValueError("Test error")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(failing_func)

        assert cb.state == CircuitState.OPEN

        # Next call should be rejected immediately
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            cb.call(failing_func)

    def test_circuit_half_open_after_timeout(self):
        """Test circuit enters half-open state after timeout."""
        cb = CircuitBreaker(failure_threshold=2, timeout=1)

        def failing_func():
            raise ValueError("Test error")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(failing_func)

        assert cb.state == CircuitState.OPEN

        # Wait for timeout
        time.sleep(1.1)

        # Next call should enter half-open state
        with pytest.raises(ValueError):
            cb.call(failing_func)

        assert cb.state == CircuitState.OPEN  # Failed again, stays open

    def test_circuit_resets_on_success_after_half_open(self):
        """Test circuit resets to closed after successful call in half-open state."""
        cb = CircuitBreaker(failure_threshold=2, timeout=1)

        call_count = [0]

        def sometimes_failing_func():
            call_count[0] += 1
            if call_count[0] <= 2:
                raise ValueError("Test error")
            return "success"

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(sometimes_failing_func)

        assert cb.state == CircuitState.OPEN

        # Wait for timeout
        time.sleep(1.1)

        # Successful call should reset circuit
        result = cb.call(sometimes_failing_func)
        assert result == "success"
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0


class TestCircuitBreakerDecorator:
    """Test circuit breaker decorator."""

    def test_decorator_with_fallback(self):
        """Test decorator returns fallback value when circuit is open."""

        @circuit_breaker(failure_threshold=2, timeout=1, fallback_return="fallback")
        def failing_func():
            raise ValueError("Test error")

        # Open the circuit
        for _ in range(2):
            result = failing_func()
            assert result == "fallback"

        # Circuit is now open, should return fallback immediately
        result = failing_func()
        assert result == "fallback"

    def test_decorator_without_fallback(self):
        """Test decorator raises exception when circuit is open without fallback."""

        @circuit_breaker(failure_threshold=2, timeout=1)
        def failing_func():
            raise ValueError("Test error")

        # Open the circuit
        with pytest.raises(ValueError):
            failing_func()
        with pytest.raises(ValueError):
            failing_func()

        # Circuit is now open, should raise exception
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            failing_func()

    def test_decorator_success(self):
        """Test decorator allows successful calls."""

        @circuit_breaker(failure_threshold=2, timeout=1)
        def successful_func():
            return "success"

        result = successful_func()
        assert result == "success"

    def test_decorator_catches_specific_exception(self):
        """Test decorator only catches specified exception type."""

        @circuit_breaker(failure_threshold=2, timeout=1, expected_exception=ValueError)
        def func_with_different_exceptions():
            raise TypeError("Different error")

        # Should not catch TypeError, should propagate
        with pytest.raises(TypeError):
            func_with_different_exceptions()
