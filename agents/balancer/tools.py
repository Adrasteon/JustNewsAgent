from typing import Any


def echo(*args, **kwargs) -> dict[str, Any]:
    """Simple tool that echoes back its input for smoke tests."""
    return {"args": list(args), "kwargs": dict(kwargs)}


def sum_numbers(numbers: list[float]) -> float:
    """Return the sum of a list of numbers."""
    return float(sum(numbers))
