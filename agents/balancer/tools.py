from typing import Any, Dict, List


def echo(*args, **kwargs) -> Dict[str, Any]:
    """Simple tool that echoes back its input for smoke tests."""
    return {"args": list(args), "kwargs": dict(kwargs)}


def sum_numbers(numbers: List[float]) -> float:
    """Return the sum of a list of numbers."""
    return float(sum(numbers))
