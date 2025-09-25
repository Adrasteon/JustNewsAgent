import pytest


@pytest.fixture
def articles():
    """Project-level fixture used by some production stress tests under agents/."""
    return [
        "Sample article one.",
        "Sample article two about space.",
        "Third sample article with some content.",
    ]
