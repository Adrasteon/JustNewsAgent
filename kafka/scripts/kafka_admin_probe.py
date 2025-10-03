"""Admin-level Kafka readiness probe used by CI.

Attempts a lightweight Kafka Admin metadata call using kafka-python to
validate that the broker is responding at the protocol level. This provides
stronger readiness checks than TCP port checks alone.

Exit codes:
 - 0: broker ready
 - 2: probe failed after retries
 - 3: unexpected error
"""
from __future__ import annotations

import sys
import time
from typing import Optional

try:
    from kafka import KafkaAdminClient
except Exception as e:  # pragma: no cover - CI image availability
    print(f"kafka-python unavailable in image: {e}", file=sys.stderr)
    raise


def probe(bootstrap: str = "broker:9092", attempts: int = 30, interval: int = 5) -> int:
    """Probe the Kafka broker using KafkaAdminClient.

    Args:
        bootstrap: broker bootstrap servers string.
        attempts: number of attempts to try.
        interval: seconds to sleep between attempts.

    Returns:
        Exit code 0 on success, 2 on probe failure.
    """
    for i in range(1, attempts + 1):
        try:
            admin = KafkaAdminClient(bootstrap_servers=bootstrap, request_timeout_ms=2000)
            # Light-weight metadata call
            topics = admin.list_topics()
            print("Kafka admin probe successful; sample topics:", list(topics)[:10])
            admin.close()
            return 0
        except Exception as exc:  # pragma: no cover - runtime branching
            print(f"Kafka admin probe attempt {i}/{attempts} failed: {exc}", file=sys.stderr)
            time.sleep(interval)
    print("Kafka admin probe failed after retries", file=sys.stderr)
    return 2


def main(argv: Optional[list[str]] = None) -> int:
    # Allow customizing via env vars or defaults in CI invocation if required
    bootstrap = "broker:9092"
    attempts = 30
    interval = 5
    try:
        return probe(bootstrap=bootstrap, attempts=attempts, interval=interval)
    except Exception as e:  # pragma: no cover - unexpected exception
        print(f"Unexpected error during Kafka admin probe: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
