"""Small handler for enqueuing evidence review and notifying humans.

This module is intentionally lightweight and imports no FastAPI or Pydantic so
it can be used from unit tests and from the FastAPI endpoint.
"""
import json
import os
from datetime import UTC, datetime
from typing import Any

from agents.common.notifications import notify_email, notify_slack
from common.observability import get_logger

logger = get_logger(__name__)


def handle_review_request(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Write the review request to the queue and notify humans (best-effort).

    Expected kwargs: evidence_manifest, reason
    Returns a dict with status and manifest.
    """
    manifest = kwargs.get('evidence_manifest')
    reason = kwargs.get('reason')

    queue_file = os.environ.get('EVIDENCE_REVIEW_QUEUE', './evidence_review_queue.jsonl')
    record = {
        'manifest': manifest,
        'reason': reason,
        'received_at': datetime.now(UTC).isoformat()
    }

    try:
        os.makedirs(os.path.dirname(queue_file) or '.', exist_ok=True)
        with open(queue_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')
    except Exception as e:
        logger.error(f'Failed to write review queue: {e}')
        return {'status': 'error', 'error': str(e)}

    # Best-effort notifications
    try:
        slack_text = f"New evidence queued for review: {manifest} (reason={reason})"
        notify_slack(slack_text)
    except Exception:
        logger.debug('Slack notification failed or skipped')

    try:
        email_subject = f"Evidence Review Requested: {manifest}"
        email_body = f"A new evidence manifest was enqueued: {manifest}\nReason: {reason}\nReceived at: {record['received_at']}"
        notify_email(email_subject, email_body)
    except Exception:
        logger.debug('Email notification failed or skipped')

    return {'status': 'ok', 'manifest': manifest}
