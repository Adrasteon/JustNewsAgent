"""Small handler for enqueuing evidence review and notifying humans.

This module is intentionally lightweight and imports no FastAPI or Pydantic so
it can be used from unit tests and from the FastAPI endpoint.
"""
import os
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any

from agents.common.notifications import notify_slack, notify_email

logger = logging.getLogger('chief_editor.handler')


def handle_review_request(kwargs: Dict[str, Any]) -> Dict[str, Any]:
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
        'received_at': datetime.now(timezone.utc).isoformat()
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
