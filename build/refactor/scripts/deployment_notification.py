#!/usr/bin/env python3
"""
JustNewsAgent Deployment Notification Script
Sends deployment status notifications to Slack, Teams, or other channels.
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Any

import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeploymentNotifier:
    """Handles deployment notifications to various channels."""

    def __init__(self, webhook_url: str | None = None):
        self.webhook_url = webhook_url or os.getenv('DEPLOYMENT_WEBHOOK_URL')
        self.github_repo = os.getenv('GITHUB_REPOSITORY', 'Adrasteon/JustNewsAgent')
        self.github_sha = os.getenv('GITHUB_SHA', 'unknown')
        self.github_ref = os.getenv('GITHUB_REF', 'unknown')

    def create_slack_message(self, environment: str, status: str, version: str) -> dict[str, Any]:
        """Create Slack message payload."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')

        color = "good" if status == "success" else "danger" if status == "failure" else "warning"

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"🚀 JustNewsAgent Deployment {'✅' if status == 'success' else '❌' if status == 'failure' else '⚠️'}"
                }
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Environment:*\n{environment.upper()}"},
                    {"type": "mrkdwn", "text": f"*Status:*\n{status.upper()}"},
                    {"type": "mrkdwn", "text": f"*Version:*\n{version}"},
                    {"type": "mrkdwn", "text": f"*Time:*\n{timestamp}"}
                ]
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Repository:* {self.github_repo}\n*Commit:* `{self.github_sha[:8]}`\n*Ref:* {self.github_ref}"
                }
            }
        ]

        # Add context for different statuses
        if status == "success":
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "✅ Deployment completed successfully! The new version is now live."
                }
            })
        elif status == "failure":
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "❌ Deployment failed. Check the CI/CD pipeline logs for details."
                }
            })
        elif status == "rollback-ready":
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "🔄 Rollback manifest prepared. Ready for manual rollback if needed."
                }
            })

        return {
            "blocks": blocks,
            "attachments": [
                {
                    "color": color,
                    "fields": [
                        {"title": "Environment", "value": environment, "short": True},
                        {"title": "Status", "value": status, "short": True},
                        {"title": "Version", "value": version, "short": True},
                        {"title": "Timestamp", "value": timestamp, "short": True}
                    ]
                }
            ]
        }

    def create_teams_message(self, environment: str, status: str, version: str) -> dict[str, Any]:
        """Create Microsoft Teams message payload."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')

        facts = [
            {"name": "Environment", "value": environment.upper()},
            {"name": "Status", "value": status.upper()},
            {"name": "Version", "value": version},
            {"name": "Time", "value": timestamp},
            {"name": "Repository", "value": self.github_repo},
            {"name": "Commit", "value": self.github_sha[:8]},
            {"name": "Ref", "value": self.github_ref}
        ]

        title = f"JustNewsAgent Deployment - {environment.upper()}"

        return {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "themeColor": "0076D7" if status == "success" else "FF0000" if status == "failure" else "FFA500",
            "title": title,
            "text": f"Deployment {status} for version {version}",
            "sections": [
                {
                    "facts": facts,
                    "markdown": True
                }
            ]
        }

    def send_notification(self, environment: str, status: str, version: str,
                         platform: str = "slack") -> bool:
        """Send deployment notification."""
        if not self.webhook_url:
            logger.warning("No webhook URL configured, skipping notification")
            return False

        try:
            if platform.lower() == "slack":
                message = self.create_slack_message(environment, status, version)
            elif platform.lower() == "teams":
                message = self.create_teams_message(environment, status, version)
            else:
                logger.error(f"Unsupported platform: {platform}")
                return False

            logger.info(f"Sending {platform} notification for {environment} deployment ({status})")

            response = requests.post(
                self.webhook_url,
                json=message,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )

            if response.status_code == 200:
                logger.info(f"Notification sent successfully to {platform}")
                return True
            else:
                logger.error(f"Failed to send notification: HTTP {response.status_code}")
                logger.error(f"Response: {response.text}")
                return False

        except requests.RequestException as e:
            logger.error(f"Failed to send notification: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending notification: {e}")
            return False

    def send_to_all_platforms(self, environment: str, status: str, version: str) -> bool:
        """Send notifications to all configured platforms."""
        success = True

        # Send to Slack if configured
        slack_url = os.getenv('SLACK_WEBHOOK_URL')
        if slack_url:
            self.webhook_url = slack_url
            if not self.send_notification(environment, status, version, "slack"):
                success = False

        # Send to Teams if configured
        teams_url = os.getenv('TEAMS_WEBHOOK_URL')
        if teams_url:
            self.webhook_url = teams_url
            if not self.send_notification(environment, status, version, "teams"):
                success = False

        return success

def main():
    """Main notification entry point."""
    parser = argparse.ArgumentParser(description='Send deployment notifications')
    parser.add_argument('--environment', required=True,
                       help='Deployment environment (staging/production)')
    parser.add_argument('--status', required=True,
                       help='Deployment status (success/failure/rollback-ready)')
    parser.add_argument('--version', required=True,
                       help='Deployment version/commit hash')
    parser.add_argument('--platform', default='all',
                       choices=['slack', 'teams', 'all'],
                       help='Notification platform (default: all)')

    args = parser.parse_args()

    notifier = DeploymentNotifier()

    logger.info(f"Sending deployment notification: {args.environment} {args.status} {args.version}")

    if args.platform == 'all':
        success = notifier.send_to_all_platforms(args.environment, args.status, args.version)
    else:
        success = notifier.send_notification(args.environment, args.status, args.version, args.platform)

    if success:
        logger.info("Deployment notification sent successfully")
        sys.exit(0)
    else:
        logger.error("Failed to send deployment notification")
        sys.exit(1)

if __name__ == "__main__":
    main()
