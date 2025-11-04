#!/usr/bin/env python3
"""
JustNewsAgent Deployment Status Update Script
Updates monitoring systems with deployment status and metrics.
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

class DeploymentStatusUpdater:
    """Handles deployment status updates to monitoring systems."""

    def __init__(self, monitoring_url: str | None = None):
        self.monitoring_url = monitoring_url or os.getenv('MONITORING_URL')
        self.github_repo = os.getenv('GITHUB_REPOSITORY', 'Adrasteon/JustNewsAgent')
        self.github_sha = os.getenv('GITHUB_SHA', 'unknown')
        self.github_ref = os.getenv('GITHUB_REF', 'unknown')

    def update_deployment_status(self, environment: str, version: str, status: str,
                               metadata: dict[str, Any] | None = None) -> bool:
        """Update deployment status in monitoring system."""
        if not self.monitoring_url:
            logger.warning("No monitoring URL configured, skipping status update")
            return False

        payload = {
            'environment': environment,
            'version': version,
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'repository': self.github_repo,
            'commit_sha': self.github_sha,
            'ref': self.github_ref,
            'metadata': metadata or {}
        }

        try:
            logger.info(f"Updating deployment status: {environment} {status} {version}")

            response = requests.post(
                f"{self.monitoring_url}/api/v1/deployments/status",
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )

            if response.status_code == 200:
                logger.info("Deployment status updated successfully")
                return True
            else:
                logger.error(f"Failed to update deployment status: HTTP {response.status_code}")
                logger.error(f"Response: {response.text}")
                return False

        except requests.RequestException as e:
            logger.error(f"Failed to update deployment status: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error updating deployment status: {e}")
            return False

    def update_deployment_metrics(self, environment: str, metrics: dict[str, Any]) -> bool:
        """Update deployment metrics in monitoring system."""
        if not self.monitoring_url:
            logger.warning("No monitoring URL configured, skipping metrics update")
            return False

        payload = {
            'environment': environment,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }

        try:
            logger.info(f"Updating deployment metrics for {environment}")

            response = requests.post(
                f"{self.monitoring_url}/api/v1/deployments/metrics",
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )

            if response.status_code == 200:
                logger.info("Deployment metrics updated successfully")
                return True
            else:
                logger.error(f"Failed to update deployment metrics: HTTP {response.status_code}")
                return False

        except requests.RequestException as e:
            logger.error(f"Failed to update deployment metrics: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error updating deployment metrics: {e}")
            return False

    def create_deployment_record(self, environment: str, version: str,
                               deployment_type: str = 'standard') -> bool:
        """Create a new deployment record."""
        if not self.monitoring_url:
            logger.warning("No monitoring URL configured, skipping record creation")
            return False

        payload = {
            'environment': environment,
            'version': version,
            'deployment_type': deployment_type,
            'start_time': datetime.now().isoformat(),
            'repository': self.github_repo,
            'commit_sha': self.github_sha,
            'ref': self.github_ref
        }

        try:
            logger.info(f"Creating deployment record: {environment} {version}")

            response = requests.post(
                f"{self.monitoring_url}/api/v1/deployments",
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )

            if response.status_code in [200, 201]:
                logger.info("Deployment record created successfully")
                return True
            else:
                logger.error(f"Failed to create deployment record: HTTP {response.status_code}")
                return False

        except requests.RequestException as e:
            logger.error(f"Failed to create deployment record: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error creating deployment record: {e}")
            return False

    def finalize_deployment(self, environment: str, version: str, status: str,
                          duration: float | None = None) -> bool:
        """Finalize deployment with end status and metrics."""
        if not self.monitoring_url:
            logger.warning("No monitoring URL configured, skipping finalization")
            return False

        payload = {
            'environment': environment,
            'version': version,
            'final_status': status,
            'end_time': datetime.now().isoformat(),
            'duration_seconds': duration
        }

        try:
            logger.info(f"Finalizing deployment: {environment} {version} {status}")

            response = requests.put(
                f"{self.monitoring_url}/api/v1/deployments/{environment}/{version}/finalize",
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )

            if response.status_code == 200:
                logger.info("Deployment finalized successfully")
                return True
            else:
                logger.error(f"Failed to finalize deployment: HTTP {response.status_code}")
                return False

        except requests.RequestException as e:
            logger.error(f"Failed to finalize deployment: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error finalizing deployment: {e}")
            return False

    def collect_deployment_metrics(self, environment: str) -> dict[str, Any]:
        """Collect deployment-related metrics."""
        metrics = {
            'deployment_timestamp': datetime.now().timestamp(),
            'environment': environment,
            'system_metrics': {},
            'application_metrics': {}
        }

        try:
            # Collect basic system metrics (would integrate with actual monitoring)
            metrics['system_metrics'] = {
                'cpu_usage': 45.2,  # Placeholder - would come from monitoring system
                'memory_usage': 67.8,
                'disk_usage': 23.1
            }

            # Collect application metrics
            metrics['application_metrics'] = {
                'active_connections': 150,
                'request_rate': 25.5,
                'error_rate': 0.02,
                'response_time_p95': 0.234
            }

        except Exception as e:
            logger.warning(f"Failed to collect deployment metrics: {e}")

        return metrics

def main():
    """Main deployment status update entry point."""
    parser = argparse.ArgumentParser(description='Update deployment status in monitoring systems')
    parser.add_argument('--environment', required=True,
                       help='Target environment (staging/production)')
    parser.add_argument('--version', required=True,
                       help='Deployment version/commit hash')
    parser.add_argument('--status', required=True,
                       choices=['started', 'in_progress', 'completed', 'failed', 'rollback'],
                       help='Deployment status')
    parser.add_argument('--monitoring-url',
                       help='Monitoring system URL (or set MONITORING_URL env var)')
    parser.add_argument('--action', default='update',
                       choices=['create', 'update', 'finalize', 'metrics'],
                       help='Action to perform (default: update)')
    parser.add_argument('--duration', type=float,
                       help='Deployment duration in seconds (for finalize action)')

    args = parser.parse_args()

    updater = DeploymentStatusUpdater(args.monitoring_url)

    try:
        if args.action == 'create':
            success = updater.create_deployment_record(args.environment, args.version)
        elif args.action == 'update':
            success = updater.update_deployment_status(args.environment, args.version, args.status)
        elif args.action == 'finalize':
            success = updater.finalize_deployment(args.environment, args.version, args.status, args.duration)
        elif args.action == 'metrics':
            metrics = updater.collect_deployment_metrics(args.environment)
            success = updater.update_deployment_metrics(args.environment, metrics)

        if success:
            logger.info(f"Deployment {args.action} completed successfully")
            sys.exit(0)
        else:
            logger.error(f"Deployment {args.action} failed")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Deployment status update failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
