#!/usr/bin/env python3
"""
JustNewsAgent Deployment Report Generator
Generates comprehensive deployment reports with analytics and metrics.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Any

import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeploymentReporter:
    """Generates deployment reports and analytics."""

    def __init__(self, monitoring_url: str | None = None):
        self.monitoring_url = monitoring_url or os.getenv('MONITORING_URL')
        self.github_repo = os.getenv('GITHUB_REPOSITORY', 'Adrasteon/JustNewsAgent')
        self.github_sha = os.getenv('GITHUB_SHA', 'unknown')
        self.github_ref = os.getenv('GITHUB_REF', 'unknown')

    def fetch_deployment_data(self, environment: str, days: int = 7) -> list[dict[str, Any]]:
        """Fetch deployment data from monitoring system."""
        if not self.monitoring_url:
            logger.warning("No monitoring URL configured, returning empty data")
            return []

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        params = {
            'environment': environment,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat()
        }

        try:
            response = requests.get(
                f"{self.monitoring_url}/api/v1/deployments",
                params=params,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                logger.info(f"Fetched {len(data)} deployment records for {environment}")
                return data
            else:
                logger.error(f"Failed to fetch deployment data: HTTP {response.status_code}")
                return []

        except requests.RequestException as e:
            logger.error(f"Failed to fetch deployment data: {e}")
            return []

    def fetch_health_metrics(self, environment: str, days: int = 7) -> list[dict[str, Any]]:
        """Fetch health metrics from monitoring system."""
        if not self.monitoring_url:
            logger.warning("No monitoring URL configured, returning empty metrics")
            return []

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        params = {
            'environment': environment,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat()
        }

        try:
            response = requests.get(
                f"{self.monitoring_url}/api/v1/health/metrics",
                params=params,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                logger.info(f"Fetched {len(data)} health metrics for {environment}")
                return data
            else:
                logger.error(f"Failed to fetch health metrics: HTTP {response.status_code}")
                return []

        except requests.RequestException as e:
            logger.error(f"Failed to fetch health metrics: {e}")
            return []

    def calculate_deployment_metrics(self, deployments: list[dict[str, Any]]) -> dict[str, Any]:
        """Calculate deployment success metrics."""
        if not deployments:
            return {
                'total_deployments': 0,
                'successful_deployments': 0,
                'failed_deployments': 0,
                'success_rate': 0.0,
                'average_duration': 0.0,
                'rollback_rate': 0.0
            }

        total = len(deployments)
        successful = sum(1 for d in deployments if d.get('final_status') == 'completed')
        failed = sum(1 for d in deployments if d.get('final_status') == 'failed')
        rollbacks = sum(1 for d in deployments if d.get('deployment_type') == 'rollback')

        # Calculate average duration for successful deployments
        durations = []
        for d in deployments:
            if d.get('final_status') == 'completed' and 'duration_seconds' in d:
                durations.append(d['duration_seconds'])

        avg_duration = sum(durations) / len(durations) if durations else 0

        return {
            'total_deployments': total,
            'successful_deployments': successful,
            'failed_deployments': failed,
            'success_rate': (successful / total) * 100 if total > 0 else 0,
            'average_duration': avg_duration,
            'rollback_rate': (rollbacks / total) * 100 if total > 0 else 0
        }

    def calculate_health_metrics(self, health_data: list[dict[str, Any]]) -> dict[str, Any]:
        """Calculate health and uptime metrics."""
        if not health_data:
            return {
                'total_checks': 0,
                'healthy_checks': 0,
                'unhealthy_checks': 0,
                'uptime_percentage': 0.0,
                'average_response_time': 0.0
            }

        total_checks = len(health_data)
        healthy_checks = sum(1 for h in health_data if h.get('overall_healthy', False))

        # Calculate uptime percentage
        uptime_percentage = (healthy_checks / total_checks) * 100 if total_checks > 0 else 0

        # Calculate average response time
        response_times = []
        for h in health_data:
            if 'services' in h:
                for service_data in h['services'].values():
                    if 'response_time' in service_data:
                        response_times.append(service_data['response_time'])

        avg_response_time = sum(response_times) / len(response_times) if response_times else 0

        return {
            'total_checks': total_checks,
            'healthy_checks': healthy_checks,
            'unhealthy_checks': total_checks - healthy_checks,
            'uptime_percentage': uptime_percentage,
            'average_response_time': avg_response_time
        }

    def generate_deployment_report(self, environment: str, days: int = 7) -> dict[str, Any]:
        """Generate comprehensive deployment report."""
        logger.info(f"Generating deployment report for {environment} (last {days} days)")

        # Fetch data
        deployments = self.fetch_deployment_data(environment, days)
        health_metrics = self.fetch_health_metrics(environment, days)

        # Calculate metrics
        deployment_metrics = self.calculate_deployment_metrics(deployments)
        health_stats = self.calculate_health_metrics(health_metrics)

        # Generate report
        report = {
            'report_metadata': {
                'environment': environment,
                'report_period_days': days,
                'generated_at': datetime.now().isoformat(),
                'repository': self.github_repo,
                'commit_sha': self.github_sha,
                'ref': self.github_ref
            },
            'deployment_metrics': deployment_metrics,
            'health_metrics': health_stats,
            'recent_deployments': deployments[-10:],  # Last 10 deployments
            'health_incidents': self._identify_health_incidents(health_metrics),
            'recommendations': self._generate_recommendations(deployment_metrics, health_stats)
        }

        logger.info(f"Deployment report generated with {len(deployments)} deployments and {len(health_metrics)} health checks")
        return report

    def _identify_health_incidents(self, health_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Identify health incidents from health data."""
        incidents = []

        for health_check in health_data:
            if not health_check.get('overall_healthy', True):
                incident = {
                    'timestamp': health_check.get('timestamp'),
                    'severity': 'high' if not health_check.get('services', {}).get('mcp_bus', {}).get('healthy', True) else 'medium',
                    'affected_services': []
                }

                if 'services' in health_check:
                    for service_name, service_data in health_check['services'].items():
                        if not service_data.get('healthy', True):
                            incident['affected_services'].append({
                                'service': service_name,
                                'error': service_data.get('error', 'Unknown error')
                            })

                incidents.append(incident)

        return incidents[-10:]  # Return last 10 incidents

    def _generate_recommendations(self, deployment_metrics: dict[str, Any],
                                health_stats: dict[str, Any]) -> list[str]:
        """Generate recommendations based on metrics."""
        recommendations = []

        # Deployment success rate recommendations
        success_rate = deployment_metrics.get('success_rate', 0)
        if success_rate < 95:
            recommendations.append(f"Improve deployment success rate (currently {success_rate:.1f}%). Consider implementing more comprehensive pre-deployment testing.")
        elif success_rate < 99:
            recommendations.append(f"Deployment success rate is good ({success_rate:.1f}%) but could be improved with additional validation steps.")

        # Health recommendations
        uptime = health_stats.get('uptime_percentage', 0)
        if uptime < 99.5:
            recommendations.append(f"Improve system uptime (currently {uptime:.1f}%). Investigate recurring health check failures.")
        elif uptime < 99.9:
            recommendations.append(f"System uptime is excellent ({uptime:.1f}%) but monitor for any degradation trends.")

        # Response time recommendations
        avg_response = health_stats.get('average_response_time', 0)
        if avg_response > 1.0:
            recommendations.append(f"Review response times (currently {avg_response:.2f}s average). Consider performance optimizations.")
        elif avg_response > 0.5:
            recommendations.append(f"Response times are acceptable ({avg_response:.2f}s average) but monitor for increases.")

        # Rollback rate recommendations
        rollback_rate = deployment_metrics.get('rollback_rate', 0)
        if rollback_rate > 10:
            recommendations.append(f"High rollback rate ({rollback_rate:.1f}%). Review deployment validation and consider canary deployment strategies.")

        if not recommendations:
            recommendations.append("All metrics are within acceptable ranges. Continue monitoring and maintaining current deployment practices.")

        return recommendations

    def save_report(self, report: dict[str, Any], output_file: str) -> None:
        """Save report to file."""
        try:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Report saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")

    def print_report_summary(self, report: dict[str, Any]) -> None:
        """Print a human-readable summary of the report."""
        print("\n" + "="*60)
        print("DEPLOYMENT REPORT SUMMARY")
        print("="*60)

        meta = report['report_metadata']
        print(f"Environment: {meta['environment']}")
        print(f"Report Period: Last {meta['report_period_days']} days")
        print(f"Generated: {meta['generated_at']}")
        print()

        # Deployment metrics
        dep = report['deployment_metrics']
        print("DEPLOYMENT METRICS:")
        print(f"  Total Deployments: {dep['total_deployments']}")
        print(f"  Success Rate: {dep['success_rate']:.1f}%")
        print(f"  Average Duration: {dep['average_duration']:.1f}s")
        print(f"  Rollback Rate: {dep['rollback_rate']:.1f}%")
        print()

        # Health metrics
        health = report['health_metrics']
        print("HEALTH METRICS:")
        print(f"  Total Health Checks: {health['total_checks']}")
        print(f"  Uptime: {health['uptime_percentage']:.1f}%")
        print(f"  Average Response Time: {health['average_response_time']:.2f}s")
        print()

        # Recent deployments
        recent = report['recent_deployments']
        if recent:
            print("RECENT DEPLOYMENTS:")
            for dep in recent[-5:]:  # Show last 5
                status = dep.get('final_status', 'unknown')
                version = dep.get('version', 'unknown')[:8]
                timestamp = dep.get('end_time', 'unknown')
                print(f"  {timestamp[:19]} - {version} - {status.upper()}")
            print()

        # Recommendations
        recs = report['recommendations']
        if recs:
            print("RECOMMENDATIONS:")
            for rec in recs:
                print(f"  • {rec}")
            print()

        print("="*60)

def main():
    """Main deployment reporting entry point."""
    parser = argparse.ArgumentParser(description='Generate deployment reports and analytics')
    parser.add_argument('--environment', required=True,
                       help='Target environment (staging/production)')
    parser.add_argument('--days', type=int, default=7,
                       help='Number of days to analyze (default: 7)')
    parser.add_argument('--monitoring-url',
                       help='Monitoring system URL (or set MONITORING_URL env var)')
    parser.add_argument('--output-file',
                       help='File to save detailed report (JSON)')
    parser.add_argument('--summary-only', action='store_true',
                       help='Print summary only, no detailed output')

    args = parser.parse_args()

    reporter = DeploymentReporter(args.monitoring_url)

    try:
        report = reporter.generate_deployment_report(args.environment, args.days)

        if args.output_file:
            reporter.save_report(report, args.output_file)

        if not args.summary_only:
            reporter.print_report_summary(report)
        else:
            # Print just key metrics
            dep = report['deployment_metrics']
            health = report['health_metrics']
            print(f"Environment: {args.environment}")
            print(f"Deployments: {dep['total_deployments']} (Success: {dep['success_rate']:.1f}%)")
            print(f"Health: {health['uptime_percentage']:.1f}% uptime, {health['average_response_time']:.2f}s avg response")

        logger.info("Deployment report generation completed successfully")

    except Exception as e:
        logger.error(f"Deployment report generation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
