#!/usr/bin/env python3
"""
Automated Documentation Quality Monitoring System
Maintains >90% quality score for JustNews V4 documentation

Author: JustNews V4 Quality Assurance System
Date: September 7, 2025
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import schedule

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('docs_quality_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DocumentationQualityMonitor:
    """Automated quality monitoring system for documentation"""

    def __init__(self, catalogue_path: str = "../docs_catalogue_v2.json"):
        self.catalogue_path = Path(catalogue_path)
        self.target_score = 90.0
        self.alert_threshold = 85.0
        self.backup_dir = Path("quality_backups")
        self.backup_dir.mkdir(exist_ok=True)

        # Quality metrics history
        self.metrics_history = []
        self.alerts_sent = set()

        logger.info("Documentation Quality Monitor initialized")

    def calculate_quality_score(self) -> dict[str, Any]:
        """Calculate comprehensive quality score"""
        try:
            with open(self.catalogue_path, encoding='utf-8') as f:
                catalogue = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load catalogue: {e}")
            return {"error": str(e)}

        total_docs = 0
        total_desc_length = 0
        tagged_docs = 0
        issues_count = 0
        category_stats = {}

        for category in catalogue.get('categories', []):
            cat_name = category.get('name', 'Unknown')
            cat_docs = 0
            cat_issues = 0

            for doc in category.get('documents', []):
                total_docs += 1
                cat_docs += 1

                desc = doc.get('description', '')
                desc_length = len(desc.strip())
                total_desc_length += desc_length

                if doc.get('tags'):
                    tagged_docs += 1

                # Check for quality issues
                if desc_length < 50:
                    issues_count += 1
                    cat_issues += 1
                if not doc.get('tags'):
                    issues_count += 1
                    cat_issues += 1
                if not doc.get('word_count'):
                    issues_count += 1
                    cat_issues += 1

            category_stats[cat_name] = {
                'documents': cat_docs,
                'issues': cat_issues
            }

        # Calculate metrics
        avg_desc_length = total_desc_length / total_docs if total_docs > 0 else 0
        tagged_percentage = (tagged_docs / total_docs) * 100 if total_docs > 0 else 0

        desc_score = min(100, avg_desc_length / 2)
        tag_score = tagged_percentage
        issue_penalty = issues_count * 5
        final_score = max(0, (desc_score + tag_score) / 2 - issue_penalty)

        metrics = {
            'timestamp': datetime.now().isoformat(),
            'total_documents': total_docs,
            'average_description_length': round(avg_desc_length, 1),
            'tagged_percentage': round(tagged_percentage, 1),
            'quality_issues': issues_count,
            'description_score': round(desc_score, 1),
            'tagging_score': round(tag_score, 1),
            'issue_penalty': issue_penalty,
            'final_score': round(final_score, 1),
            'category_breakdown': category_stats,
            'status': self._get_status(final_score)
        }

        self.metrics_history.append(metrics)
        return metrics

    def _get_status(self, score: float) -> str:
        """Get status based on quality score"""
        if score >= self.target_score:
            return "EXCELLENT"
        elif score >= self.alert_threshold:
            return "WARNING"
        else:
            return "CRITICAL"

    def generate_quality_report(self) -> str:
        """Generate comprehensive quality report"""
        if not self.metrics_history:
            return "No metrics available"

        latest = self.metrics_history[-1]

        report = f"""
# 📊 Documentation Quality Report
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🎯 Overall Quality Score: **{latest['final_score']}/100**

### 📈 Key Metrics
- **Total Documents:** {latest['total_documents']}
- **Average Description Length:** {latest['average_description_length']} characters
- **Tagged Documents:** {latest['tagged_percentage']}%
- **Quality Issues:** {latest['quality_issues']}

### 📊 Score Breakdown
- **Description Score:** {latest['description_score']}/100
- **Tagging Score:** {latest['tagging_score']}/100
- **Issue Penalty:** -{latest['issue_penalty']} points

### 📂 Category Breakdown
"""

        for cat_name, stats in latest['category_breakdown'].items():
            report += f"- **{cat_name}:** {stats['documents']} docs, {stats['issues']} issues\n"

        report += f"""

### 🚨 Status: **{latest['status']}**

"""

        if latest['status'] == "CRITICAL":
            report += "**⚠️ CRITICAL: Quality score below acceptable threshold!**\n"
            report += "Immediate action required to maintain documentation standards.\n"
        elif latest['status'] == "WARNING":
            report += "**⚠️ WARNING: Quality score approaching critical threshold.**\n"
            report += "Monitor closely and address issues promptly.\n"
        else:
            report += "**✅ EXCELLENT: Quality standards maintained.**\n"

        return report

    def backup_catalogue(self) -> bool:
        """Create timestamped backup of catalogue"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = self.backup_dir / f"docs_catalogue_backup_{timestamp}.json"

            with open(self.catalogue_path, encoding='utf-8') as src:
                with open(backup_path, 'w', encoding='utf-8') as dst:
                    dst.write(src.read())

            logger.info(f"Catalogue backup created: {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return False

    def send_alert_email(self, subject: str, body: str) -> bool:
        """Send alert email (placeholder - configure SMTP settings)"""
        try:
            # Placeholder for email configuration
            # Configure SMTP settings as needed
            logger.info(f"Alert email would be sent: {subject}")
            return True
        except Exception as e:
            logger.error(f"Email alert failed: {e}")
            return False

    def check_quality_thresholds(self) -> None:
        """Check quality thresholds and trigger alerts if needed"""
        metrics = self.calculate_quality_score()

        if 'error' in metrics:
            logger.error(f"Quality check failed: {metrics['error']}")
            return

        score = metrics['final_score']
        alert_key = f"{metrics['timestamp'][:10]}_{score}"

        if score < self.alert_threshold and alert_key not in self.alerts_sent:
            subject = f"⚠️ Documentation Quality Alert: {score}/100"
            body = self.generate_quality_report()
            self.send_alert_email(subject, body)
            self.alerts_sent.add(alert_key)
            logger.warning(f"Quality alert triggered: {score}/100")

        logger.info(f"Quality check completed: {score}/100 ({metrics['status']})")

    def run_monitoring_cycle(self) -> None:
        """Run complete monitoring cycle"""
        logger.info("Starting quality monitoring cycle")

        # Calculate quality metrics
        self.check_quality_thresholds()

        # Generate and save report
        report = self.generate_quality_report()
        report_path = Path("quality_reports") / f"quality_report_{datetime.now().strftime('%Y%m%d')}.md"
        report_path.parent.mkdir(exist_ok=True)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        # Create backup
        self.backup_catalogue()

        logger.info("Quality monitoring cycle completed")

    def start_continuous_monitoring(self, interval_hours: int = 24) -> None:
        """Start continuous quality monitoring"""
        logger.info(f"Starting continuous monitoring (every {interval_hours} hours)")

        # Run initial check
        self.run_monitoring_cycle()

        # Schedule regular checks
        schedule.every(interval_hours).hours.do(self.run_monitoring_cycle)

        while True:
            schedule.run_pending()
            time.sleep(3600)  # Check every hour

def main():
    """Main function for command-line usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Documentation Quality Monitor")
    parser.add_argument("--catalogue", default="../docs_catalogue_v2.json",
                       help="Path to documentation catalogue")
    parser.add_argument("--continuous", action="store_true",
                       help="Run continuous monitoring")
    parser.add_argument("--interval", type=int, default=24,
                       help="Monitoring interval in hours")

    args = parser.parse_args()

    monitor = DocumentationQualityMonitor(args.catalogue)

    if args.continuous:
        monitor.start_continuous_monitoring(args.interval)
    else:
        monitor.run_monitoring_cycle()
        print(monitor.generate_quality_report())

if __name__ == "__main__":
    main()
