"""
Tools for the Dashboard Agent.
"""

import logging

logger = logging.getLogger("dashboard_tools")

def log_event(event: str, details: dict):
    """Logs an event for the dashboard agent."""
    logger.info(f"Event: {event}, Details: {details}")

def format_status_response(status_data: dict) -> dict:
    """Formats the status response for the dashboard UI."""
    return {
        "agent_count": len(status_data),
        "agents": status_data
    }

def process_command_response(response: dict) -> dict:
    """Processes the response from a command sent to another agent."""
    return {
        "status": response.get("status", "unknown"),
        "details": response
    }
