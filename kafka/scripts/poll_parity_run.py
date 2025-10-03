"""Poll GitHub Actions workflow runs for a given workflow file and branch.

This script uses the public GitHub Actions API to look for a workflow run
associated with the given branch (head_branch). It polls until a run is
found and completes or until a timeout is reached.

Usage:
  python kafka/scripts/poll_parity_run.py --workflow parity-e2e-dispatch.yml --branch trigger/parity-e2e-run --tries 40 --sleep 15

Note: This script uses unauthenticated API requests; for private repos or
higher rate limits, set the GITHUB_TOKEN env var and the script will use it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

try:
    from urllib.request import urlopen, Request
except Exception:  # pragma: no cover - stdlib
    from urllib2 import urlopen, Request


GITHUB_API = "https://api.github.com"


def fetch_workflow_runs(owner: str, repo: str, workflow_file: str) -> Optional[dict]:
    url = f"{GITHUB_API}/repos/{owner}/{repo}/actions/workflows/{workflow_file}/runs?per_page=50"
    headers = {"Accept": "application/vnd.github+json"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"
    req = Request(url, headers=headers)
    with urlopen(req) as fh:
        data = fh.read().decode("utf-8")
        return json.loads(data)


def find_run_for_branch(json_data: dict, branch: str) -> Optional[dict]:
    for r in json_data.get("workflow_runs", []):
        if r.get("head_branch") == branch:
            return r
    return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Poll parity workflow runs for a branch")
    parser.add_argument("--owner", default="Adrasteon")
    parser.add_argument("--repo", default="JustNewsAgent")
    parser.add_argument("--workflow", default="parity-e2e-dispatch.yml")
    parser.add_argument("--branch", required=True)
    parser.add_argument("--tries", type=int, default=40)
    parser.add_argument("--sleep", type=int, default=15)
    args = parser.parse_args(argv)

    for attempt in range(1, args.tries + 1):
        print(f"poll attempt {attempt}/{args.tries}")
        try:
            runs = fetch_workflow_runs(args.owner, args.repo, args.workflow)
        except Exception as e:  # pragma: no cover - network
            print(f"Failed to fetch workflow runs: {e}", file=sys.stderr)
            runs = None

        if runs:
            r = find_run_for_branch(runs, args.branch)
            if r:
                run_id = r.get("id")
                status = r.get("status")
                conclusion = r.get("conclusion")
                html = r.get("html_url")
                print(f"Found run id={run_id} status={status} conclusion={conclusion} url={html}")
                if status == "completed":
                    print("Run completed; exiting")
                    return 0
                # Not completed -- show where to monitor and continue
                print("Run found but not completed yet; waiting...")
        time.sleep(args.sleep)

    print("Timed out waiting for run to appear or complete", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
