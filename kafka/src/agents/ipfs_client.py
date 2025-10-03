"""IPFS client shim that uses the local `ipfs` binary to add data.

This implementation avoids adding python-only IPFS dependencies and instead
calls the system `ipfs` command-line tool. In a self-hosted setup this
provides a transparent and auditable path into IPFS without external
cloud dependencies.
"""
from __future__ import annotations

import subprocess
from typing import Optional


class IpfsClient:
    def __init__(self, api_addr: Optional[str] = None):
        # If `api_addr` is passed, set IPFS_ADDR env var or configure accordingly.
        self.api_addr = api_addr

    def add_bytes(self, data: bytes) -> str:
        """Add bytes to a local IPFS daemon via the ipfs CLI and return the CID.

        Requires a local ipfs daemon (ipfs daemon) running and accessible to
        the executing user. The method writes the data to stdin of `ipfs add`.
        """
        # Use `ipfs add -q -` to read from stdin and output the CID
        proc = subprocess.Popen(["ipfs", "add", "-q", "-"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = proc.communicate(data)
        if proc.returncode != 0:
            raise RuntimeError(f"ipfs add failed: {stderr.decode('utf-8')}")
        cid = stdout.decode("utf-8").strip()
        return cid

    def get_gateway_url(self, cid: str) -> str:
        """Return a public gateway URL for the given CID (developer-friendly).

        Note: For full decentralization, serve via your local gateway or
        pinning service; do not rely on public gateways for production
        integrity guarantees.
        """
        return f"https://ipfs.io/ipfs/{cid}"
