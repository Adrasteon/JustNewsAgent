"""Central DB connection helper for JustNews scripts.

Usage: from scripts.db_config import get_db_conn
"""

from __future__ import annotations

import os

import psycopg2


def get_db_conn():
    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        return psycopg2.connect(database_url)

    params = {
        "host": os.environ.get("JUSTNEWS_DB_HOST", "localhost"),
        "port": int(os.environ.get("JUSTNEWS_DB_PORT", 5432)),
        "dbname": os.environ.get("JUSTNEWS_DB_NAME", "justnews"),
        "user": os.environ.get("JUSTNEWS_DB_USER", os.getenv("USER")),
        "password": os.environ.get("JUSTNEWS_DB_PASSWORD"),
    }
    # If password not supplied via env, attempt to read ~/.pgpass (libpq format)
    if not params.get("password"):
        try:
            pgpass_path = os.path.expanduser("~/.pgpass")
            if os.path.exists(pgpass_path):
                with open(pgpass_path, encoding="utf-8") as fh:
                    for ln in fh:
                        ln = ln.strip()
                        if not ln or ln.startswith("#"):
                            continue
                        # host:port:database:username:password
                        parts = ln.split(":")
                        if len(parts) != 5:
                            continue
                        h, p, db, u, pw = parts
                        host_match = h == params["host"] or h == "*"
                        port_match = str(p) == str(params["port"]) or p == "*"
                        db_match = db == params["dbname"] or db == "*"
                        if host_match and port_match and db_match:
                            # If the script explicitly set JUSTNEWS_DB_USER, prefer it.
                            explicit_user = (
                                os.environ.get("JUSTNEWS_DB_USER") is not None
                            )
                            if explicit_user:
                                if u == params["user"] or u == "*":
                                    params["password"] = pw
                                    break
                                else:
                                    continue
                            # If user not explicit, adopt the username from pgpass entry and the password
                            params["user"] = u
                            params["password"] = pw
                            break
        except Exception:
            # best-effort; fall back to leaving password unset
            pass

    return psycopg2.connect(**params)
