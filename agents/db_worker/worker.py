"""DB worker agent (FastAPI)

This agent exposes a single endpoint POST /handle_ingest which accepts a JSON
payload with 'article_payload' and 'statements' (list of (sql, params)). On
startup it registers with the MCP Bus as the 'db_worker' agent so other agents
can call it via the MCP Bus /call mechanism. The agent will attempt to use
psycopg2 when available to execute transactions; otherwise it returns a clear
error message explaining the missing dependency.
"""

import os
import logging
from typing import Any, Dict, List
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except Exception:
    psycopg2 = None

logger = logging.getLogger("db_worker")
logging.basicConfig(level=logging.INFO)

MCP_BUS_URL = os.environ.get('MCP_BUS_URL', 'http://localhost:8000')
DB_DSN = os.environ.get('POSTGRES_DSN') or os.environ.get('DATABASE_URL')


class IngestRequest(BaseModel):
    article_payload: Dict[str, Any]
    # statements should be list of [sql, params]
    statements: List[List[Any]] = []


def register_with_mcp_bus(port: int = 0):
    """Register this agent with the MCP Bus so it can be called via /call.

    This mirrors other agents' behavior in the repo (they POST /register).
    """
    try:
        payload = {
            "name": "db_worker",
            "address": f"http://localhost:{port}",
        }
        resp = requests.post(f"{MCP_BUS_URL}/register", json=payload, timeout=(2, 5))
        resp.raise_for_status()
        logger.info("Registered db_worker with MCP Bus")
    except Exception as e:
        logger.warning(f"Could not register db_worker with MCP Bus: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Register with MCP Bus on startup (best effort)
    try:
        register_with_mcp_bus()
    except Exception:
        logger.debug("MCP Bus registration failed during startup")
    yield


app = FastAPI(title="DB Worker", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/handle_ingest")
def handle_ingest(req: IngestRequest):
    article_payload = req.article_payload
    statements = req.statements or []

    logger.info(f"Received ingest for url={article_payload.get('url')}")

    if psycopg2 is None:
        raise HTTPException(status_code=500, detail="psycopg2 is not installed on this host")

    if not DB_DSN:
        raise HTTPException(status_code=500, detail="No POSTGRES_DSN/DATABASE_URL configured for DB worker")

    # Execute transaction
    try:
        conn = psycopg2.connect(DB_DSN)
        try:
            chosen_source_id = None
            with conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    for sql, params in statements:
                        cur.execute(sql, params)
                        try:
                            row = cur.fetchone()
                            if row and 'id' in row:
                                pass  # ID retrieved but not used in this context
                        except Exception:
                            # ignore fetch errors
                            pass

                    # If an article_id is available, call the stored-proc to pick canonical
                    article_id = article_payload.get('article_id')
                    if article_id:
                        try:
                            cur.execute('SELECT * FROM canonical_select_and_update(%s);', (article_id,))
                            sp_row = cur.fetchone()
                            if sp_row and 'chosen_source_id' in sp_row:
                                chosen_source_id = sp_row['chosen_source_id']
                            else:
                                # Some Postgres versions return the column as the first field
                                try:
                                    chosen_source_id = list(sp_row.values())[0] if sp_row else None
                                except Exception:
                                    chosen_source_id = None
                        except Exception as e:
                            logger.warning(f"Stored-proc call failed: {e}")

        finally:
            conn.close()

        resp = {"status": "ok", "url": article_payload.get('url')}
        if chosen_source_id is not None:
            resp['chosen_source_id'] = chosen_source_id
        return resp
    except Exception as e:
        logger.exception("DB transaction failed")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    import uvicorn
    PORT = int(os.environ.get('DB_WORKER_PORT', 8010))
    # Best-effort register with MCP Bus pointing at our address
    try:
        register_with_mcp_bus(PORT)
    except Exception:
        pass
    uvicorn.run(app, host='0.0.0.0', port=PORT)
