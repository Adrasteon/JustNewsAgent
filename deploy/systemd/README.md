# Systemd scaffold for JustNews

This folder contains a native deployment scaffold:

- `units/justnews@.service`: instanced unit template (use `justnews@<name>`) 
- `env/global.env`: global environment variables
- `env/*.env`: per-service environment variables
- `DEPLOYMENT.md`: step-by-step usage

## Instances

Create services by enabling instances:

- MCP Bus: `justnews@mcp_bus`
- Chief Editor: `justnews@chief_editor`
- Scout: `justnews@scout`
- Fact Checker: `justnews@fact_checker`
- Analyst: `justnews@analyst`
- Synthesizer: `justnews@synthesizer`
- Critic: `justnews@critic`
- Memory: `justnews@memory`
- Reasoning: `justnews@reasoning`
- NewsReader: `justnews@newsreader`

The unit loads `/etc/justnews/global.env` and `/etc/justnews/<instance>.env` if present.

## Notes
- Use absolute paths to the Python interpreter in the env files.
- Set `SERVICE_DIR` to the folder with your `main.py` for each service.
- Set `EXEC_START` to the full start command (e.g., `$JUSTNEWS_PYTHON main.py`).
- Pin GPUs per service with `CUDA_VISIBLE_DEVICES`.
- Consider rate limiting and TLS at ingress; ensure `/health`, `/ready`, `/warmup` endpoints.
