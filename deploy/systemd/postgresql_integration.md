---
title: PostgreSQL integration
description: Configure and verify DB connectivity for JustNews Memory
---

# PostgreSQL integration

Set the database URL in `/etc/justnews/global.env` (adjust credentials/host):

```
JUSTNEWS_DB_URL=postgresql://user:pass@localhost:5432/justnews
```

## Verification (on-host)

Use the helper to verify connectivity quickly:

```
sudo ./deploy/systemd/helpers/db-check.sh
```

If `psql` is available, the script runs `SELECT 1`. Otherwise, it checks the Memory service health endpoint as a proxy.

See also: `deploy/systemd/QUICK_REFERENCE.md` for the minimal env examples.

