# Architecture (imported from CLAUDE.md via @docs/architecture.md)

```
web/src/client  --HTTP/JSON-->  api/routes  -->  api/services  -->  api/repositories  -->  Postgres
                                     |
                                     +--> issue tracker (read-only MCP server for Claude, not the app)
```

- Routes validate input (pydantic models) and call exactly one service function.
- Services own business rules (pricing, refunds) and are pure where possible.
- Repositories are the only layer that issues SQL; always use bound parameters.
