---
name: api-endpoint-checklist
description: Step-by-step checklist for adding or changing an HTTP endpoint in api/routes. Use when creating a new route, changing a request/response model, or touching authentication on an endpoint.
---

# API endpoint checklist

Only the name and description above sit in context by default; this body is loaded
when the task matches, so it can be detailed without costing tokens every session.

1. **Route** - add it under `api/routes/<resource>.py`; keep it thin (validate, call one service).
2. **Auth** - include the `require_user` dependency; admin-only routes use `require_role("admin")`.
3. **Models** - define pydantic request/response models in `api/schemas/`; money fields are `int` cents.
4. **Service** - business logic in `api/services/`; no SQL here, call a repository.
5. **Repository** - parameterised SQL only (`:name` bind parameters, never f-strings).
6. **Tests** - happy path, auth failure (401/403), validation failure (422), and one edge case.
7. **Docs** - update the OpenAPI summary/description; note breaking changes in the PR body.

Run `make lint` and `pytest tests/api -k <resource>` before handing back.
