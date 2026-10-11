# Orders Platform - team memory for Claude Code

Monorepo: a Python orders API (`api/`) and a TypeScript web client (`web/`).
This file is loaded into every Claude Code session in this repo - keep it short,
factual, and about *this* codebase. Personal preferences belong in `~/.claude/CLAUDE.md`.

## Commands
- Install: `make setup` (creates `.venv`, runs `npm ci` in `web/`)
- Lint: `make lint` (ruff + eslint) - must pass before every commit
- Test everything: `make test`; single API test: `pytest tests/api/test_orders.py -k refund`
- Web unit tests: `npm run test:unit` (run inside `web/`)
- Database migration: `make migrate` - always ask a human first

## Architecture
- `api/` FastAPI service; `api/routes/` are thin, business logic lives in `api/services/`
- `web/` React + TypeScript; it talks to the API only through `web/src/client/`
- `migrations/` SQL migrations; never edit files under `migrations/applied/`
- Module boundaries and data flow: @docs/architecture.md

## Conventions
- Money is integer cents (`int`), never floats. Round half-up only at display time.
- Every new endpoint needs: auth dependency, input model, and a test in `tests/api/`.
- Conventional Commits (`feat(api): ...`, `fix(web): ...`); branch names `<type>/<ticket>-<slug>`.
- Prefer small PRs (< 400 changed lines). Run the `code-reviewer` subagent before opening one.

## Guard rails
- Never read or print `.env*` files or anything in `secrets/` (also enforced by permissions).
- Do not add dependencies without asking; lockfiles change only via the package manager.
- If a hook blocks an edit, explain the change you wanted instead of working around it.
