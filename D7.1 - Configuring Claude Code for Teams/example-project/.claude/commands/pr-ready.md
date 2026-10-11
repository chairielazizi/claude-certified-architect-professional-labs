---
description: Run the pre-PR checklist (lint, tests, read-only review) and report readiness
argument-hint: [base-branch]
allowed-tools: Bash(make lint), Bash(make test), Bash(git diff:*), Bash(git status)
---

Check whether the current branch is ready for a pull request against `$ARGUMENTS`
(use `main` if no base branch was given).

1. Run `git status` and `git diff --stat $ARGUMENTS...HEAD` and summarise what changed.
2. Run `make lint` and `make test`; if either fails, stop and report the failures.
3. Ask the `code-reviewer` subagent to review `git diff $ARGUMENTS...HEAD`.
4. Reply with: a 3-bullet change summary, test results, review findings by severity,
   and a suggested Conventional Commit title. Do not commit or push.
