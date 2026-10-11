---
name: it-access
description: Fix account and access problems - VPN resets, password resets, account unlocks and access-group changes. Use when a user cannot sign in, needs VPN reset, or needs access to an application or shared drive.
tools: it_reset_vpn_access, it_reset_password, it_unlock_account, it_grant_group_access
docs: DOC-09, DOC-10
---
# IT access

## When to use
Sign-in failures, VPN problems, locked accounts, or requests to add someone to an access group.

## Steps
1. VPN not working or device replaced -> `it_reset_vpn_access` (revokes the old certificate, sends a new profile).
2. Forgotten password -> `it_reset_password`; locked after failed attempts -> `it_unlock_account`.
3. Lost or stolen device -> tell the user to report it to Security first (DOC-10), then reset.
4. Access-group changes need a named approving manager for `it_grant_group_access`.

## Output
What was done and where the user will receive the new credential or profile.
