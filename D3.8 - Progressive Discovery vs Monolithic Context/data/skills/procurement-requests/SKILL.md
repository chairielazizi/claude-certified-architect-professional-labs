---
name: procurement-requests
description: Create and track purchase requests and explain who must approve them. Use when the user wants to buy goods or services or asks about purchase approvals and vendors.
tools: procurement_create_purchase_request, procurement_get_request_status
docs: DOC-04, DOC-05
---
# Procurement requests

## When to use
Buying anything over USD 1,000, checking a purchase request, or questions about approvals and vendors.

## Steps
1. Compute the total (quantity x unit price) and `load_doc` DOC-04 to tell the user who will approve it.
2. Call `procurement_create_purchase_request` with item, quantity and unit price in USD.
3. Standard laptops and monitors also go through IT's catalog - mention it, do not block on it.
4. New vendor needed -> explain onboarding from DOC-05 instead of creating the request.

## Output
Request ID, total, and the approver level from DOC-04.
