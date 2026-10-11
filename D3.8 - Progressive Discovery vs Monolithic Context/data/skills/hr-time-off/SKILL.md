---
name: hr-time-off
description: Answer questions about PTO balances and time-off requests, and submit PTO requests. Use when the user asks how many vacation or PTO days are left or wants to book time off.
tools: hr_get_pto_balance, hr_submit_pto_request
docs: DOC-06
---
# HR time off

## When to use
Questions about remaining vacation/PTO days, booking time off, or PTO rules (accrual, carryover).

## Steps
1. "How many days are left?" -> call `hr_get_pto_balance` with the employee's full name. It returns the LIVE
   remaining balance with approved and pending requests already subtracted.
2. Do NOT use `payroll_get_leave_accrual` for balances: it reports days earned per month, not days remaining.
3. Booking time off -> confirm start and end dates with the user, then call `hr_submit_pto_request`.
4. Policy questions (carryover limits, accrual rate) -> `load_doc` DOC-06.

## Output
One sentence with the number of days and the year. Mention that pending requests are already subtracted.
