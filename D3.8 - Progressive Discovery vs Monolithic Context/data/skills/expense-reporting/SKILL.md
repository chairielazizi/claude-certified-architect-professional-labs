---
name: expense-reporting
description: Explain expense and meal policies and file expense reports. Use for questions about what can be expensed (client dinners, per diem, travel costs) or to create and submit an expense report.
tools: expense_create_report, expense_add_line_item, expense_submit_report, expense_get_per_diem_rate
docs: DOC-01, DOC-03
---
# Expense reporting

## When to use
Questions about reimbursable costs, meal limits, per diem, or filing an expense report.

## Steps
1. Policy question about client meals or entertainment -> `load_doc` DOC-01 and answer from it.
2. Per diem question -> `expense_get_per_diem_rate` for the city and date (rates in DOC-03).
3. Filing -> `expense_create_report`, one `expense_add_line_item` per receipt, then `expense_submit_report`.
4. For the receipt rules checklist, read the bundled file `receipts-checklist.md` (only when filing).

## Output
Answer policy questions in two or three sentences and cite the document ID.
