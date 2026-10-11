from datetime import date

RESTOCKING_FEE_RATE = 0.15
RETURN_WINDOW_DAYS = 30


def compute_refund_amount(order_total: float, opened: bool, defective: bool) -> float:
    """Refund owed for a returned order. Opened, non-defective items pay the restocking fee."""
    if defective or not opened:
        return round(order_total, 2)
    return round(order_total * (1 - RESTOCKING_FEE_RATE), 2)


def refund_window_open(delivered_on: date, today: date) -> bool:
    """True while the customer can still start a return."""
    return (today - delivered_on).days <= RETURN_WINDOW_DAYS
