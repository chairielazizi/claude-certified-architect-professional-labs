from datetime import date


def warranty_end_date(order_date: date, warranty_months: int) -> date:
    """Warranty starts on the order date and runs for the product's warranty_months."""
    month_index = order_date.month - 1 + warranty_months
    year = order_date.year + month_index // 12
    month = month_index % 12 + 1
    return date(year, month, min(order_date.day, 28))


def is_under_warranty(order_date: date, warranty_months: int, today: date) -> bool:
    return today <= warranty_end_date(order_date, warranty_months)
