apply_discount(price_cents: int, percent: int) -> int

Return the price after a percentage discount, in integer cents.
- The result is rounded half-up to the nearest cent: (price_cents * (100 - percent) + 50) // 100.
- percent must be an integer in [0, MAX_DISCOUNT_PERCENT] where MAX_DISCOUNT_PERCENT = 50;
  otherwise raise ValueError.
- price_cents must be >= 0; otherwise raise ValueError.
