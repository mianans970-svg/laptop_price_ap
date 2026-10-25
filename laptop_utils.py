import numpy as np

PRICE_MIN_PKR = 20_000       # cheapest realistic laptop
PRICE_MAX_PKR = 2_500_000    # most expensive laptop in the dataset is about Rs 1.9 million

def safe_expm1(log_price):
    """Inverse of log1p, with predictions kept inside a realistic price range."""
    lo, hi = np.log1p(PRICE_MIN_PKR), np.log1p(PRICE_MAX_PKR)
    return np.expm1(np.clip(log_price, lo, hi))
