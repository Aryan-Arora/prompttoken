"""Cost estimates.

The rate below is an ILLUSTRATIVE placeholder, not a looked-up price. Replace
it with your provider's published input price (and record the date) before
quoting any dollar figure.
"""

INPUT_USD_PER_MTOK = 3.00
PRICE_NOTE = "Illustrative rate, not a published price. Edit backend/app/pricing.py."


def cost_saved_per_1k_calls(tokens_saved):
    return tokens_saved * 1000 * INPUT_USD_PER_MTOK / 1_000_000
