"""Small, deterministic strategies used by the transparent demo mode."""


def deliberately_leaky(prices: list[float]) -> list[int]:
    """A seeded failure: uses the next close to choose the current signal."""
    result = []
    for index, price in enumerate(prices):
        if index + 1 >= len(prices):
            result.append(0)
        else:
            result.append(1 if prices[index + 1] > price else -1)
    return result


def momentum(prices: list[float]) -> list[int]:
    """A causal baseline: follow the most recent observed price change."""
    result = [0]
    for index in range(1, len(prices)):
        result.append(1 if prices[index] >= prices[index - 1] else -1)
    return result


LEAKY_SOURCE = '''def signals(prices):
    """BUG: looks at the next close, which is not known at decision time."""
    return [0 if i + 1 >= len(prices) else (1 if prices[i + 1] > prices[i] else -1)
            for i in range(len(prices))]
'''

MOMENTUM_SOURCE = '''def signals(prices):
    """Use only prices observed up to the current row."""
    signals = [0]
    for i in range(1, len(prices)):
        signals.append(1 if prices[i] >= prices[i - 1] else -1)
    return signals
'''
