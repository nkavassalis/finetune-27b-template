"""The order training examples are visited in, per data-parallel rank.

Every rank draws from the same seeded shuffle of all example indices, then
takes every WORLD-th one, so ranks never overlap within an epoch and together
cover the data exactly once. Resuming from step S skips exactly the examples
an uninterrupted run would have consumed, so a resumed run is the same run.
Stdlib only, so this is tested directly (tests/test_order.py).
"""
from __future__ import annotations

import itertools
import random
from collections.abc import Iterator


def examples(n: int, world: int = 1, rank: int = 0, seed: int = 7) -> Iterator[int]:
    """Infinite stream of example indices for one rank, epoch after epoch."""
    rng = random.Random(seed)
    for _ in itertools.count():
        order = list(range(n))
        rng.shuffle(order)
        yield from order[rank::world]


def resumed(n: int, world: int, rank: int, seed: int, *, step: int, grad_accum: int) -> Iterator[int]:
    """The stream positioned just after optimizer step `step`."""
    stream = examples(n, world, rank, seed)
    for _ in range(step * grad_accum):
        next(stream)
    return stream
