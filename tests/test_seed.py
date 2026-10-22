import random

import numpy as np

from recops.seed import set_seed


def test_set_seed_is_reproducible() -> None:
    set_seed(123)
    a = (random.random(), np.random.rand(3))
    set_seed(123)
    b = (random.random(), np.random.rand(3))
    assert a[0] == b[0]
    assert np.array_equal(a[1], b[1])
