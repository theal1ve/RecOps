import numpy as np

from recops.monitoring.drift import quality_alert
from recops.monitoring.psi import level, psi
from recops.promotion import decide


def test_psi_detects_shift_and_ignores_noise() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=20000)
    assert psi(a, rng.normal(size=20000)) < 0.02
    assert psi(a, rng.normal(loc=1.0, size=20000)) > 0.25


def test_psi_edge_cases() -> None:
    assert psi(np.ones(100), np.ones(100)) == 0.0
    assert psi(np.ones(100), np.zeros(100)) > 1.0
    assert psi(np.array([np.nan] * 50 + [1.0] * 50), np.ones(100)) > 0.1
    assert level(0.3, 0.1, 0.25) == "alert" and level(0.05, 0.1, 0.25) == "ok"


def test_promotion_rule() -> None:
    assert decide(0.1, None, 0.01)[0]  # bootstrap
    assert decide(0.102, 0.1, 0.01)[0]
    assert not decide(0.1005, 0.1, 0.01)[0]  # better, but below the required margin
    assert not decide(0.09, 0.1, 0.01)[0]  # worse -> rollback (production stays)
    assert not decide(0.0, 0.0, 0.01)[0]


def test_quality_alert() -> None:
    assert quality_alert(0.9, 1.0, 0.05)[0]
    assert not quality_alert(0.99, 1.0, 0.05)[0]
