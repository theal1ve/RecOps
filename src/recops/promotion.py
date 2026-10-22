"""Promote / reject rule for a candidate model."""

from __future__ import annotations


def decide(
    candidate: float, production: float | None, min_rel_improvement: float
) -> tuple[bool, str]:
    """Promote only if the candidate beats production by ``min_rel_improvement`` (relative).

    Both metrics must be measured on the same holdout. Without a production model the
    first candidate is promoted (bootstrap). Returns ``(promote, human-readable reason)``.
    """
    if production is None:
        return True, "bootstrap: no production model yet"
    required = production * (1.0 + min_rel_improvement)
    if candidate > production and candidate >= required:
        return (
            True,
            f"candidate {candidate:.4f} >= required {required:.4f} (production {production:.4f})",
        )
    return (
        False,
        f"candidate {candidate:.4f} < required {required:.4f} (production {production:.4f})",
    )
