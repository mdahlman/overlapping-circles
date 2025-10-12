from __future__ import annotations
from .dual import Dual


def canonical_code_label_invariant(d: Dual) -> str:
    """Expose the label‑invariant canonical code via the Dual implementation."""
    return d.canonical_code()
