from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Set
import collections

RegionId = int
EdgeLabel = int
Mask = int
Adj = Dict[RegionId, List[Tuple[RegionId, EdgeLabel]]]


def _normalize_adj(adj: Adj) -> Adj:
    return {u: sorted(neis) for u, neis in adj.items()}


def _edges_undirected(adj: Adj) -> List[Tuple[int, int, int]]:
    seen = set()
    out = []
    for u, neis in adj.items():
        for v, lbl in neis:
            a, b = (u, v) if u <= v else (v, u)
            key = (a, b, lbl)
            if key in seen:
                continue
            # ensure the opposite appears
            assert any(
                (x == u and y == lbl)
                for (x, y) in [(w, l) for (w, l) in adj.get(v, [])]
            ), f"Missing symmetry for edge {(u, v, lbl)}"
            seen.add(key)
            out.append((a, b, lbl))
    return sorted(out)


@dataclass(frozen=True)
class Dual:
    """Combinatorial dual of a circle arrangement.

    - masks: region -> bitmask of circle membership (bit i-1 is 1 if inside circle i)
    - adj:   region -> list of (neighbor, label) directed entries; stored sorted

    NOTE: Heavy validations live in tests, not here, per project philosophy.
    """

    N: int
    masks: Dict[RegionId, Mask]
    adj: Adj

    def __post_init__(self):
        object.__setattr__(self, "adj", _normalize_adj(self.adj))
        if 0 not in self.masks or self.masks[0] != 0:
            raise ValueError("Region 0 must exist with mask 0 (outside region).")

    # --- inexpensive inspectors ---
    def regions(self) -> List[int]:
        return sorted(self.masks.keys())

    def edges(self) -> List[Tuple[int, int, int]]:
        return _edges_undirected(self.adj)

    def unlabeled_region_graph(self) -> Dict[int, List[int]]:
        g: Dict[int, Set[int]] = collections.defaultdict(set)
        for u, neis in self.adj.items():
            for v, _lbl in neis:
                g[u].add(v)
                g[v].add(u)
        return {k: sorted(v) for k, v in g.items()}

    def to_mask_struct_code_fixed_labels(self) -> str:
        """Structure code *holding labels fixed* (region-ID-free).
        Same spirit as your existing _mask_struct_code_fixed_labels().
        """
        # multiset of masks
        mask_bag = sorted(self.masks.values())
        # per-label edge multiset described in mask space
        lab_edges: Dict[int, List[Tuple[int, int]]] = collections.defaultdict(list)
        for u, v, lbl in self.edges():
            mu, mv = self.masks[u], self.masks[v]
            a, b = (mu, mv) if mu <= mv else (mv, mu)
            lab_edges[lbl].append((a, b))
        parts = [f"N={self.N}", f"M={','.join(map(str, mask_bag))}"]
        for lbl in range(1, self.N + 1):
            e_part = ";".join(f"{a}-{b}" for a, b in sorted(lab_edges.get(lbl, [])))
            parts.append(f"L{lbl}=[{e_part}]")
        return "|".join(parts)
