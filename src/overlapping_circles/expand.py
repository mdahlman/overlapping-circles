from __future__ import annotations
from typing import Mapping, Iterable, List, Tuple, Dict, Set
import collections

from .dual import Dual


def _region_graph_cycles(g: Mapping[int, Iterable[int]]) -> List[List[int]]:
    """Enumerate simple cycles with Johnson's algorithm (undirected adaptation)."""
    # Lightweight cycle enumeration suitable for small graphs (our N is small for now).
    nodelist = sorted(g)
    index = {v: i for i, v in enumerate(nodelist)}

    # convert to directed both ways for ease
    DG = {u: sorted(neis) for u, neis in g.items()}
    blocked = {u: False for u in DG}
    B = {u: set() for u in DG}
    stack: List[int] = []
    cycles: List[List[int]] = []

    def unblock(u: int):
        blocked[u] = False
        while B[u]:
            w = B[u].pop()
            if blocked[w]:
                unblock(w)

    def circuit(v: int, s: int) -> bool:
        found = False
        stack.append(v)
        blocked[v] = True
        for w in DG[v]:
            if w < s:  # reduce duplicates by start index
                continue
            if w == s and len(stack) >= 3:
                cycles.append(stack[:] + [s])
                found = True
            elif not blocked[w]:
                if circuit(w, s):
                    found = True
        if found:
            unblock(v)
        else:
            for w in DG[v]:
                B[w].add(v)
        stack.pop()
        return found

    for i, s in enumerate(nodelist):
        # restrict to nodes >= s to orient search
        DG2 = {u: [w for w in DG[u] if w >= s] for u in DG}
        for u in DG:
            blocked[u] = False
            B[u].clear()
        DG = DG2
        circuit(s, s)
    # Normalize cycles to canonical rotation starting at min node
    normed = []
    for cyc in cycles:
        cyc = cyc[:-1]
        m = min(cyc)
        i = cyc.index(m)
        rot = cyc[i:] + cyc[:i]
        # choose direction
        r1 = rot
        r2 = [rot[0]] + list(reversed(rot[1:]))
        best = r1 if tuple(r1) < tuple(r2) else r2
        normed.append(best)
    # dedup
    uniq = []
    seen = set()
    for c in normed:
        t = tuple(c)
        if t not in seen:
            seen.add(t)
            uniq.append(c)
    return uniq


def _parity_ok(d: Dual, cycle: List[int]) -> bool:
    # For each existing label ℓ, the cycle must cross ℓ an even number of times.
    counts = collections.Counter()
    pos = {u: i for i, u in enumerate(cycle)}
    step = {cycle[i]: cycle[(i + 1) % len(cycle)] for i in range(len(cycle))}
    # build fast lookup of label on an edge
    label_lookup = {}
    for u, v, lbl in d.edges():
        a, b = (u, v) if u in pos and v in pos else (None, None)
        if a is None:
            continue
        label_lookup[(u, v)] = lbl
        label_lookup[(v, u)] = lbl
    ok = True
    for i, u in enumerate(cycle):
        v = cycle[(i + 1) % len(cycle)]
        lbl = label_lookup.get((u, v))
        if lbl is None:
            # Not adjacent in adj? Then it's not a valid cycle in region graph built from adj.
            return False
        counts[lbl] += 1
    return all(c % 2 == 0 for c in counts.values())


def expand_by_all_simple_cycles(d: Dual) -> List[Dual]:
    """Skeleton for N→N+1 expansion via region-graph cycles.

    NOTE: This returns an empty list for now (stub construct step). The cycle finder
    and parity check are in place; constructing the new Dual requires splitting edges
    along the chosen cycle and flipping a new bit on one side. That edit sequence is
    nontrivial and will be added next.
    """
    g = d.unlabeled_region_graph()
    cycles = _region_graph_cycles(g)
    out: List[Dual] = []
    for cyc in cycles:
        if not _parity_ok(d, cyc):
            continue
        # TODO: implement edge splitting + L-bit flip + new L-labeled arcs
        # placeholder: skip until construction is coded
        _ = cyc
    return out
