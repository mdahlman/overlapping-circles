from __future__ import annotations
from typing import List, Tuple, Dict, Set, Mapping, Iterable
import collections

from .dual import Dual


def _region_graph_cycles(g: Mapping[int, Iterable[int]]) -> List[List[int]]:
    # Johnson-like simple cycle enumeration (undirected). Minimal for now.
    nodelist = sorted(g)
    DG = {u: sorted(set(g[u])) for u in nodelist}
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
            if w < s:
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

    for s in nodelist:
        DG2 = {u: [w for w in DG[u] if w >= s] for u in DG}
        for u in DG:
            blocked[u] = False
            B[u].clear()
        DG = DG2
        circuit(s, s)

    normed = []
    seen = set()
    for cyc in cycles:
        cyc = cyc[:-1]
        m = min(cyc)
        i = cyc.index(m)
        rot = cyc[i:] + cyc[:i]
        r1 = rot
        r2 = [rot[0]] + list(reversed(rot[1:]))
        best = r1 if tuple(r1) < tuple(r2) else r2
        t = tuple(best)
        if t not in seen:
            seen.add(t)
            normed.append(best)
    return normed


def _unlabeled_from(d: Dual) -> Dict[int, List[int]]:
    # adjacency by region, ignoring labels; keep neighbors sorted for determinism
    return {u: sorted({v for (v, _lbl) in d.adj.get(u, [])}) for u in d.masks.keys()}


def expand_by_all_simple_cycles(d: Dual) -> List[Dual]:
    """Enumerate all N+1 arrangements by inserting a new circle along each simple cycle."""
    results: List[Dual] = []
    g = _unlabeled_from(d)
    cycles = _region_graph_cycles(g)
    next_label = d.N + 1

    seen_edges = set()

    for cyc in cycles:
        # skip degenerate or duplicated cycles
        if len(cyc) < 3 or cyc[0] > min(cyc[1:]):
            continue  # canonicalize orientation

        masks = dict(d.masks)
        adj = {k: list(v) for k, v in d.adj.items()}

        max_r = max(masks)
        new_ids = {}
        for r in cyc:
            max_r += 1
            new_id = max_r
            new_ids[r] = new_id
            masks[new_id] = masks[r] | (1 << (next_label - 1))
            adj[new_id] = []

        # build new-circle edges once per boundary edge
        edge_pairs = set()
        for i in range(len(cyc)):
            u = cyc[i]
            v = cyc[(i + 1) % len(cyc)]
            edge_pairs.add(tuple(sorted((u, v))))

        for u, v in edge_pairs:
            nu, nv = new_ids[u], new_ids[v]
            # connect original↔new copies once
            adj[u].append((nu, next_label))
            adj[nu].append((u, next_label))
            adj[v].append((nv, next_label))
            adj[nv].append((v, next_label))

        try:
            Gnew = Dual(N=next_label, masks=masks, adj=adj)
            Gnew.validate()
            results.append(Gnew)
        except AssertionError:
            continue

    # dedup by canonical code
    uniq: Dict[str, Dual] = {}
    for g in results:
        key = g.canonical_code()
        uniq.setdefault(key, g)
    return list(uniq.values())


def embedding_cycles(
    boundary: Dict[int, List[Tuple[int, int]]],
) -> list[list[Tuple[int, int, int]]]:
    """
    Traverse the rotation system (boundary) of a Dual to extract all directed
    edge cycles ("darts"). Each undirected edge (u,v,lbl) appears twice, once
    per direction, and each directed arc appears in exactly one returned cycle.

    Returns:
        List of cycles, each a list of (u, v, lbl) triples forming a simple closed
        loop in the embedding.
    """
    seen = set()
    cycles = []

    for u, arcs in boundary.items():
        for v, lbl in arcs:
            dart = (u, v, lbl)
            if dart in seen:
                continue
            cycle = []
            cur = dart
            while cur not in seen:
                seen.add(cur)
                cycle.append(cur)
                cu, cv, cl = cur
                # find reverse (cv, cu, cl) in cv's cyclic order
                b = boundary[cv]
                try:
                    j = b.index((cu, cl))
                except ValueError:
                    raise AssertionError(
                        f"Edge ({cu},{cv},{cl}) missing reverse in boundary[{cv}]"
                    )
                # next dart is the successor of the reverse arc in cv's cyclic order
                nxt = b[(j + 1) % len(b)]
                cur = (cv, nxt[0], nxt[1])
            cycles.append(cycle)
    return cycles


def _edge_label(d: Dual, u: int, v: int) -> int:
    """Label of the unique edge between u and v (assert uniqueness)."""
    labs = [lbl for (w, lbl) in d.adj[u] if w == v]
    assert labs, f"no edge {u}-{v}"
    # disallow multi-edges for small N; if you allow them, pick the one used by cycle
    assert len(labs) == 1, f"multi-edge {u}-{v} not supported yet"
    return labs[0]


def _ccw_interval(
    boundary: List[Tuple[int, int]], a: Tuple[int, int], b: Tuple[int, int]
) -> Set[Tuple[int, int]]:
    """
    Return the open CCW sector between arc 'a' and arc 'b' in 'boundary'.
    'boundary' is a cyclic list of (neighbor,label) in CCW order.
    """
    n = len(boundary)
    ia = boundary.index(a)
    ib = boundary.index(b)
    out = set()
    i = (ia + 1) % n
    while i != ib:
        out.add(boundary[i])
        i = (i + 1) % n
    return out


def insert_circle_along_region_cycle(d: Dual, cycle: List[int]) -> Dual:
    """
    Insert new circle (label N+1) guided by a region cycle.
    Supports:
      - len(cycle) == 1  (zero-cross: circle entirely inside one region)
      - len(cycle) == 2  (two-cross: cross one existing edge twice)
      - len(cycle) >= 3  (multi-cross along a simple cycle)
    Requires d.boundary (CCW rotation system).
    """
    assert (
        d.boundary is not None
    ), "insert_circle_along_region_cycle requires d.boundary"
    N1 = d.N + 1
    newbit = 1 << (N1 - 1)

    masks = dict(d.masks)
    adj: Dict[int, List[Tuple[int, int]]] = {u: list(neis) for u, neis in d.adj.items()}
    boundary: Dict[int, List[Tuple[int, int]]] = {
        u: list(arcs) for u, arcs in d.boundary.items()
    }

    # ---- Case A: zero-cross (len == 1) ----
    if len(cycle) == 1:
        r = cycle[0]
        # duplicate region r -> r_in with new bit set
        new_id = max(masks) + 1 if masks else 1
        masks[new_id] = masks[r] | newbit
        adj.setdefault(new_id, [])
        boundary[new_id] = []

        # add the new circle edge between r and r_in
        adj[r].append((new_id, N1))
        adj[new_id].append((r, N1))
        boundary[r].append((new_id, N1))
        boundary[new_id].append((r, N1))

        Gnew = Dual(N=N1, masks=masks, adj=adj, boundary=boundary)
        Gnew.validate()
        return Gnew

    # ---- Case B/C: len >= 2 ----
    k = len(cycle)
    # record labels on steps
    steps = []
    for i in range(k):
        u = cycle[i]
        v = cycle[(i + 1) % k]
        lbl = _edge_label(d, u, v)
        steps.append((u, v, lbl))

    # choose inside sector at each visited region using CCW interval
    inside_sector: Dict[int, Set[Tuple[int, int]]] = {}
    if k == 2:
        # Two-cross on a single undirected edge {u,v}: pick a consistent interior
        u, v = cycle[0], cycle[1]
        lu = _edge_label(d, u, v)
        lv = lu
        arcs_u = boundary[u]
        arcs_v = boundary[v]
        # The two darts at u are (v,lu) and its reverse neighbor in arcs_u.
        # Define "inside" as the open sector from (v,lu) to its CCW successor.
        j_u = arcs_u.index((v, lu))
        a_prev_u = (v, lu)
        a_next_u = arcs_u[(j_u + 1) % len(arcs_u)]
        inside_sector[u] = _ccw_interval(arcs_u, a_prev_u, a_next_u)

        j_v = arcs_v.index((u, lv))
        a_prev_v = (u, lv)
        a_next_v = arcs_v[(j_v + 1) % len(arcs_v)]
        inside_sector[v] = _ccw_interval(arcs_v, a_prev_v, a_next_v)
    else:
        # k >= 3
        for i in range(k):
            r = cycle[i]
            p = cycle[(i - 1) % k]
            n = cycle[(i + 1) % k]
            lp = _edge_label(d, r, p)
            ln = _edge_label(d, r, n)
            arcs = boundary[r]
            inside_sector[r] = _ccw_interval(arcs, (p, lp), (n, ln))

    # duplicate each visited region: r_out = r; r_in = new id
    max_r = max(masks) if masks else 0
    in_id: Dict[int, int] = {}
    for r in cycle:
        max_r += 1
        in_id[r] = max_r
        masks[max_r] = masks[r] | newbit
        adj.setdefault(max_r, [])
        boundary[max_r] = []

    # helper: is the arc a cycle arc at r?
    cycle_neighbors = {
        r: {cycle[(i - 1) % k], cycle[(i + 1) % k]} for i, r in enumerate(cycle)
    }
    cycle_labels = {
        (
            min(cycle[i], cycle[(i + 1) % k]),
            max(cycle[i], cycle[(i + 1) % k]),
        ): _edge_label(d, cycle[i], cycle[(i + 1) % k])
        for i in range(k)
    }

    # redistribute non-cycle arcs to in/out per inside_sector; remove cycle arcs for now
    original_arcs = {r: list(boundary[r]) for r in cycle}
    for r in cycle:
        boundary[r] = []
        for v, lbl in original_arcs[r]:
            if v in cycle_neighbors[r] and lbl == cycle_labels[(min(r, v), max(r, v))]:
                # skip; to be split after
                continue
            if (v, lbl) in inside_sector[r]:
                # attach to r_in; also edit neighbor's reference
                boundary[in_id[r]].append((v, lbl))
                # neighbor boundary
                vb = boundary[v]
                j = vb.index((r, lbl))
                vb[j] = (in_id[r], lbl)
                # adjacency
                adj[in_id[r]].append((v, lbl))
                adj[v] = [
                    (in_id[r] if (x == r and l == lbl) else x, l) for (x, l) in adj[v]
                ]
            else:
                boundary[r].append((v, lbl))
                # adjacency for r already there; neighbor unchanged

    # split cycle arcs into out-out and in-in copies
    handled: Set[Tuple[int, int]] = set()
    for i in range(k):
        u = cycle[i]
        v = cycle[(i + 1) % k]
        key = (min(u, v), max(u, v))
        if key in handled:
            continue
        handled.add(key)
        l = cycle_labels[key]

        # remove old u-v arc from both
        if (v, l) in boundary[u]:
            boundary[u].remove((v, l))
        if (u, l) in boundary[v]:
            boundary[v].remove((u, l))
        adj[u] = [(w, lbl) for (w, lbl) in adj[u] if not (w == v and lbl == l)]
        adj[v] = [(w, lbl) for (w, lbl) in adj[v] if not (w == u and lbl == l)]

        # add outside copy
        boundary[u].append((v, l))
        boundary[v].append((u, l))
        adj[u].append((v, l))
        adj[v].append((u, l))

        # add inside copy
        ui, vi = in_id[u], in_id[v]
        boundary[ui].append((vi, l))
        boundary[vi].append((ui, l))
        adj[ui].append((vi, l))
        adj[vi].append((ui, l))

    # add the new circle edges between r and r_in
    for r in cycle:
        ri = in_id[r]
        boundary[r].append((ri, N1))
        boundary[ri].append((r, N1))
        adj[r].append((ri, N1))
        adj[ri].append((r, N1))

    Gnew = Dual(N=N1, masks=masks, adj=adj, boundary=boundary)
    Gnew.validate()
    return Gnew


def expand_generically(d: Dual, log: bool = False) -> List[Dual]:
    """Enumerate all N+1 expansions by inserting a new circle along:
    - every single region (zero-cross),
    - every undirected edge as a 2-cycle (two-cross),
    - every simple region cycle (multi-cross)."""
    assert d.boundary is not None, "expand_generically requires d.boundary"

    candidate_idx = 1

    # zero-cross on each region
    cycles: List[List[int]] = [[r] for r in sorted(d.masks.keys())]

    # two-cross on each undirected edge
    seen_edges: Set[Tuple[int, int]] = set()
    for u, neis in d.adj.items():
        for v, _ in neis:
            a, b = (u, v) if u < v else (v, u)
            if (a, b) not in seen_edges:
                seen_edges.add((a, b))
                cycles.append([a, b])

    # multi-cross cycles from region graph
    g = _unlabeled_from(d)
    for cyc in _region_graph_cycles(g):
        if len(cyc) >= 3:
            cycles.append(cyc)

    results: Dict[str, Dual] = {}
    for cyc in cycles:
        try:
            H = insert_circle_along_region_cycle(d, cyc)
            if log:
                print(
                    f"{candidate_idx}  cycle {cyc}:\n   {H.current_code()}\n → {H.canonical_code()}"
                )
                candidate_idx += 1
            results.setdefault(H.canonical_code(), H)
        except AssertionError:
            continue
    return [results[k] for k in sorted(results)]
