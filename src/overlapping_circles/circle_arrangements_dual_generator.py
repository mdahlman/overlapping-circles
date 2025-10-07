from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable, Optional, Set, FrozenSet
import itertools
import collections

Mask = int
EdgeLabel = int
RegionId = int


@dataclass(frozen=True)
class Edge:
    u: RegionId
    v: RegionId
    label: EdgeLabel

    def key(self) -> Tuple[int, int, int]:
        a, b = sorted((self.u, self.v))
        return (a, b, self.label)


@dataclass
class Dual:
    N: int
    masks: Dict[RegionId, Mask]
    adj: Dict[RegionId, List[Tuple[RegionId, EdgeLabel]]]

    def regions(self) -> List[RegionId]:
        return list(self.masks.keys())

    def edges(self) -> List[Edge]:
        counter: Dict[Tuple[int, int, int], int] = collections.Counter()
        for u, neis in self.adj.items():
            for v, lbl in neis:
                a, b = sorted((u, v))
                counter[(a, b, lbl)] += 1
        out: List[Edge] = []
        for (a, b, lbl), cnt in counter.items():
            for _ in range(cnt // 2):
                out.append(Edge(a, b, lbl))
        return out

    def validate_basic(self) -> None:
        for u, neis in self.adj.items():
            for v, lbl in neis:
                assert any(
                    x == u and l2 == lbl for (x, l2) in self.adj[v]
                ), f"asym edge {(u,v,lbl)}"
        for e in self.edges():
            mu, mv = self.masks[e.u], self.masks[e.v]
            bit = 1 << (e.label - 1)
            assert (
                mu ^ mv == bit
            ), f"mask flip rule fails on edge {(e.u,e.v,e.label)} with {mu:0{self.N}b} vs {mv:0{self.N}b}"

    def validate_circle_cycles(self) -> None:
        by_label = collections.Counter(e.label for e in self.edges())
        for i in range(1, self.N + 1):
            if by_label[i] == 0:
                raise AssertionError(f"label {i} has no edges (broken circle)")
        seen: Set[int] = set()
        regions = self.regions()
        stack = [regions[0]]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            for y, _ in self.adj[x]:
                if y not in seen:
                    stack.append(y)
        if set(regions) != seen:
            raise AssertionError("dual not connected")

    def validate(self) -> None:
        self.validate_basic()
        self.validate_circle_cycles()

    def canonical_code(self) -> str:
        """Canonical code omitting the outside region (mask 0...0) from V: but
        preserving edges to the outside explicitly as Ei_out:[...].
        Edge/mask labels and multiplicities are preserved. Circle labels are fixed.
        """
        verts = [r for r in self.regions() if r != 0]
        # --- 1-WL refinement including outside-adjacency as a special color (-1) ---
        colors: Dict[RegionId, int] = {r: self.masks[r] for r in verts}
        while True:
            sigs: Dict[RegionId, tuple] = {}
            for r in verts:
                bucket: List[Tuple[int, int]] = []
                for v, lbl in self.adj[r]:
                    if v == 0:
                        bucket.append((lbl, -1))  # outside as special color
                    elif v in colors:
                        bucket.append((lbl, colors[v]))
                sigs[r] = (colors[r], tuple(sorted(bucket)))
            # compress to integers
            mapping: Dict[tuple, int] = {}
            next_id = 0
            comp: Dict[RegionId, int] = {}
            for r in verts:
                key = sigs[r]
                if key not in mapping:
                    mapping[key] = next_id
                    next_id += 1
                comp[r] = mapping[key]
            if comp == colors:
                break
            colors = comp
        # --- buckets by color ---
        buckets: Dict[int, List[RegionId]] = collections.defaultdict(list)
        for r, c in colors.items():
            buckets[c].append(r)
        for b in buckets.values():
            b.sort()
        orderings: List[List[RegionId]] = [buckets[k] for k in sorted(buckets)]

        # --- encoder with explicit in/out per label ---
        def encode(order: List[RegionId]) -> str:
            idx = {r: i for i, r in enumerate(order)}
            vertex_part = "V:" + ",".join(f"{self.masks[r]:0{self.N}b}" for r in order)
            in_edges: Dict[int, List[Tuple[int, int]]] = collections.defaultdict(list)
            out_edges: Dict[int, List[int]] = collections.defaultdict(list)
            for e in self.edges():
                if e.u == 0 and e.v == 0:
                    continue
                if e.u == 0 or e.v == 0:
                    other = e.v if e.u == 0 else e.u
                    if other in idx:
                        out_edges[e.label].append(idx[other])
                    continue
                a, b = idx[e.u], idx[e.v]
                if a > b:
                    a, b = b, a
                in_edges[e.label].append((a, b))
            chunks = []
            for lbl in range(1, self.N + 1):
                pairs = sorted(in_edges.get(lbl, []))
                outs = sorted(out_edges.get(lbl, []))
                in_str = ";".join(f"{a}-{b}" for a, b in pairs)
                out_str = ";".join(str(x) for x in outs)
                chunks.append(f"E{lbl}_in:[" + in_str + "]")
                chunks.append(f"E{lbl}_out:[" + out_str + "]")
            return vertex_part + "|" + "|".join(chunks)

        best: Optional[str] = None

        def bt(i: int, prefix: List[RegionId]):
            nonlocal best
            if i == len(orderings):
                code = encode(prefix)
                if best is None or code < best:
                    best = code
                return
            block = orderings[i]
            for perm in sorted(
                itertools.permutations(block),
                key=lambda p: [(self.masks[r], len(self.adj[r]), r) for r in p],
            ):
                bt(i + 1, prefix + list(perm))

        bt(0, [])
        assert best is not None
        return best


def dual_N1() -> Dual:
    masks = {0: 0b0, 1: 0b1}
    adj = {0: [(1, 1), (1, 1)], 1: [(0, 1), (0, 1)]}
    G = Dual(N=1, masks=masks, adj=adj)
    G.validate()
    return G


@dataclass
class NewCircleSpec:
    host_region: RegionId
    label_sequence: Tuple[EdgeLabel, ...]


def add_circle_from_spec(G: Dual, spec: NewCircleSpec) -> Dual:
    assert spec.host_region in G.masks
    N_new = G.N + 1
    if G.N != 1:
        raise NotImplementedError
    host_mask = G.masks[spec.host_region]
    seq = spec.label_sequence
    if seq == ():
        if host_mask == 0b0:
            masks = {0: 0b00, 1: 0b01, 2: 0b10}
            adj = {0: [(1, 1), (2, 2)], 1: [(0, 1)], 2: [(0, 2)]}
        elif host_mask == 0b1:
            masks = {0: 0b00, 1: 0b01, 2: 0b11}
            adj = {0: [(1, 1)], 1: [(0, 1), (2, 2)], 2: [(1, 2)]}
        else:
            raise AssertionError
        H = Dual(N=2, masks=masks, adj=adj)
        H.validate()
        return H
    if seq == (1, 1):
        masks = {0: 0b00, 1: 0b01, 2: 0b10, 3: 0b11}
        adj = {
            0: [(1, 1), (2, 2)],
            1: [(0, 1), (3, 2)],
            2: [(0, 2), (3, 1)],
            3: [(2, 1), (1, 2)],
        }
        H = Dual(N=2, masks=masks, adj=adj)
        H.validate()
        return H
    raise ValueError


def enumerate_N2_from_N1() -> List[Dual]:
    G1 = dual_N1()
    specs = [NewCircleSpec(0, ()), NewCircleSpec(1, ()), NewCircleSpec(0, (1, 1))]
    outs = []
    seen = set()
    for s in specs:
        H = add_circle_from_spec(G1, s)
        code = H.canonical_code()
        if code not in seen:
            seen.add(code)
            outs.append(H)
    outs.sort(key=lambda g: g.canonical_code())
    return outs


# --- Helpers to build a specific N=2 (overlap) and expand to a few N=3 examples ---


def dual_N2_overlap() -> Dual:
    # Reuse the builder from add_circle_from_spec
    G1 = dual_N1()
    H = add_circle_from_spec(G1, NewCircleSpec(0, (1, 1)))
    return H


# Three N=3 expansions from N=2 overlap:
#  A) C3 disjoint from both (placed in the outside region 000)
#  B) C3 crosses only C1 (two crossings), positioned in the half with C2=0
#  C) Full 3-circle Venn (C3 crosses C1 and C2 twice, pattern 1–2–1–2)


def n3_from_overlap_disjoint() -> Dual:
    # Regions (exclude 000 in code, but we store it here for adj construction):
    # 000 outside, 001(C1 only),010(C2 only),011(C1&C2), plus 100 (C3 only)
    masks = {
        0: 0b000,  # outside
        1: 0b001,  # C1 only
        2: 0b010,  # C2 only
        3: 0b011,  # C1&C2
        4: 0b100,  # C3 only
    }
    # Build adj via known boundaries from overlap + new C3-outside boundary
    adj = {i: [] for i in masks}

    def add(u, v, lbl):
        adj[u].append((v, lbl))
        adj[v].append((u, lbl))

    # Circle 1 edges: 000-001 and 010-011
    add(0, 1, 1)
    add(2, 3, 1)
    # Circle 2 edges: 000-010 and 001-011
    add(0, 2, 2)
    add(1, 3, 2)
    # Circle 3 edges: 000-100
    add(0, 4, 3)
    G = Dual(N=3, masks=masks, adj=adj)
    G.validate()
    return G


def n3_from_overlap_cross_C1_only() -> Dual:
    # Split only the C2=0 side by C3: produce 100 and 101; leave C2=1 side unchanged
    masks = {
        0: 0b000,  # outside
        1: 0b001,  # C1 only
        2: 0b010,  # C2 only
        3: 0b011,  # C1&C2
        4: 0b100,  # C3 only
        5: 0b101,  # C1&C3
    }
    adj = {i: [] for i in masks}

    def add(u, v, lbl):
        adj[u].append((v, lbl))
        adj[v].append((u, lbl))

    # Circle 1 edges (as in overlap) but extended where present: 000-001, 010-011, 100-101
    add(0, 1, 1)
    add(2, 3, 1)
    add(4, 5, 1)
    # Circle 2 edges (as in overlap): 000-010, 001-011
    add(0, 2, 2)
    add(1, 3, 2)
    # Circle 3 edges (only on C2=0 side): 000-100, 001-101
    add(0, 4, 3)
    add(1, 5, 3)
    G = Dual(N=3, masks=masks, adj=adj)
    G.validate()
    return G


def n3_from_overlap_venn() -> Dual:
    # All 8 masks (include 000 outside), full Venn
    masks = {i: i for i in range(8)}  # 0..7 with bit0=C1, bit1=C2, bit2=C3
    adj = {i: [] for i in masks}

    def add(u, v, lbl):
        adj[u].append((v, lbl))
        adj[v].append((u, lbl))

    # For each circle i, connect pairs that differ exactly in bit (i-1)
    def edges_for_label(lbl):
        bit = 1 << (lbl - 1)
        for m in range(8):
            n = m ^ bit
            if m < n:  # add each undirected edge once
                add(m, n, lbl)

    edges_for_label(1)
    edges_for_label(2)
    edges_for_label(3)
    G = Dual(N=3, masks=masks, adj=adj)
    G.validate()
    return G


def demo_expand_from_overlap_to_n3() -> List[Dual]:
    outs = [
        n3_from_overlap_disjoint(),
        n3_from_overlap_cross_C1_only(),
        n3_from_overlap_venn(),
    ]
    # dedup and order by canonical code
    dedup = {}
    for g in outs:
        dedup[g.canonical_code()] = g
    return [dedup[k] for k in sorted(dedup)]


if __name__ == "__main__":
    print("N=1 base arrangement (canonical code):")
    G1 = dual_N1()
    print(G1.canonical_code())
    print("Enumerated N=2 arrangements (canonical codes):")
    for G2 in enumerate_N2_from_N1():
        print(G2.canonical_code())
    print("From N=2 overlap → sample N=3 expansions (canonical codes):")
    for G3 in demo_expand_from_overlap_to_n3():
        print(G3.canonical_code())
