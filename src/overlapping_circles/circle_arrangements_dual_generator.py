from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable, Optional, Set, FrozenSet
import itertools
import collections
import argparse

Mask = int
EdgeLabel = int
RegionId = int


@dataclass(frozen=True)
class Edge:
    u: RegionId
    v: RegionId
    label: EdgeLabel


@dataclass
class Dual:
    N: int
    masks: Dict[RegionId, Mask]
    adj: Dict[RegionId, List[Tuple[RegionId, EdgeLabel]]]

    def _mask_struct_code_fixed_labels(self) -> str:
        """Region-ID-free canonical string based only on masks and label structure.
        Labels are assumed fixed here; use label_invariant_mask_struct_code() to minimize over permutations.
        """
        masks = sorted(m for r, m in self.masks.items() if r != 0)  # omit outside
        header = "VSET:" + ",".join(f"{m:0{self.N}b}" for m in masks)

        # Gather edges as multiplicity-preserving multisets keyed by label
        in_edges: Dict[int, List[Tuple[int, int]]] = collections.defaultdict(list)
        out_edges: Dict[int, List[int]] = collections.defaultdict(list)

        # Build once: region -> mask
        rmask = self.masks

        # Count undirected edges with multiplicity from adj
        counter: Dict[Tuple[int, int, int], int] = collections.Counter()
        for u, neis in self.adj.items():
            for v, lbl in neis:
                a, b = (u, v) if u <= v else (v, u)
                counter[(a, b, lbl)] += 1

        for (a, b, lbl), cnt in counter.items():
            # each undirected edge must appear twice in adj
            mult = cnt // 2
            if a == 0 and b == 0:
                continue
            if a == 0 or b == 0:
                other = b if a == 0 else a
                for _ in range(mult):
                    out_edges[lbl].append(rmask[other])
            else:
                mu, mv = rmask[a], rmask[b]
                lo, hi = (mu, mv) if mu <= mv else (mv, mu)
                for _ in range(mult):
                    in_edges[lbl].append((lo, hi))

        chunks = [header]
        for lbl in range(1, self.N + 1):
            ins = ";".join(
                f"{u:0{self.N}b}-{v:0{self.N}b}"
                for u, v in sorted(in_edges.get(lbl, []))
            )
            outs = ";".join(f"{m:0{self.N}b}" for m in sorted(out_edges.get(lbl, [])))
            chunks.append(f"E{lbl}_in:[{ins}]")
            chunks.append(f"E{lbl}_out:[{outs}]")
        return "|".join(chunks)

    def label_invariant_mask_struct_code(self) -> str:
        """Full label-invariant structural code: minimize mask-struct code across all label permutations."""
        best: Optional[str] = None
        for perm_tuple in itertools.permutations(range(1, self.N + 1)):
            perm = [0] + list(perm_tuple)
            # Relabel masks
            new_masks: Dict[RegionId, Mask] = {}
            for r, m in self.masks.items():
                nm = 0
                for old_label in range(1, self.N + 1):
                    if (m >> (old_label - 1)) & 1:
                        new_l = perm[old_label]
                        nm |= 1 << (new_l - 1)
                new_masks[r] = nm
            # Relabel adjacency labels
            new_adj: Dict[RegionId, List[Tuple[RegionId, EdgeLabel]]] = {
                u: [(v, perm[lbl]) for v, lbl in neis] for u, neis in self.adj.items()
            }
            G = Dual(N=self.N, masks=new_masks, adj=new_adj)
            code = G._mask_struct_code_fixed_labels()
            if best is None or code < best:
                best = code
        assert best is not None
        return best

    # ---- derived ----
    def edges(self) -> List[Edge]:
        """Return undirected multigraph edges with multiplicity preserved."""
        counter: Dict[Tuple[int, int, int], int] = collections.Counter()
        for u, neis in self.adj.items():
            for v, lbl in neis:
                a, b = (u, v) if u <= v else (v, u)
                counter[(a, b, lbl)] += 1
        out: List[Edge] = []
        for (a, b, lbl), cnt in counter.items():
            for _ in range(cnt // 2):
                out.append(Edge(a, b, lbl))
        return out

    # ---- validation ----
    def validate_basic(self) -> None:
        for u, neis in self.adj.items():
            for v, lbl in neis:
                assert any(
                    x == u and l2 == lbl for (x, l2) in self.adj[v]
                ), f"asym edge {(u,v,lbl)}"
        for u, neis in self.adj.items():
            for v, lbl in neis:
                mu, mv = self.masks[u], self.masks[v]
                bit = 1 << (lbl - 1)
                assert (
                    mu ^ mv == bit
                ), f"mask flip rule fails on edge ({u},{v},{lbl}) with {mu:0{self.N}b} vs {mv:0{self.N}b}"

    def validate_circle_cycles(self) -> None:
        by_label = collections.Counter(
            lbl for u, neis in self.adj.items() for _, lbl in neis
        )
        for i in range(1, self.N + 1):
            if by_label[i] == 0:
                raise AssertionError(f"label {i} missing edges")
        regions = list(self.masks.keys())
        seen: Set[int] = set()
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

    # ---- canonicalization ----
    def _canonical_code_fixed_labels(self) -> str:
        verts = [r for r in self.masks if r != 0]
        colors: Dict[RegionId, int] = {r: self.masks[r] for r in verts}
        while True:
            sigs: Dict[RegionId, tuple] = {}
            for r in verts:
                bucket: List[Tuple[int, int]] = []
                for v, lbl in self.adj[r]:
                    if v == 0:
                        bucket.append((lbl, -1))
                    elif v in colors:
                        bucket.append((lbl, colors[v]))
                sigs[r] = (colors[r], tuple(sorted(bucket)))
            mapping: Dict[tuple, int] = {}
            comp: Dict[RegionId, int] = {}
            next_id = 0
            for r in verts:
                key = sigs[r]
                if key not in mapping:
                    mapping[key] = next_id
                    next_id += 1
                comp[r] = mapping[key]
            if comp == colors:
                break
            colors = comp

        buckets: Dict[int, List[RegionId]] = collections.defaultdict(list)
        for r, c in colors.items():
            buckets[c].append(r)
        for b in buckets.values():
            b.sort()
        orderings = [buckets[k] for k in sorted(buckets)]

        def finalize(order: List[RegionId]) -> str:
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
                if e.u not in idx or e.v not in idx:
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
                code = finalize(prefix)
                if best is None or code < best:
                    best = code
                return
            block = orderings[i]
            for perm_order in sorted(
                itertools.permutations(block),
                key=lambda p: [(self.masks[r], len(self.adj[r]), r) for r in p],
            ):
                bt(i + 1, prefix + list(perm_order))

        bt(0, [])
        assert best is not None
        return best

    def canonical_code(self) -> str:
        best: Optional[str] = None
        for perm_tuple in itertools.permutations(range(1, self.N + 1)):
            perm = [0] + list(perm_tuple)
            # Relabel masks and adj directly
            new_masks: Dict[RegionId, Mask] = {}
            for r, m in self.masks.items():
                nm = 0
                for old_label in range(1, self.N + 1):
                    if (m >> (old_label - 1)) & 1:
                        new_l = perm[old_label]
                        nm |= 1 << (new_l - 1)
                new_masks[r] = nm

            new_adj: Dict[RegionId, List[Tuple[RegionId, EdgeLabel]]] = {}
            for u, neis in self.adj.items():
                new_adj[u] = [(v, perm[lbl]) for v, lbl in neis]

            # Evaluate canonical code directly on the relabeled structure
            code = Dual(
                N=self.N, masks=new_masks, adj=new_adj
            )._canonical_code_fixed_labels()

            if best is None or code < best:
                best = code
        assert best is not None
        return best


# --- N=1 and N=2 generation (restored) ---


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
    if G.N != 1:
        raise NotImplementedError("N=1 -> N=2 only in this helper")
    host_mask = G.masks[spec.host_region]
    seq = spec.label_sequence
    if seq == ():
        if host_mask == 0b0:  # disjoint
            masks = {0: 0b00, 1: 0b01, 2: 0b10}
            adj = {0: [(1, 1), (2, 2)], 1: [(0, 1)], 2: [(0, 2)]}
        elif host_mask == 0b1:  # nested (C2 inside C1)
            masks = {0: 0b00, 1: 0b01, 2: 0b11}
            adj = {0: [(1, 1)], 1: [(0, 1), (2, 2)], 2: [(1, 2)]}
        else:
            raise AssertionError
        H = Dual(N=2, masks=masks, adj=adj)
        H.validate()
        return H
    if seq == (1, 1):  # overlap
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
    outs: List[Dual] = []
    seen: Set[str] = set()
    for s in specs:
        H = add_circle_from_spec(G1, s)
        code = H.canonical_code()
        if code not in seen:
            seen.add(code)
            outs.append(H)
    outs.sort(key=lambda g: g.canonical_code())
    return outs


# --- N=2 overlap and N=3 expansions (restored real builders) ---


def dual_N2_overlap() -> Dual:
    return add_circle_from_spec(dual_N1(), NewCircleSpec(0, (1, 1)))


_DEF_OL_MASKS = {0: 0b000, 1: 0b001, 2: 0b010, 3: 0b011}


def _base_overlap_adj() -> Dict[int, List[Tuple[int, int]]]:
    adj = {i: [] for i in _DEF_OL_MASKS}

    def add(u, v, lbl):
        adj.setdefault(u, []).append((v, lbl))
        adj.setdefault(v, []).append((u, lbl))

    add(0, 1, 1)
    add(2, 3, 1)  # C1
    add(0, 2, 2)
    add(1, 3, 2)  # C2
    return adj


# Disjoint/nested: place C3 wholly inside host region h in {0,1,2,3}


def build_n3_disjoint_or_nested(host: int) -> Dual:
    masks = dict(_DEF_OL_MASKS)
    new_id = max(masks) + 1
    masks[new_id] = _DEF_OL_MASKS[host] | 0b100
    adj = _base_overlap_adj()
    adj.setdefault(new_id, [])

    def add(u, v, lbl):
        adj.setdefault(u, []).append((v, lbl))
        adj.setdefault(v, []).append((u, lbl))

    add(host, new_id, 3)
    G = Dual(N=3, masks=masks, adj=adj)
    G.validate()
    return G


# Cross only C1: side_c2 ∈ {0,1}


def build_n3_cross_only_c1(side_c2: int) -> Dual:
    assert side_c2 in (0, 1)
    masks = dict(_DEF_OL_MASKS)
    adj = _base_overlap_adj()

    def add(u, v, lbl):
        adj.setdefault(u, []).append((v, lbl))
        adj.setdefault(v, []).append((u, lbl))

    if side_c2 == 0:
        m0 = max(masks) + 1
        masks[m0] = 0b100
        m1 = max(masks) + 1
        masks[m1] = 0b101
        adj.setdefault(m0, [])
        adj.setdefault(m1, [])
        add(0, m0, 3)
        add(1, m1, 3)
        add(m0, m1, 1)
    else:
        m0 = max(masks) + 1
        masks[m0] = 0b110
        m1 = max(masks) + 1
        masks[m1] = 0b111
        adj.setdefault(m0, [])
        adj.setdefault(m1, [])
        add(2, m0, 3)
        add(3, m1, 3)
        add(m0, m1, 1)
    G = Dual(N=3, masks=masks, adj=adj)
    G.validate()
    return G


# Cross only C2: side_c1 ∈ {0,1}


def build_n3_cross_only_c2(side_c1: int) -> Dual:
    assert side_c1 in (0, 1)
    masks = dict(_DEF_OL_MASKS)
    adj = _base_overlap_adj()

    def add(u, v, lbl):
        adj.setdefault(u, []).append((v, lbl))
        adj.setdefault(v, []).append((u, lbl))

    if side_c1 == 0:
        m0 = max(masks) + 1
        masks[m0] = 0b100
        m1 = max(masks) + 1
        masks[m1] = 0b110
        adj.setdefault(m0, [])
        adj.setdefault(m1, [])
        add(0, m0, 3)
        add(2, m1, 3)
        add(m0, m1, 2)
    else:
        m0 = max(masks) + 1
        masks[m0] = 0b101
        m1 = max(masks) + 1
        masks[m1] = 0b111
        adj.setdefault(m0, [])
        adj.setdefault(m1, [])
        add(1, m0, 3)
        add(3, m1, 3)
        add(m0, m1, 2)
    G = Dual(N=3, masks=masks, adj=adj)
    G.validate()
    return G


# Full 3-circle Venn


def build_n3_venn() -> Dual:
    masks = {i: i for i in range(8)}
    adj: Dict[int, List[Tuple[int, int]]] = {i: [] for i in masks}

    def add(u, v, lbl):
        adj.setdefault(u, []).append((v, lbl))
        adj.setdefault(v, []).append((u, lbl))

    for lbl in (1, 2, 3):
        bit = 1 << (lbl - 1)
        for m in range(8):
            n = m ^ bit
            if m < n:
                add(m, n, lbl)
    G = Dual(N=3, masks=masks, adj=adj)
    G.validate()
    return G


# Public API with logging


def enumerate_all_n3_from_overlap(log: bool = False) -> List[Dual]:
    candidates: List[Tuple[str, Dual]] = []
    # disjoint/nested placements
    for host, name in [
        (0, "disjoint"),
        (1, "nested_01"),
        (2, "nested_10"),
        (3, "nested_11"),
    ]:
        candidates.append((name, build_n3_disjoint_or_nested(host)))
    # cross-only
    for side in (0, 1):
        candidates.append((f"cross_C1_sideC2={side}", build_n3_cross_only_c1(side)))
        candidates.append((f"cross_C2_sideC1={side}", build_n3_cross_only_c2(side)))
    # venn
    candidates.append(("venn", build_n3_venn()))

    if log:
        print(f"[pre-dedup] generated ({len(candidates)} candidates):")
        for tag, g in candidates:
            print(f" - {tag}: {g.canonical_code()}")

    dedup: Dict[str, Dual] = {}
    for _tag, g in candidates:
        key = g.label_invariant_mask_struct_code()
        dedup[key] = g

    if log:
        print(f"[post-dedup] unique ({len(dedup)}):")
        for k in sorted(dedup):
            print(f" - {k}")

    return [dedup[k] for k in sorted(dedup)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", action="store_true", help="Enable detailed logging")
    args = parser.parse_args()

    print("N=1 base arrangement (canonical code):")
    G1 = dual_N1()
    print(G1.canonical_code())

    print("\nEnumerated N=2 arrangements (canonical codes):")
    for G2 in enumerate_N2_from_N1():
        print(G2.canonical_code())

    print("\nFrom N=2 overlap → all N=3 expansions (canonical codes):")
    results = enumerate_all_n3_from_overlap(log=args.log)
    for G3 in results:
        print(G3.canonical_code())
