from __future__ import annotations
import warnings

from overlapping_circles_core.expand import (
    expand_by_all_simple_cycles,
    expand_generically,
)

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r"'overlapping_circles_core\.generators' found in sys\.modules",
)
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set
import collections
import itertools
import argparse

from .dual import Dual

# --- N=1 and N=2 generation (extracted from your original file) ---


def dual_N1() -> Dual:
    # Two regions: outside (0) and inside (1)
    masks = {0: 0b0, 1: 0b1}
    adj = {0: [(1, 1)], 1: [(0, 1)]}
    boundary = {
        0: [(1, 1)],  # circle 1 separates 0↔1
        1: [(0, 1)],
    }
    return Dual(N=1, masks=masks, adj=adj, boundary=boundary)


@dataclass
class NewCircleSpec:
    host_region: int
    label_sequence: Tuple[int, ...]


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


def derive_boundary_from_adj(H: Dual) -> Dict[int, List[Tuple[int, int]]]:
    """
    Derive a plausible cyclic boundary order (rotation system) for a Dual
    purely from its adjacency and region-mask structure.
    Works generically for small N (e.g. N=2) and extends later.
    """
    boundary: Dict[int, List[Tuple[int, int]]] = {}
    for r, arcs in H.adj.items():

        def sort_key(av: Tuple[int, int]):
            v, lbl = av
            inside_r = (H.masks[r] >> (lbl - 1)) & 1
            inside_v = (H.masks[v] >> (lbl - 1)) & 1
            # 0 = outside→inside transition, 1 = inside→outside
            orientation = 0 if inside_r < inside_v else 1
            return (lbl, orientation)

        ordered = sorted(arcs, key=sort_key)
        boundary[r] = ordered
    return boundary


def enumerate_N2_from_N1() -> List[Dual]:
    G1 = dual_N1()
    specs = [NewCircleSpec(0, ()), NewCircleSpec(1, ()), NewCircleSpec(0, (1, 1))]
    outs: List[Dual] = []
    seen: Set[str] = set()
    for s in specs:
        H = add_circle_from_spec(G1, s)
        boundary = derive_boundary_from_adj(H)
        H = H.with_boundary(boundary)

        code = H.canonical_code()
        if code not in seen:
            seen.add(code)
            outs.append(H)
    outs.sort(key=lambda g: g.canonical_code())
    return outs


# --- N=2 overlap and N=3 expansions (extracted) ---


def dual_N2_overlap() -> Dual:
    return add_circle_from_spec(dual_N1(), NewCircleSpec(0, (1, 1)))


_DEF_OL_MASKS = {0: 0b000, 1: 0b001, 2: 0b010, 3: 0b011}


def _base_overlap_adj() -> Dict[int, List[Tuple[int, int]]]:
    adj = {i: [] for i in _DEF_OL_MASKS}

    def add(u, v, lbl):
        adj.setdefault(u, []).append((v, lbl))
        adj.setdefault(v, []).append((u, lbl))

    add(0, 1, 1)
    add(0, 2, 2)
    add(1, 3, 2)
    add(2, 3, 1)
    return adj


def build_n3_disjoint_or_nested(host: int) -> Dual:
    assert host in (0, 1, 2, 3)
    masks = dict(_DEF_OL_MASKS)
    adj = _base_overlap_adj()

    def add(u, v, lbl):
        adj.setdefault(u, []).append((v, lbl))
        adj.setdefault(v, []).append((u, lbl))

    if host == 0:
        m0 = max(masks) + 1
        masks[m0] = 0b100
        m1 = max(masks) + 1
        masks[m1] = 0b110
        adj.setdefault(m0, [])
        adj.setdefault(m1, [])
        add(0, m0, 3)
        add(2, m1, 3)
        add(m0, m1, 2)
    elif host == 1:
        m0 = max(masks) + 1
        masks[m0] = 0b100
        m1 = max(masks) + 1
        masks[m1] = 0b101
        adj.setdefault(m0, [])
        adj.setdefault(m1, [])
        add(0, m0, 3)
        add(1, m1, 3)
        add(m0, m1, 1)
    elif host == 2:
        m0 = max(masks) + 1
        masks[m0] = 0b110
        m1 = max(masks) + 1
        masks[m1] = 0b111
        adj.setdefault(m0, [])
        adj.setdefault(m1, [])
        add(2, m0, 3)
        add(3, m1, 3)
        add(m0, m1, 1)
    elif host == 3:
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


# Full 3‑circle Venn


def build_n3_venn() -> Dual:
    # Regions 0..7 mapped to binary masks 000..111
    masks: Dict[int, int] = {i: i for i in range(8)}
    adj: Dict[int, List[Tuple[int, int]]] = {i: [] for i in masks}

    def add(u: int, v: int, lbl: int) -> None:
        adj[u].append((v, lbl))
        adj[v].append((u, lbl))

    # Add an edge between every pair of regions that differ in exactly one bit.
    for u in range(8):
        for v in range(u + 1, 8):
            diff = masks[u] ^ masks[v]
            if diff and (diff & (diff - 1)) == 0:  # exactly one bit set
                lbl = diff.bit_length()  # 1 for 001, 2 for 010, 3 for 100
                add(u, v, lbl)

    G = Dual(N=3, masks=masks, adj=adj)
    G.validate()
    return G


def enumerate_all_n3_from_overlap(log: bool = False) -> List[Dual]:
    candidates: List[Tuple[str, Dual]] = []
    # disjoint/nested based on host region
    for host, name in [
        (0, "disjoint"),
        (1, "nested_01"),
        (2, "nested_10"),
        (3, "nested_11"),
    ]:
        candidates.append((name, build_n3_disjoint_or_nested(host)))
    # cross‑only
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


def enumerate_all_n3_generic(log: bool = False) -> List[Dual]:
    n2 = enumerate_N2_from_N1()
    out: Dict[str, Dual] = {}
    for G2 in n2:
        exps = expand_generically(G2)
        if log:
            print(f"{G2.canonical_code()} → {len(exps)} expansions")
        for H in exps:
            out.setdefault(H.canonical_code(), H)
    if log:
        print(f"[post-dedup] unique N=3: {len(out)}")
    return [out[k] for k in sorted(out)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", action="store_true", help="Enable detailed logging")
    args = parser.parse_args()

    # --- N = 1 ---
    print("N=1 base arrangement (canonical code):")
    G1 = dual_N1()
    print(G1.canonical_code())

    # --- N = 2 ---
    print("\nEnumerated N=2 arrangements (canonical codes):")
    all_n2 = enumerate_N2_from_N1()
    for G2 in all_n2:
        print(G2.canonical_code())

    # --- N = 3 ---
    print("\nEnumerated N=3 arrangements (canonical codes):")
    all_n3 = enumerate_all_n3_generic(log=args.log)
    for G3 in all_n3:
        print(G3.canonical_code())

    print(f"\n[summary] generated {len(all_n3)} unique N=3 arrangements")


if __name__ == "__main__":
    main()
