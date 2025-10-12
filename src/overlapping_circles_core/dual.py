from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable, Optional, Set, FrozenSet
import itertools
import collections
import argparse
import re

Mask = int
EdgeLabel = int
RegionId = int


def _bitstr(x: int, N: int) -> str:
    return format(x, f"0{N}b")


@dataclass(frozen=True)
class Edge:
    u: RegionId
    v: RegionId
    lbl: EdgeLabel


@dataclass(frozen=True)
class Dual:
    """
    Combinatorial dual structure for a circle arrangement (geometry‑free).

    - masks: region_id -> bitmask of circle membership
    - adj: adjacency list with edge labels: region -> list[(neighbor, label)]

    Validation and canonicalization methods are kept here exactly as in the
    original implementation you uploaded, so behavior and canonical strings
    are preserved. Labels are assumed fixed inside `_canonical_code_fixed_labels()`;
    use `label_invariant_mask_struct_code()` / `canonical_code()` to minimize
    over permutations.
    """

    N: int
    masks: Dict[RegionId, Mask]
    adj: Dict[RegionId, List[Tuple[RegionId, EdgeLabel]]]
    boundary: Optional[Dict[RegionId, List[Tuple[RegionId, EdgeLabel]]]] = field(
        default=None, compare=False, hash=False, repr=False
    )

    # ---- inexpensive helpers / inspectors (verbatim behavior) ----
    def _mask_struct_code_fixed_labels(self) -> str:
        """
        Generate a structural code string for this dual with fixed labels.

        Format: "V:mask1,mask2,...|E1_in:[edge_pairs]|E1_out:[boundary_regions]|..."
        where edge_pairs are "inside_region-outside_region" for each circle label.
        """
        regions = sorted(self.masks)
        header = "V:" + ",".join(_bitstr(self.masks[r], self.N) for r in regions)

        # Count undirected edge multiplicities: (u, v, label) -> count
        edge_counter: Dict[Tuple[int, int, int], int] = collections.Counter()
        for u, neighbors in self.adj.items():
            for v, label in neighbors:
                # Normalize edge to (smaller_id, larger_id, label)
                a, b = (u, v) if u <= v else (v, u)
                edge_counter[(a, b, label)] += 1

        # Separate edges by label into interior and boundary edges
        interior_edges: Dict[int, List[str]] = collections.defaultdict(list)
        boundary_regions: Dict[int, List[int]] = collections.defaultdict(list)

        for (u, v, label), count in edge_counter.items():
            # Each undirected edge appears twice in adjacency list, so divide by 2
            multiplicity = count // 2

            if u == 0 and v == 0:
                # Self-loop on exterior region (should not occur in valid arrangements)
                continue
            elif u == 0 or v == 0:
                # Boundary edge: one endpoint is exterior region (id=0)
                interior_region = v if u == 0 else u
                boundary_regions[label].extend([interior_region] * multiplicity)
            else:
                # Interior edge: both endpoints are actual regions
                u_contains_circle = (self.masks[u] >> (label - 1)) & 1
                v_contains_circle = (self.masks[v] >> (label - 1)) & 1

                if u_contains_circle == v_contains_circle:
                    # Invalid: edge between regions with same circle membership
                    continue

                # Determine which region is inside vs outside the circle
                inside_region = u if u_contains_circle == 1 else v
                outside_region = v if u_contains_circle == 1 else u
                edge_pair = f"{inside_region}-{outside_region}"
                interior_edges[label].extend([edge_pair] * multiplicity)

        # Build the final code string
        chunks = [header]
        for label in range(1, self.N + 1):
            # Sort interior edges by region IDs for canonical ordering
            interior_list = sorted(
                interior_edges.get(label, []),
                key=lambda s: tuple(map(int, s.split("-"))),
            )
            interior_str = ";".join(interior_list)

            # Sort boundary regions for canonical ordering
            boundary_list = sorted(boundary_regions.get(label, []))
            boundary_str = ";".join(str(r) for r in boundary_list)

            chunks.append(f"E{label}_in:[{interior_str}]")
            chunks.append(f"E{label}_out:[{boundary_str}]")

        return "|".join(chunks)

    def label_invariant_mask_struct_code(self) -> str:
        """Minimize mask‑struct code across all label permutations (original logic)."""
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

            code = Dual(
                N=self.N, masks=new_masks, adj=new_adj
            )._mask_struct_code_fixed_labels()
            if best is None or code < best:
                best = code
        assert best is not None

        return best

    def with_boundary(
        self, boundary: Dict[RegionId, List[Tuple[RegionId, EdgeLabel]]]
    ) -> "Dual":
        """
        Return a new Dual with this same data but with an explicit boundary order.
        Does not modify self (class is frozen).
        """
        adj = {u: [(v, lbl) for (v, lbl) in arcs] for u, arcs in boundary.items()}
        return Dual(N=self.N, masks=self.masks, adj=adj, boundary=boundary)

    # A slightly stronger region‑relabel invariant (block backtracking), preserved
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
            new_colors: Dict[RegionId, int] = {}
            for r in verts:
                if sigs[r] not in mapping:
                    mapping[sigs[r]] = len(mapping) + 1
                new_colors[r] = mapping[sigs[r]]
            if new_colors == colors:
                break
            colors = new_colors

        # refine to a canonical string by ordering color blocks deterministically
        blocks: Dict[int, List[int]] = collections.defaultdict(list)
        for r, c in colors.items():
            blocks[c].append(r)
        orderings = [
            sorted(v, key=lambda r: (self.masks[r], len(self.adj[r]), r))
            for _, v in sorted(blocks.items())
        ]

        best: Optional[str] = None

        def bt(i: int, prefix: List[int]):
            nonlocal best
            if i == len(orderings):
                perm_order = prefix
                # remap region ids according to perm_order + 0 fixed
                rank: Dict[int, int] = {0: 0}
                for idx, r in enumerate(perm_order, start=1):
                    rank[r] = idx
                # rebuild code with fixed labels but canonical region order
                header = "V:" + ",".join(
                    _bitstr(self.masks[r], self.N) for r in perm_order
                )
                in_edges: Dict[int, List[Tuple[int, int]]] = collections.defaultdict(
                    list
                )
                out_edges: Dict[int, List[int]] = collections.defaultdict(list)
                counter: Dict[Tuple[int, int, int], int] = collections.Counter()
                for u, neis in self.adj.items():
                    for v, lbl in neis:
                        a, b = (u, v) if u <= v else (v, u)
                        counter[(a, b, lbl)] += 1
                for (a, b, lbl), cnt in counter.items():
                    mult = cnt // 2
                    if a == 0 and b == 0:
                        continue
                    if a == 0 or b == 0:
                        other = b if a == 0 else a
                        for _ in range(mult):
                            out_edges[lbl].append(rank[other])
                    else:
                        # Use ranked region IDs instead of raw masks
                        ra, rb = rank[a], rank[b]
                        lo, hi = (ra, rb) if ra <= rb else (rb, ra)
                        for _ in range(mult):
                            in_edges[lbl].append((lo, hi))
                chunks = [header]
                for lbl in range(1, self.N + 1):
                    ins = ";".join(f"{u}-{v}" for u, v in sorted(in_edges.get(lbl, [])))
                    outs = ";".join(str(r) for r in sorted(out_edges.get(lbl, [])))
                    chunks.append(f"E{lbl}_in:[{ins}]")
                    chunks.append(f"E{lbl}_out:[{outs}]")
                code = "|".join(chunks)
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
        seen.add(regions[0])
        while stack:
            u = stack.pop()
            for v, _ in self.adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        if len(seen) != len(self.masks):
            raise AssertionError("region graph is disconnected")

    def validate(self) -> None:
        self.validate_basic()
        self.validate_circle_cycles()
