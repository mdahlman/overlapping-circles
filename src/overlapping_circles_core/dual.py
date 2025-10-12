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


# --- helper: relabel circles by a permutation sigma on {1..N} ---
# sigma is a tuple like (2,3,1) meaning: new bit-1 = old bit-2, new bit-2 = old bit-3, new bit-3 = old bit-1
def _relabel_circles(self: "Dual", sigma: tuple[int, ...]) -> "Dual":
    assert len(sigma) == self.N and set(sigma) == set(range(1, self.N + 1))

    # Permute bits in every region mask
    def permute_bits(x: int) -> int:
        # build new mask y where bit i (1-based) comes from old bit sigma[i]-1
        y = 0
        for new_pos_1b, old_label in enumerate(sigma, start=1):
            if (x >> (old_label - 1)) & 1:
                y |= 1 << (new_pos_1b - 1)
        return y

    new_masks = {r: permute_bits(m) for r, m in self.masks.items()}

    # Relabel edge labels only; region ids stay the same here
    # old label L becomes new label L' = sigma^{-1}(L)
    inv = {sigma[i]: (i + 1) for i in range(len(sigma))}
    new_adj = {}
    for u, edges in self.adj.items():
        new_adj[u] = [(v, inv[lbl]) for (v, lbl) in edges]

    # Boundary rotation system: only labels change
    new_boundary = None
    if self.boundary is not None:
        nb = {}
        for rgn, cyc in self.boundary.items():
            nb[rgn] = [(nbr, inv[lbl]) for (nbr, lbl) in cyc]
        new_boundary = nb

    return self.__class__(N=self.N, masks=new_masks, adj=new_adj, boundary=new_boundary)


@dataclass(frozen=True)
class Edge:
    u: RegionId
    v: RegionId
    lbl: EdgeLabel


@dataclass(frozen=True)
class Dual:
    def current_code(self) -> str:
        """
        Simple string representation of the current arrangement, using the present region and adjacency data.
        No relabeling or canonicalization is performed.
        """
        # Exclude outside region (mask == 0)
        regions = [r for r, m in self.masks.items() if m != 0]
        # Use the order as stored in self.masks
        header = "V:" + ",".join(_bitstr(self.masks[r], self.N) for r in regions)

        in_edges: Dict[int, List[Tuple[int, int]]] = collections.defaultdict(list)
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
                    out_edges[lbl].append(other)
            else:
                lo, hi = (a, b) if a <= b else (b, a)
                for _ in range(mult):
                    in_edges[lbl].append((lo, hi))
        chunks = [header]
        for lbl in range(1, self.N + 1):
            ins = ";".join(f"{u}-{v}" for u, v in sorted(in_edges.get(lbl, [])))
            outs = ";".join(str(r) for r in sorted(out_edges.get(lbl, [])))
            chunks.append(f"E{lbl}_in:[{ins}]")
            chunks.append(f"E{lbl}_out:[{outs}]")
        code = "|".join(chunks)
        return code

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
        """
        Produce a deterministic canonical string for this Dual
        assuming circle labels are fixed.
        Region order is label-invariant (by weight, then bitstring),
        and the same region mapping is used for header and edges.
        """
        # 1) choose region order (exclude outside region 0)
        regions = [r for r, m in self.masks.items() if m != 0]

        # Pure lexicographic bitstring order (rightmost bit = C1, consistent with your convention)
        perm_order = sorted(regions, key=lambda rid: _bitstr(self.masks[rid], self.N))

        # 2) consistent mapping: RegionId → header position
        pos = {rid: i + 1 for i, rid in enumerate(perm_order)}

        # 3) header built from exactly that order
        header_masks = [self.masks[r] for r in perm_order]
        header = "V:" + ",".join(_bitstr(m, self.N) for m in header_masks)

        # 4) collect edges per label using the same pos mapping
        import collections

        ein = {lbl: collections.Counter() for lbl in range(1, self.N + 1)}
        eout = {lbl: collections.Counter() for lbl in range(1, self.N + 1)}

        for u, neis in self.adj.items():
            for v, lbl in neis:
                if v == 0:
                    # count outside edges only in the (u,0) direction (u>0)
                    if u != 0 and u in pos:
                        eout[lbl][pos[u]] += 1
                    # ignore (0,u)
                    continue
                if u == 0:
                    # ignore (0,v) to avoid double counting
                    continue

                # interior: count each undirected edge exactly once by original id order
                if u in pos and v in pos and u < v:
                    a, b = pos[u], pos[v]
                    if a > b:
                        a, b = b, a
                    ein[lbl][(a, b)] += 1

        # 5) NO normalization by //=2 — we counted each edge once on purpose

        # 6) emit string parts deterministically
        parts = [header]
        for lbl in range(1, self.N + 1):
            pairs = []
            for (a, b), mult in sorted(ein[lbl].items()):
                pairs.extend([f"{a}-{b}"] * mult)
            outs = []
            for a, mult in sorted(eout[lbl].items()):
                outs.extend([str(a)] * mult)
            parts.append(f"E{lbl}_in:[{';'.join(pairs)}]")
            parts.append(f"E{lbl}_out:[{';'.join(outs)}]")

        return "|".join(parts)

    def canonical_code(self) -> str:
        """
        Canonical string minimizing over all permutations of circle labels.
        Uses lexicographic minimization of the fixed-label canonical form.
        """
        best: Optional[str] = None
        for perm_tuple in itertools.permutations(range(1, self.N + 1)):
            perm = [0] + list(perm_tuple)

            # Build inverse permutation: inv[old_label] = new_label
            inv = [0] * (self.N + 1)
            for new_label in range(1, self.N + 1):
                old_label = perm[new_label]
                inv[old_label] = new_label

            # Relabel masks
            # Re-index regions so that identical masks get identical region IDs
            # (ensures region adjacencies remain valid after mask relabeling)
            new_masks = {}
            new_adj = {}

            for u, neis in self.adj.items():
                # compute new mask for this region
                mu = self.masks.get(u, 0)
                new_mask = 0
                for old_label in range(1, self.N + 1):
                    if (mu >> (old_label - 1)) & 1:
                        new_label = inv[old_label]
                        new_mask |= 1 << (new_label - 1)
                new_masks[u] = new_mask

                # now relabel its edges using same inverse mapping
                new_adj[u] = [(v, inv[lbl]) for v, lbl in neis]

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
