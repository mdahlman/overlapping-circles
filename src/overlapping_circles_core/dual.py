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
        regions = sorted(self.masks)
        header = "V:" + ",".join(_bitstr(self.masks[r], self.N) for r in perm_order)

        # Build labeled edge multisets in mask space
        in_edges: Dict[int, List[Tuple[int, int]]] = collections.defaultdict(list)
        out_edges: Dict[int, List[int]] = collections.defaultdict(list)
        rmask = self.masks
        counter: Dict[Tuple[int, int, int], int] = collections.Counter()
        for u, neis in self.adj.items():
            for v, lbl in neis:
                a, b = (u, v) if u <= v else (v, u)
                counter[(a, b, lbl)] += 1

        # Build region-ID based edge lists
        # counter holds multiplicities of undirected edges
        in_edges: Dict[int, List[str]] = collections.defaultdict(list)
        out_edges: Dict[int, List[int]] = collections.defaultdict(list)
        for (a, b, lbl), cnt in counter.items():
            # each undirected edge must appear twice in adj
            mult = cnt // 2
            if a == 0 and b == 0:
                continue
            if a == 0 or b == 0:
                other = b if a == 0 else a
                out_edges[lbl].extend([other] * mult)
            else:
                a_in = (self.masks[a] >> (lbl - 1)) & 1
                b_in = (self.masks[b] >> (lbl - 1)) & 1
                if a_in == b_in:
                    continue  # invalid edge for label; skip
                inside, outside = (a, b) if a_in == 1 else (b, a)
                in_edges[lbl].extend([f"{inside}-{outside}"] * mult)
        chunks = [header]
        for lbl in range(1, self.N + 1):
            ins = ";".join(
                sorted(
                    in_edges.get(lbl, []), key=lambda s: tuple(map(int, s.split("-")))
                )
            )
            outs = ";".join(str(x) for x in sorted(out_edges.get(lbl, [])))
            chunks.append(f"E{lbl}_in:[{ins}]")
            chunks.append(f"E{lbl}_out:[{outs}]")
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

        # Legacy-format postprocessing: convert mask bitstrings in edges to region IDs
        # Expect header like "V:001,010,011,..."
        if best and best.startswith("V:"):
            head, *rest = best.split("|")
            bitlist = head.split(":", 1)[1].split(",") if ":" in head else []
            index_map = {
                b: str(i + 1) for i, b in enumerate(bitlist)
            }  # 1-based region IDs

            def repl_pair(m):
                a, b = m.group(1), m.group(2)
                return f"{index_map.get(a, a)}-{index_map.get(b, b)}"

            def repl_single(m):
                a = m.group(1)
                return index_map.get(a, a)

            body = "|".join(rest)
            # Replace pairs like 010-011 inside brackets
            body = re.sub(r"([01]{%d})-([01]{%d})" % (self.N, self.N), repl_pair, body)
            # Replace singles like [010;110]
            body = re.sub(r"(?<=\[)([01]{%d})(?=[;\]])" % self.N, repl_single, body)
            best = head + "|" + body if rest else head
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
                rmask = self.masks
                for (a, b, lbl), cnt in counter.items():
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
                    outs = ";".join(
                        f"{m:0{self.N}b}" for m in sorted(out_edges.get(lbl, []))
                    )
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

        # Legacy-format postprocessing: convert mask bitstrings in edges to region IDs
        # Expect header like "V:001,010,011,..."
        if best and best.startswith("V:"):
            head, *rest = best.split("|")
            bitlist = head.split(":", 1)[1].split(",") if ":" in head else []
            index_map = {
                b: str(i + 1) for i, b in enumerate(bitlist)
            }  # 1-based region IDs

            def repl_pair(m):
                a, b = m.group(1), m.group(2)
                return f"{index_map.get(a, a)}-{index_map.get(b, b)}"

            def repl_single(m):
                a = m.group(1)
                return index_map.get(a, a)

            body = "|".join(rest)
            # Replace pairs like 010-011 inside brackets
            body = re.sub(r"([01]{%d})-([01]{%d})" % (self.N, self.N), repl_pair, body)
            # Replace singles like [010;110]
            body = re.sub(r"(?<=\[)([01]{%d})(?=[;\]])" % self.N, repl_single, body)
            best = head + "|" + body if rest else head
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

        # Legacy-format postprocessing: convert mask bitstrings in edges to region IDs
        # Expect header like "V:001,010,011,..."
        if best and best.startswith("V:"):
            head, *rest = best.split("|")
            bitlist = head.split(":", 1)[1].split(",") if ":" in head else []
            index_map = {
                b: str(i + 1) for i, b in enumerate(bitlist)
            }  # 1-based region IDs

            def repl_pair(m):
                a, b = m.group(1), m.group(2)
                return f"{index_map.get(a, a)}-{index_map.get(b, b)}"

            def repl_single(m):
                a = m.group(1)
                return index_map.get(a, a)

            body = "|".join(rest)
            # Replace pairs like 010-011 inside brackets
            body = re.sub(r"([01]{%d})-([01]{%d})" % (self.N, self.N), repl_pair, body)
            # Replace singles like [010;110]
            body = re.sub(r"(?<=\[)([01]{%d})(?=[;\]])" % self.N, repl_single, body)
            best = head + "|" + body if rest else head
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
