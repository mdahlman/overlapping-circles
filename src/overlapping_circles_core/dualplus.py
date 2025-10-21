"""
Core data model for circle arrangements in the affine plane.

Defines DualPlus — a complete, reversible representation that
encodes both the dual region-adjacency graph and the local
intersection order data required to reconstruct the primal graph.

PrimalGraph() can be derived directly from this structure.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
import itertools
import networkx as nx
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────────────
# Basic entities
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Region:
    """A connected region (dual vertex)."""

    id: str  # opaque, unique identifier (e.g. "r0")
    membership: Optional[Tuple[int, ...]] = None  # optional attribute, non-unique


@dataclass
class DualEdge:
    """A boundary segment between two regions along a circle."""

    id: str
    a: str  # Region.id of one side
    b: str  # Region.id of the other side
    circle: int  # circle whose boundary this arc belongs to


@dataclass
class DualPlus:
    """
    Complete combinatorial representation of a circle arrangement.

    Contents:
      • regions:     all connected regions (dual vertices)
      • dual_edges:  adjacency edges labeled by circle id
      • containment: directed pairs (inner, outer)
      • W:           cyclic intersection order for each circle

    This structure is canonical and sufficient to reconstruct
    both the primal and dual graphs.
    """

    regions: Dict[str, Region] = field(default_factory=dict)
    dual_edges: List[DualEdge] = field(default_factory=list)
    containment: Set[Tuple[int, int]] = field(default_factory=set)
    W: Dict[int, List[int]] = field(default_factory=dict)

    # ──────────────────────────────────────────────────────────────────────
    # Derived views
    # ──────────────────────────────────────────────────────────────────────
    def make_primal(self):
        """
        Construct and return a PrimalGraph representation.

        Stub only — to be implemented later.
        Will convert W and dual_edges into explicit intersection
        vertices and circle arcs.
        """
        raise NotImplementedError("PrimalGraph generation not yet implemented.")

    # ──────────────────────────────────────────────────────────────────────
    # Convenience and debugging helpers
    # ──────────────────────────────────────────────────────────────────────
    def summary(self) -> str:
        """Return a compact human-readable summary."""
        lines = []
        lines.append(
            f"DualPlus with {len(self.regions)} regions, "
            f"{len(self.dual_edges)} edges, "
            f"{len(self.containment)} containments"
        )
        lines.append(f"Regions (dual vertices): {self.list_regions()}")
        lines.append("cyclic words:")
        for cid, seq in sorted(self.W.items()):
            lines.append(f"  W{cid}: {seq}")
        if self.containment:
            lines.append(f"  Containment: {sorted(self.containment)}")
        return "\n".join(lines)

    def generate_label(self) -> str:
        """
        Produce a straightforward label string for this DualPlus instance.

        Format follows the description in definitions.md:
          W1:[+j,-k,...];W2:[...];...;C:2⊂1,3⊂1

        Notes:
          • This is not canonicalized. It reflects the current stored order in self.W.
          • Circles are emitted in ascending order of their ids as found in self.W.
          • Each entry in a W_i list is an int: positive → "+j", negative → "-j".
          • Containment pairs (a, b) are sorted ascending and joined by commas; empty → "C:-".
        """

        # Format the W block(s)
        w_parts: List[str] = []
        for cid in sorted(self.W.keys()):
            seq = self.W.get(cid, [])
            tokens = []
            for v in seq:
                sign = "+" if v >= 0 else "-"
                tokens.append(f"{sign}{abs(v)}")
            w_parts.append(f"W{cid}:[{','.join(tokens)}]")

        # Format containment C
        if not self.containment:
            c_part = "C:-"
        else:
            pairs = sorted(self.containment)
            c_part = "C:" + ",".join(f"{a}⊂{b}" for a, b in pairs)

        return ";".join(w_parts + [c_part])

    def list_regions(self) -> List[str]:
        """
        List bitmask strings for all regions.

        Rules:
          • Bit width N = number of circles = len(self.W) (or inferred from membership if W empty).
          • If Region.membership is a tuple of length N of 0/1 values, use as-is.
          • Otherwise, treat Region.membership as an iterable of circle ids contained in the region
            and build a bitmask with those ids set.
          • Sort by: (1) descending popcount (more nested first), then (2) lex order ascending.

        Returns a list of strings like ["001", "011", ...].
        The first (leftmost) bit corresponds to circle 1.
        Regions without membership are skipped. (But perhaps this should throw an error instead.)
        """

        # Determine number of circles N
        if self.W:
            N = max(self.W.keys())  # circle ids are 1-based
        else:
            # Fallback: infer from the largest membership reference
            max_id = 0
            for r in self.regions.values():
                m = r.membership
                if not m:
                    continue
                # If tuple of 0/1 bits
                if isinstance(m, tuple) and all(isinstance(x, int) for x in m):
                    if all(x in (0, 1) for x in m):
                        max_id = max(max_id, len(m))
                        continue
                # Otherwise treat as set of circle ids
                try:
                    max_id = max(max_id, max(int(x) for x in m))
                except Exception:
                    pass
            N = max_id

        def to_mask_str(membership: Tuple[int, ...]) -> str:
            # Case A: tuple of 0/1 bits of length N
            if len(membership) == N and all(x in (0, 1) for x in membership):
                return "".join(str(x) for x in membership)
            # Case B: interpret as circle id list
            bits = [0] * N
            for x in membership:
                i = int(x)
                if 1 <= i <= N:
                    bits[i - 1] = 1
            return "".join(str(b) for b in bits)

        masks: List[str] = []
        for r in self.regions.values():
            if r.membership is None:
                continue
            try:
                mask = to_mask_str(tuple(r.membership))
            except Exception:
                # If membership is not iterable, skip
                continue
            masks.append(mask)

        # Sort: descending popcount, then lexicographic ascending
        masks.sort(key=lambda s: (-s.count("1"), s))
        return masks

    def generate_canonical_label(self) -> str:
        """
        Compute the canonical label string as defined in definitions.md.

        Canonicalization considers:
          • Permutation of circle labels 1..N
          • Cyclic rotation of each W_i
          • Reversal of direction along any circle (reverse order and flip signs)

        Selection rule:
          1) Minimize the lexicographic tuple (W1_norm, W2_norm, ..., WN_norm),
             where each W_i is independently normalized over its rotations and
             reversals for the given permutation.
          2) If tied, choose lexicographically largest containment set C
             when pairs (a,b) are sorted in descending lexicographic order.

        Returns a string in the same format as generate_label().
        Note: This is a combinatorial canonicalization and does not depend
        on regions or dual_edges.
        """

        if not self.W:
            # No circles → just containment on empty set
            return "C:-"

        # Establish the list of original circle ids and a mapping to a compact 1..N domain.
        orig_ids = sorted(self.W.keys())
        N = len(orig_ids)

        # Helper: convert an integer token to its signed string form.
        def token_str(v: int) -> str:
            return ("+" if v >= 0 else "-") + str(abs(v))

        # Helper: normalize a cyclic signed word under rotations and reversed+negated rotations.
        def normalize_word(seq: List[int]) -> Tuple[str, ...]:
            if not seq:
                return tuple()

            # Generate all rotations of a sequence of strings
            def rotations(xs: List[str]) -> List[Tuple[str, ...]]:
                return [tuple(xs[i:] + xs[:i]) for i in range(len(xs))]

            # Original tokens
            toks = [token_str(v) for v in seq]
            candidates = rotations(toks)

            # Reversed with sign flipped
            toks_rev = [token_str(-v) for v in reversed(seq)]
            candidates += rotations(toks_rev)

            return min(candidates)

        # Apply a permutation mapping old ids → new labels in 1..N.
        # perm is a tuple of new labels assigned to orig_ids in order.
        # Example: orig_ids=[3,5,7], perm=(2,1,3) means 3→2, 5→1, 7→3
        def apply_perm_and_normalize(perm: Tuple[int, ...]):
            id_to_new = {oid: new for oid, new in zip(orig_ids, perm)}

            # Build normalized W words for new labels 1..N
            new_words: List[Tuple[str, ...]] = [tuple() for _ in range(N)]

            for oid in orig_ids:
                new_label = id_to_new[oid]
                # Relabel the sequence values by perm, preserving signs
                seq_old = self.W.get(oid, [])
                seq_new = [((1 if v >= 0 else -1) * id_to_new[abs(v)]) for v in seq_old]
                new_words[new_label - 1] = normalize_word(seq_new)

            # Compute containment under permutation
            cont_new: Set[Tuple[int, int]] = set()
            for a, b in self.containment:
                cont_new.add((id_to_new[a], id_to_new[b]))

            # For primary key: the tuple of word tuples
            primary_key = tuple(new_words)

            # For tie-breaker: pairs sorted in descending lexicographic order (numeric)
            secondary_key = tuple(sorted(cont_new, reverse=True))

            return primary_key, secondary_key, new_words, cont_new

        best_primary = None
        best_secondary = None
        best_words = None
        best_cont = None

        for perm in itertools.permutations(range(1, N + 1)):
            primary, secondary, words, cont = apply_perm_and_normalize(perm)
            if (
                (best_primary is None)
                or (primary < best_primary)
                or (
                    primary == best_primary
                    and (best_secondary is None or secondary > best_secondary)
                )
            ):
                best_primary = primary
                best_secondary = secondary
                best_words = words
                best_cont = cont

        # Format output using the best permutation's normalized words
        assert best_words is not None and best_cont is not None

        w_parts: List[str] = []
        for cid in range(1, N + 1):
            toks = list(best_words[cid - 1])
            w_parts.append(f"W{cid}:[{','.join(toks)}]")

        if not best_cont:
            c_part = "C:-"
        else:
            pairs = sorted(best_cont)
            c_part = "C:" + ",".join(f"{a}⊂{b}" for a, b in pairs)

        return ";".join(w_parts + [c_part])


# ──────────────────────────────────────────────────────────────────────────────
# Derived placeholder class (for later implementation)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class PrimalGraph:
    """Placeholder for geometric / intersection view of an arrangement."""

    vertices: Dict[str, Tuple[int, int]] = field(
        default_factory=dict
    )  # id → (circle_i, circle_j)
    edges: List[Tuple[int, str, str]] = field(
        default_factory=list
    )  # (circle, start_vid, end_vid)

    def summary(self) -> str:
        return (
            f"PrimalGraph with {len(self.vertices)} vertices "
            f"and {len(self.edges)} edges"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Visualization utility
# ──────────────────────────────────────────────────────────────────────────────
def draw_dualplus(dp: DualPlus, title: str | None = None) -> None:
    """
    Visualize a DualPlus instance as an undirected labeled graph.

    • Regions → nodes (labeled by region id and bitmask membership)
    • DualEdges → edges (labeled by the circle that separates them)

    Args:
        dp: DualPlus instance to visualize.
        title: Optional title for the plot.
    """
    G = nx.Graph()

    # Build combined labels like "r1:011"
    for rid, region in dp.regions.items():
        mask = getattr(region, "membership", None)
        mask_str = "".join(str(b) for b in mask) if mask is not None else "?"
        label = f"{region.id}:{mask_str}"
        G.add_node(label)

    # Add edges labeled by circle IDs
    for e in dp.dual_edges:
        a_label = f"{dp.regions[e.a].id}:{''.join(str(b) for b in dp.regions[e.a].membership)}"  # type: ignore
        b_label = f"{dp.regions[e.b].id}:{''.join(str(b) for b in dp.regions[e.b].membership)}"  # type: ignore
        G.add_edge(a_label, b_label, label=f"C{e.circle}")

    # Layout and rendering
    pos = nx.spring_layout(G, seed=2)
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="lightgray",
        node_size=1200,
        font_size=10,
        font_weight="bold",
        ax=ax,
    )
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=nx.get_edge_attributes(G, "label"),
        font_color="blue",
        ax=ax,
    )
    if title:
        ax.set_title(title, pad=12)
    ax.axis("off")
    plt.show()
