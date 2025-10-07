from __future__ import annotations
from dataclasses import dataclass
from itertools import permutations, product
from typing import Dict, FrozenSet, List, Literal, Set, Tuple

Label = Literal["A", "B", "C"]
LABELS: Tuple[Label, Label, Label] = ("A", "B", "C")


@dataclass(frozen=True)
class ArrangementN3:
    """
    Purely combinatorial N=3 arrangement (no geometry).
    - poset: A 'contains' B iff (A,B) in poset (strict; irreflexive; transitive).
    - edges: unordered pairs that intersect (2-point crossing), only for incomparable pairs.
    - signature: canonical string "CCC|EEE|T" where
        - CCC are trits for (AB,AC,BC): 0=incomparable, 1=first>second, 2=second>first
        - EEE are edge bits for (AB,AC,BC)
        - T is triple-intersection bit for ABC (0 empty, 1 non-empty)
    - readable: human-friendly label (containment forest + edges + triple flag).
    """

    poset: FrozenSet[Tuple[Label, Label]]
    edges: FrozenSet[FrozenSet[Label]]
    signature: str
    readable: str


# ----------------------------
# Poset utilities
# ----------------------------


def _is_dag(rels: Set[Tuple[Label, Label]]) -> bool:
    if any(a == b for a, b in rels):
        return False
    if any((b, a) in rels for a, b in rels):
        return False
    closure = set(rels)
    changed = True
    while changed:
        changed = False
        to_add = {
            (a, d) for (a, b) in closure for (c, d) in closure if b == c and a != d
        }
        new = to_add - closure
        if new:
            closure |= new
            changed = True
            if any(a == b for (a, b) in closure):
                return False
    return True


def _transitive_closure(
    rels: Set[Tuple[Label, Label]],
) -> FrozenSet[Tuple[Label, Label]]:
    closure: Set[Tuple[Label, Label]] = set(rels)
    changed = True
    while changed:
        changed = False
        to_add = {
            (a, d) for (a, b) in closure for (c, d) in closure if b == c and a != d
        }
        new = to_add - closure
        if new:
            closure |= new
            changed = True
    return frozenset(closure)


def enumerate_posets_3() -> List[FrozenSet[Tuple[Label, Label]]]:
    pairs = [(a, b) for a in LABELS for b in LABELS if a != b]
    unique: Set[FrozenSet[Tuple[Label, Label]]] = set()
    for mask in range(1 << len(pairs)):
        rels: Set[Tuple[Label, Label]] = {
            pairs[i] for i in range(len(pairs)) if (mask >> i) & 1
        }
        if not _is_dag(rels):
            continue
        unique.add(_transitive_closure(rels))
    return list(unique)


# ----------------------------
# Basic relations
# ----------------------------


def _contains(poset: FrozenSet[Tuple[Label, Label]], a: Label, b: Label) -> bool:
    return (a, b) in poset


def _comparable(poset: FrozenSet[Tuple[Label, Label]], a: Label, b: Label) -> bool:
    return _contains(poset, a, b) or _contains(poset, b, a)


def incomparable_pairs(poset: FrozenSet[Tuple[Label, Label]]) -> List[FrozenSet[Label]]:
    return [
        frozenset({u, v})
        for u, v in (("A", "B"), ("A", "C"), ("B", "C"))
        if not _comparable(poset, u, v)
    ]


def _has_edge(edges: FrozenSet[FrozenSet[Label]], u: Label, v: Label) -> bool:
    return frozenset({u, v}) in edges


# ----------------------------
# Consistency (disks)
# ----------------------------


def _consistent(
    poset: FrozenSet[Tuple[Label, Label]], edges: FrozenSet[FrozenSet[Label]]
) -> bool:
    # No edge between comparable pairs
    for e in edges:
        u, v = tuple(e)  # type: ignore[misc]
        if _comparable(poset, u, v):
            return False
    # If A ⊃ B and C incomparable with A:
    #  - if B∩C, then A∩C must also hold (since B ⊂ A)
    #  - if A disjoint C, then B must be disjoint C
    for a, b in poset:
        for c in LABELS:
            if c in (a, b):
                continue
            if _comparable(poset, a, c):
                continue
            bc = _has_edge(edges, b, c)
            ac = _has_edge(edges, a, c)
            if bc and not ac:
                return False
            if (not ac) and bc:
                return False
    return True


# ----------------------------
# Triple-intersection inference (N=3)
# ----------------------------


def _infer_triple(
    poset: FrozenSet[Tuple[Label, Label]], edges: FrozenSet[FrozenSet[Label]]
) -> int | None:
    """
    Return 1 if A∩B∩C is forced non-empty, 0 if forced empty, or None if ambiguous (both realizable).
    Ambiguity occurs only in the antichain with edges AB=AC=BC=1.
    """
    AB = _has_edge(edges, "A", "B")
    AC = _has_edge(edges, "A", "C")
    BC = _has_edge(edges, "B", "C")

    # Chain cases: any chain implies triple non-empty (the smallest disk survives)
    chains = [
        (("A", "B"), ("B", "C")),
        (("B", "A"), ("A", "C")),
        (("A", "C"), ("C", "B")),
        (("C", "A"), ("A", "B")),
        (("B", "C"), ("C", "A")),
        (("C", "B"), ("B", "A")),
    ]
    for r1, r2 in chains:
        if r1 in poset and r2 in poset:
            return 1

    # Fork (one contains the other two): triple exists iff the two children intersect
    forks = [
        ("A", "B", "C"),
        ("B", "A", "C"),
        ("C", "A", "B"),
    ]  # parent, child1, child2
    for p, u, v in forks:
        if _contains(poset, p, u) and _contains(poset, p, v):
            return 1 if _has_edge(edges, u, v) else 0

    # Single containment (one pair, third incomparable): triple exists iff (child ∩ third)
    singles = [
        ("A", "B", "C"),
        ("A", "C", "B"),
        ("B", "C", "A"),
        ("B", "A", "C"),
        ("C", "A", "B"),
        ("C", "B", "A"),
    ]
    for p, ch, t in singles:
        if (
            _contains(poset, p, ch)
            and (not _comparable(poset, p, t))
            and (not _comparable(poset, ch, t))
        ):
            return 1 if _has_edge(edges, ch, t) else 0

    # Antichain:
    if len(poset) == 0:
        if AB and AC and BC:
            return None  # ambiguous: both realizable
        # If any pair is missing, triple cannot exist
        return 0

    # Default (should not occur)
    return 0


# ----------------------------
# Canonical signature with trits + triple flag
# ----------------------------


def _pair_trit(poset: FrozenSet[Tuple[Label, Label]], x: Label, y: Label) -> str:
    if _contains(poset, x, y):
        return "1"
    elif _contains(poset, y, x):
        return "2"
    else:
        return "0"


def _encode_signature(
    poset: FrozenSet[Tuple[Label, Label]],
    edges: FrozenSet[FrozenSet[Label]],
    T: int,
    order: Tuple[Label, Label, Label],
) -> str:
    a, b, c = order
    c_bits = [
        _pair_trit(poset, a, b),
        _pair_trit(poset, a, c),
        _pair_trit(poset, b, c),
    ]

    def E(u: Label, v: Label) -> str:
        return "1" if frozenset({u, v}) in edges else "0"

    e_bits = [E(a, b), E(a, c), E(b, c)]
    return "".join(c_bits) + "|" + "".join(e_bits) + "|" + str(T)


def _relabel_to_order(
    poset: FrozenSet[Tuple[Label, Label]],
    edges: FrozenSet[FrozenSet[Label]],
    order: Tuple[Label, Label, Label],
) -> Tuple[FrozenSet[Tuple[Label, Label]], FrozenSet[FrozenSet[Label]]]:
    a, b, c = order
    mapping: Dict[Label, Label] = {a: "A", b: "B", c: "C"}  # type: ignore[assignment]
    poset2 = frozenset((mapping[p], mapping[ch]) for (p, ch) in poset)  # type: ignore[misc]
    edges2 = frozenset(
        frozenset({mapping[u], mapping[v]}) for (u, v) in (tuple(e) for e in edges)  # type: ignore[misc]
    )
    return poset2, edges2


def _canonicalize(
    poset: FrozenSet[Tuple[Label, Label]],
    edges: FrozenSet[FrozenSet[Label]],
    T: int,
) -> Tuple[FrozenSet[Tuple[Label, Label]], FrozenSet[FrozenSet[Label]], int, str]:
    best_sig: str | None = None
    best_order: Tuple[Label, Label, Label] | None = None
    for order in permutations(LABELS):
        sig = _encode_signature(poset, edges, T, order)
        if best_sig is None or sig < best_sig:
            best_sig = sig
            best_order = order
    assert best_sig is not None and best_order is not None
    poset_c, edges_c = _relabel_to_order(poset, edges, best_order)
    # Recompute T on canonical structure (T stays same for N=3 but do it for consistency)
    T_c = _infer_triple(poset_c, edges_c)
    if T_c is None:
        T_c = T
    sig_c = _encode_signature(poset_c, edges_c, T_c, ("A", "B", "C"))
    return poset_c, edges_c, T_c, sig_c


# ----------------------------
# Readable label
# ----------------------------


def _readable_label(
    poset: FrozenSet[Tuple[Label, Label]],
    edges: FrozenSet[FrozenSet[Label]],
    T: int,
) -> str:
    children: Dict[Label, List[Label]] = {L: [] for L in LABELS}
    has_parent: Dict[Label, bool] = {L: False for L in LABELS}
    for p, c in poset:
        children[p].append(c)
        has_parent[c] = True
    roots = [L for L in LABELS if not has_parent[L]]

    def format_node(u: Label) -> str:
        if not children[u]:
            return u
        subs = sorted(format_node(v) for v in children[u])
        return f"{u}(" + "; ".join(subs) + ")"

    forest_str = "; ".join(sorted(format_node(r) for r in roots))
    edge_str = ", ".join(sorted("".join(sorted(e)) for e in edges)) if edges else "∅"
    triple_str = "✓" if T == 1 else "✗"
    return f"{forest_str} | edges: {edge_str} | triple: {triple_str}"


# ----------------------------
# Main generator
# ----------------------------


def generate_arrangements_n3() -> List[ArrangementN3]:
    """
    Enumerate all labeled, combinatorially valid N=3 arrangements (disks), with nesting allowed.
    Intersections only on incomparable pairs. Canonicalize by signature "CCC|EEE|T"
    and deduplicate. In the unique ambiguous case (antichain + all three edges),
    emit both T=0 and T=1 variants.
    """
    seen: Dict[str, ArrangementN3] = {}

    for poset in enumerate_posets_3():
        inc = incomparable_pairs(poset)
        for bits in product([0, 1], repeat=len(inc)):
            edges: FrozenSet[FrozenSet[Label]] = frozenset(
                {p for p, b in zip(inc, bits) if b == 1}
            )
            if not _consistent(poset, edges):
                continue

            T = _infer_triple(poset, edges)
            if T is None:
                # Ambiguous (antichain + triangle): emit both variants
                T_candidates = [0, 1]
            else:
                T_candidates = [T]

            for T_val in T_candidates:
                poset_c, edges_c, T_c, sig = _canonicalize(poset, edges, T_val)
                if sig not in seen:
                    readable = _readable_label(poset_c, edges_c, T_c)
                    seen[sig] = ArrangementN3(
                        poset=poset_c, edges=edges_c, signature=sig, readable=readable
                    )

    return sorted(seen.values(), key=lambda a: (a.signature, a.readable))


# ----------------------------
# CLI preview
# ----------------------------

if __name__ == "__main__":
    arrs = generate_arrangements_n3()
    for a in arrs:
        print(a.signature, " :: ", a.readable)
    print(f"Total N=3 arrangements (unique, labeled): {len(arrs)}")
