from __future__ import annotations
from typing import Dict, List, Tuple, Iterable
import itertools
import collections

from .dual import Dual


def _label_fingerprint(d: Dual, lbl: int) -> Tuple:
    """Cheap per-label fingerprint to limit permutations.
    Includes degree histogram of masks touched by the label.
    """
    deg = collections.Counter()
    for u, v, l in d.edges():
        if l != lbl:
            continue
        deg[d.masks[u]] += 1
        deg[d.masks[v]] += 1
    return tuple(sorted(deg.items()))


def _label_blocks(d: Dual) -> List[List[int]]:
    fps = {lbl: _label_fingerprint(d, lbl) for lbl in range(1, d.N + 1)}
    groups: Dict[Tuple, List[int]] = collections.defaultdict(list)
    for lbl, fp in fps.items():
        groups[fp].append(lbl)
    return [sorted(v) for v in groups.values()]


def canonical_code_label_invariant(
    d: Dual, *, worst_case_full_perms: bool = False
) -> str:
    """Return the minimal fixed-label structure code over label permutations.

    Uses per-label fingerprints to restrict permutations to blocks of indistinguishable labels.
    Falls back to full S_N if worst_case_full_perms=True.
    """
    code_best = None
    if worst_case_full_perms:
        blocks = [list(range(1, d.N + 1))]
    else:
        blocks = _label_blocks(d)
    # build product of permutations per block
    perms_iter: Iterable[Dict[int, int]] = ({},)
    for block in blocks:
        new_iter = []
        for p in perms_iter:
            for perm in itertools.permutations(block):
                mp = dict(p)
                for src, dst in zip(block, perm):
                    mp[src] = dst
                new_iter.append(mp)
        perms_iter = new_iter

    def relabel_code(mp: Dict[int, int]) -> str:
        # relabel labels in the structure code by substituting label ids
        # We rebuild a relabeled code by remapping Lx sections.
        fixed = d.to_mask_struct_code_fixed_labels()
        # The fixed code has predictable segments: N=..|M=..|L1=[...]|L2=[...]|...
        head, *tails = fixed.split("|")
        mseg = tails[0]
        lsegs = tails[1:]
        # parse Lk segments
        pairs = []
        for seg in lsegs:
            assert seg.startswith("L") and "]" in seg
            k = int(seg[1 : seg.index("=")])
            body = seg[seg.index("[") :]
            pairs.append((mp[k], f"L{mp[k]}{body}"))
        pairs.sort(key=lambda x: x[0])
        new_tail = [mseg] + [s for _, s in pairs]
        return "|".join([head] + new_tail)

    for mp in perms_iter:
        code = relabel_code(mp)
        if code_best is None or code < code_best:
            code_best = code
    assert code_best is not None
    return code_best
