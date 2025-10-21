# Canonical Labeling of Circle Arrangements (Affine Plane)

Let an *arrangement of \(N\) circles in the affine plane* be given, with no tangencies and no triple intersection points.  
Each arrangement partitions the plane into connected regions bounded by arcs of circles.

We define its **canonical labeling** as a unique symbolic representation consisting of:

---

### 1. Intersection structure

A sequence of cyclic words

\[
W_1,\, W_2,\, \dots,\, W_N
\]

where \(W_i\) records, in cyclic order along circle \(i\), the signed labels of every other circle it crosses.

- The alphabet of \(W_i\) is \(\{\pm 1,\dots,\pm N\}\setminus\{\pm i\}\).
- The symbol `+j` marks the point where traversal of \(C_i\) enters the interior of \(C_j\);  
  the symbol `-j` marks where it leaves.

---

### 2. Containment structure

A set of ordered pairs

\[
C = \{\,a⊂b\,\}
\]

indicating that circle \(a\) lies entirely inside circle \(b\) without intersection.  
The containment relations form a directed acyclic graph on \(\{1,\dots,N\}\).

---

### 3. Canonical selection rule

Among all labelings equivalent under:

- permutation of circle labels \(1..N\),
- cyclic rotation of each \(W_i\),
- reversal of direction along any circle (reversing both order and sign of symbols),

choose exactly one labeling according to the following ordering:

1. **Primary key:** the lexicographically smallest concatenation  
   \(W_1\,|\,W_2\,|\,\dots\,|\,W_N\)  
   after each word has been individually normalized over its rotations and reversals.

2. **Secondary key (tie-breaker):** among labelings with identical \(W\)-bundle,  
   choose the one whose containment set \(C\) is lexicographically **largest**  
   when its pairs \(a⊂b\) are listed in descending lexicographic order.

The pair

\[
(W_1,\dots,W_N;\, C)
\]

obtained by this rule is the **canonical labeling** of the arrangement.

---

### Example (N = 2)

| Type | Canonical labeling |
|------|--------------------|
| Disjoint | `W1:[];W2:[];C:-` |
| Nested | `W1:[];W2:[];C:2⊂1` |
| Intersecting | `W1:[+2,-2];W2:[+1,-1];C:-` |

---

This definition uniquely identifies every combinatorial type of circle arrangement  
in the affine plane (up to isotopy preserving the outer region) and is unambiguous for all \(N \ge 1\).
