from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import math

from sympy import Point, Circle as SymCircle, pi, atan2, N
from sympy.geometry.util import intersection as sym_intersection


@dataclass(frozen=True)
class Circle:
    cx: float
    cy: float
    r: float

    def sym(self) -> SymCircle:
        result = SymCircle(Point(self.cx, self.cy), self.r)
        if not isinstance(result, SymCircle):
            raise ValueError(f"Expected Circle, got {type(result)}")
        return result

    def contains_strict(self, p: Point) -> bool:
        # Strict inside test; avoids boundary issues by nudging sample points later.
        center = Point(self.cx, self.cy)
        return float(p.distance(center)) < self.r


@dataclass(frozen=True)
class Vertex:
    pt: Point


@dataclass
class Arc:
    circle_idx: int  # which circle this arc lies on
    start_angle: float  # radians, numeric (for ordering)
    end_angle: float  # radians, numeric (CCW wrap-aware)
    v_start: Vertex  # start intersection vertex
    v_end: Vertex  # end intersection vertex

    # Helper: mid-angle for sampling
    def mid_angle(self) -> float:
        a, b = self.start_angle, self.end_angle
        # Normalize to [0, 2pi)
        twopi = 2 * math.pi
        a = a % twopi
        b = b % twopi
        if b < a:
            b += twopi
        return (a + b) / 2


@dataclass
class HalfEdge:
    arc: Arc
    origin: Vertex
    twin: Optional[HalfEdge] = None
    next: Optional[HalfEdge] = None
    prev: Optional[HalfEdge] = None
    face: Optional[Face] = None


@dataclass
class Face:
    boundary: List[HalfEdge]  # one cycle (handle holes later)
    bitmask: str  # which circles cover this face, e.g. "101" for circles 0 and 2


def _angle(cx: float, cy: float, p: Point) -> float:
    """Return angle for ordering around the circle."""
    return math.atan2(float(p.y - cy), float(p.x - cx))  # type: ignore


def _ordered_angles_on_circle(c: Circle, pts: List[Point]) -> List[Tuple[float, Point]]:
    """Return list of (angle, point) tuples ordered CCW around the circle."""
    angles_pts = [(_angle(c.cx, c.cy, p), p) for p in pts]
    angles_pts.sort(key=lambda ap: ap[0])
    return angles_pts


def _between_ccw(a: float, b: float, x: float) -> bool:
    """Is angle x on the CCW arc from a to b (assuming a->b CCW, with wrap)?"""
    # Normalize to [0, 2pi)
    twopi = 2 * math.pi
    a = a % twopi
    b = b % twopi
    x = x % twopi
    if b < a:
        b += twopi
    if x < a:
        x += twopi
    return a < x < b


def _param_point_on_circle(c: Circle, theta: float) -> Point:
    """Return point on circle at angle theta (radians)."""
    return Point(c.cx + c.r * math.cos(theta), c.cy + c.r * math.sin(theta))


def _small_inward_offset(p: Point, c: Circle, epsilon=1e-6) -> Point:
    """Return a point slightly inside the circle from point p."""
    # Vector from center to p
    vx = float(p.x - c.cx)  # type: ignore
    vy = float(p.y - c.cy)  # type: ignore
    norm = math.hypot(vx, vy) or 1.0
    # Normalize and scale by epsilon
    vx = (vx / norm) * epsilon
    vy = (vy / norm) * epsilon
    return Point(float(p.x) - vx, float(p.y) - vy)  # type: ignore


def _classify_point_bitmask(p: Point, circles: List[Circle]) -> str:
    """Return bitmask string indicating which circles contain point p."""
    bits = []
    for c in circles:
        bits.append("1" if c.contains_strict(p) else "0")
    return "".join(bits)


def build_arrangement(
    circles: List[Circle],
) -> Tuple[List[Vertex], List[Arc], List[Face]]:
    """
    Build a circular-arc arrangement for the given circles.
    Returns vertices, arcs, faces with exact labeling.
    Assumes general position (no tangency, no triple intersection on a single point).
    """
    n = len(circles)
    sym = [c.sym() for c in circles]

    # Step 1: Find all intersection points
    #         Collect per-circle intersection points
    circle_int_pts: List[List[Point]] = [[] for _ in range(n)]
    all_vertices: Dict[Tuple[float, float], Vertex] = (
        {}
    )  # key by numeric (x,y) for uniqueness

    for idx in range(n):
        for j in range(idx + 1, n):
            inters = sym_intersection(sym[idx], sym[j])  # 0, 1 (tangency), or 2 points
            for p in inters:
                if not isinstance(p, Point):
                    continue  # skip non-point intersections
                # Get numeric coordinates
                # Deduplicate vertices by coordinates (good enough with sympy's exactness)
                px, py = float(N(getattr(p, "x"))), float(N(getattr(p, "y")))
                key = (px, py)
                v = all_vertices.get(key)
                if v is None:
                    v = Vertex(pt=p)
                    all_vertices[key] = v
                circle_int_pts[idx].append(p)
                circle_int_pts[j].append(p)
    vertices = list(all_vertices.values())

    # Step 2: For each circle, order its intersection points CCW and create arcs
    arcs: List[Arc] = []
    for idx, c in enumerate(circles):
        pts = circle_int_pts[idx]
        if len(pts) < 2:
            continue  # No arcs if fewer than 2 intersection points
        # Order points CCW around circle
        ang_points = _ordered_angles_on_circle(c, pts)
        k = len(ang_points)
        for t in range(k):
            a1, p1 = ang_points[t]
            a2, p2 = ang_points[(t + 1) % k]  # wrap around
            # Lookup vertices (by numeric key)
            av = all_vertices[(float(N(getattr(p1, "x"))), float(N(getattr(p1, "y"))))]
            bv = all_vertices[(float(N(getattr(p2, "x"))), float(N(getattr(p2, "y"))))]
            arc = Arc(
                circle_idx=idx,
                start_angle=a1,
                end_angle=a2,
                v_start=av,
                v_end=bv,
            )
            arcs.append(arc)

    # Step 3: Create half-edges and stitch twins (two directions per arc)
    half_edges: List[HalfEdge] = []
    # Map from (circle_idx, start_vertex, end_vertex) to half-edge for twin lookup
    half_edge_map: Dict[
        Tuple[int, Tuple[float, float], Tuple[float, float]], HalfEdge
    ] = {}

    def key_of(v: Vertex) -> Tuple[float, float]:
        x = float(N(getattr(v.pt, "x")))
        y = float(N(getattr(v.pt, "y")))
        return (x, y)  # use numeric key for lookup

    for arc in arcs:
        he_fwd = HalfEdge(arc=arc, origin=arc.v_start)
        he_rev_arc = Arc(
            circle_idx=arc.circle_idx,
            start_angle=arc.end_angle,
            end_angle=arc.start_angle,
            v_start=arc.v_end,
            v_end=arc.v_start,
        )
        he_rev = HalfEdge(arc=he_rev_arc, origin=arc.v_end)
        he_fwd.twin = he_rev
        he_rev.twin = he_fwd
        half_edges.extend([he_fwd, he_rev])

        half_edge_map[(arc.circle_idx, key_of(arc.v_start), key_of(arc.v_end))] = he_fwd
        half_edge_map[(arc.circle_idx, key_of(arc.v_end), key_of(arc.v_start))] = he_rev

    # Step 4: Link half-edges around each vertex (CCW order) so we can "turn left" to walk faces
    vertex_to_outgoing: Dict[Vertex, List[HalfEdge]] = {}
    for he in half_edges:
        vertex_to_outgoing.setdefault(he.origin, []).append(he)

    def angle_of_halfedge(he: HalfEdge) -> float:
        c = circles[he.arc.circle_idx]
        # angle at origin along the circle towards arc.mid (direction of travel on the circle)
        # Use arc.start_angle when moving forward; for twin, we encoded reverse angles.
        return he.arc.start_angle

    for _, hes in vertex_to_outgoing.items():
        hes.sort(key=angle_of_halfedge)

    # Step 5: Link 'next' pointers: at the end of an edge, choose the "most counterclockwise" outgoing from that endpoint,
    #    i.e., from he.twin.origin, pick the previous edge in the circular order (left turn).
    for he in half_edges:
        end_vertex = he.twin.origin  # type: ignore
        outgoing = vertex_to_outgoing[end_vertex]
        if len(outgoing) == 0:
            continue  # isolated vertex?
        # Find he.twin in outgoing list
        idx = outgoing.index(he.twin)  # type: ignore
        # The next half-edge is the one before he.twin in CCW order (left turn)
        he.next = outgoing[idx - 1]  # wrap-around works in Python
        if he.next:
            he.next.prev = he

    # Step 6: Walk faces by following next pointers, assign bitmask labels
    faces: List[Face] = []
    visited_halfedges = set()  # store id() of visited half-edges

    def walk_face(start_he: HalfEdge) -> List[HalfEdge]:
        cycle = []
        he = start_he
        while True:
            cycle.append(he)
            visited_halfedges.add(id(he))
            he = he.next
            if he is None or he == start_he:
                break
        return cycle

    for he in half_edges:
        if id(he) in visited_halfedges:
            continue
        cycle = walk_face(he)
        if not cycle:
            continue

        # Sample a point inside the face by taking mid-angle of the first half-edge's arc
        # Strategy: take mid-angle of the arc, get point on circle, nudge inward
        mid_theta = he.arc.mid_angle()
        c = circles[he.arc.circle_idx]
        boundary_pt = _param_point_on_circle(c, mid_theta)
        sample_pt = _small_inward_offset(boundary_pt, c)

        bitmask = _classify_point_bitmask(sample_pt, circles)
        face = Face(boundary=cycle, bitmask=bitmask)
        for e in cycle:
            e.face = face
        faces.append(face)

    return vertices, arcs, faces
