"""Compute anchor positions and tours that sweep a region."""

from __future__ import annotations

from collections import defaultdict

from cgshop2027_pyutils.grid import CellSet, rasterize, rasterize_ring
from cgshop2027_pyutils.schemas import CGSHOP2027Instance, CGSHOP2027Solution, CutterTour
from cgshop2027_pyutils.verify import SolutionValidator


def _cutter_offsets(instance: CGSHOP2027Instance) -> tuple[CellSet, list[tuple[int, int]]]:
    center_x, center_y = instance.cutter_center
    cutter = rasterize_ring(instance.cutter).translated(-center_x, -center_y)
    return cutter, list(cutter.cells())


def _row_center_x_range(
    row_xs: list[int], y: int, cy: int, offsets: list[tuple[int, int]]
) -> tuple[int, int] | None:
    """Inclusive cx range that covers every cell in the row at fixed center y = cy."""
    cx_lo = float("inf")
    cx_hi = float("-inf")
    for x in row_xs:
        local_lo = float("inf")
        local_hi = float("-inf")
        for dx, dy in offsets:
            if y - dy != cy:
                continue
            cx = x - dx
            local_lo = min(local_lo, cx)
            local_hi = max(local_hi, cx)
        if local_lo == float("inf"):
            return None
        cx_lo = min(cx_lo, local_lo)
        cx_hi = max(cx_hi, local_hi)
    return int(cx_lo), int(cx_hi)


def _choose_row_cy(y: int, offsets: list[tuple[int, int]]) -> list[int]:
    """Center-y values that can contribute to covering row y."""
    dys = {dy for _, dy in offsets}
    return sorted({y - dy for dy in dys})


def _row_sweep(
    row_xs: list[int], y: int, offsets: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Anchor positions covering one grid row, as a horizontal sweep."""
    if not row_xs:
        return []
    best: list[tuple[int, int]] | None = None
    for cy in _choose_row_cy(y, offsets):
        span = _row_center_x_range(row_xs, y, cy, offsets)
        if span is None:
            continue
        cx_lo, cx_hi = span
        if cx_lo == cx_hi:
            sweep = [(cx_lo, cy)]
        else:
            sweep = [(cx_lo, cy), (cx_hi, cy)]
        if best is None or len(sweep) < len(best):
            best = sweep
    if best is None:
        anchors: list[tuple[int, int]] = []
        for x in row_xs:
            for dx, dy in offsets:
                anchors.append((x - dx, y - dy))
        return _remove_immediate_repeats(anchors)
    return best


def _manhattan_connect(
    start: tuple[int, int], end: tuple[int, int]
) -> list[tuple[int, int]]:
    """Corner points connecting two anchors with axis-parallel moves."""
    if start == end:
        return []
    if start[0] == end[0] or start[1] == end[1]:
        return [end]
    return [(end[0], start[1]), end]


def _remove_immediate_repeats(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not points:
        return points
    out = [points[0]]
    for point in points[1:]:
        if point != out[-1]:
            out.append(point)
    return out


def _close_tour(corners: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Ensure the closing edge back to the start is axis-parallel."""
    if len(corners) < 2:
        return corners
    first = corners[0]
    last = corners[-1]
    if first[0] == last[0] or first[1] == last[1]:
        return corners
    aligned = (first[0], last[1])
    corners = corners + _manhattan_connect(last, aligned)
    last = corners[-1]
    if first[0] != last[0] and first[1] != last[1]:
        corners = corners + _manhattan_connect(last, (last[0], first[1]))
    return corners


def _serpentine_rows(
    rows: dict[int, list[int]], offsets: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Build a boustrophedon path covering all rows."""
    corners: list[tuple[int, int]] = []
    for index, y in enumerate(sorted(rows)):
        sweep = _row_sweep(sorted(rows[y]), y, offsets)
        if index % 2 == 1:
            sweep = list(reversed(sweep))
        if corners and sweep:
            corners.extend(_manhattan_connect(corners[-1], sweep[0]))
        corners.extend(sweep)
    return _close_tour(_remove_immediate_repeats(corners))


def _split_rows(
    rows: dict[int, list[int]], num_parts: int
) -> list[dict[int, list[int]]]:
    """Split rows into vertical strips by x coordinate."""
    if num_parts <= 1:
        return [rows]
    all_x = sorted({x for xs in rows.values() for x in xs})
    if not all_x:
        return [rows] * num_parts
    width = len(all_x) / num_parts
    bounds = [all_x[int(i * width)] for i in range(num_parts)]
    bounds[0] = all_x[0]
    parts: list[dict[int, list[int]]] = [defaultdict(list) for _ in range(num_parts)]
    for y, xs in rows.items():
        for x in xs:
            part = min(num_parts - 1, _strip_index(x, bounds))
            parts[part][y].append(x)
    return [dict(part) for part in parts]


def _strip_index(x: int, bounds: list[int]) -> int:
    index = 0
    for i, bound in enumerate(bounds):
        if x >= bound:
            index = i
    return index


def _anchors_to_tour(anchors: list[tuple[int, int]]) -> CutterTour:
    if not anchors:
        anchors = [(0, 0)]
    xs, ys = zip(*anchors, strict=True)
    return CutterTour(x=list(xs), y=list(ys))


def solve_instance(instance: CGSHOP2027Instance) -> CGSHOP2027Solution:
    """
    Build a feasible solution using boustrophedon row sweeps.

    With multiple cutters the region is split into vertical strips; each cutter
    sweeps its strip independently.
    """
    region = rasterize(instance.region_to_cover)
    _, offsets = _cutter_offsets(instance)
    rows: dict[int, list[int]] = defaultdict(list)
    for x, y in region.cells():
        rows[y].append(x)

    row_parts = _split_rows(rows, instance.number_of_cutters)
    tours: list[CutterTour] = []
    for part in row_parts:
        anchors = _serpentine_rows(part, offsets)
        if not anchors:
            x0, y0, _, _ = region.bounds
            anchors = [(x0, y0)]
        tours.append(_anchors_to_tour(anchors))

    while len(tours) < instance.number_of_cutters:
        tours.append(tours[-1].model_copy(deep=True))

    return CGSHOP2027Solution(
        instance_uid=instance.instance_uid,
        tours=tours[: instance.number_of_cutters],
        meta={"algorithm": "boustrophedon-strip"},
    )


def verify_solution(
    instance: CGSHOP2027Instance, solution: CGSHOP2027Solution
) -> list[str]:
    return SolutionValidator(instance).check_for_errors(solution)
