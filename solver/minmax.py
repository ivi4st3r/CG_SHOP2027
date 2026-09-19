"""Min-max split: cover the region, then cut into k balanced tours."""

from __future__ import annotations

from collections import defaultdict

from cgshop2027_pyutils.grid import rasterize
from cgshop2027_pyutils.schemas import CGSHOP2027Instance, CGSHOP2027Solution, CutterTour

from solver.coverage import (
    _anchors_to_tour,
    _close_tour,
    _cutter_offsets,
    _manhattan_connect,
    _remove_immediate_repeats,
    _row_sweep,
    verify_solution,
)


def _segment_length(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(b[0] - a[0]) + abs(b[1] - a[1])


def _path_length(points: list[tuple[int, int]]) -> int:
    return sum(
        _segment_length(points[i], points[i + 1]) for i in range(len(points) - 1)
    )


def _prefix_lengths(points: list[tuple[int, int]]) -> list[int]:
    prefix = [0]
    for i in range(len(points) - 1):
        prefix.append(prefix[-1] + _segment_length(points[i], points[i + 1]))
    return prefix


def _cycle_length(points: list[tuple[int, int]]) -> int:
    """Length of the closed tour, including the return to the start."""
    points = _remove_immediate_repeats(points)
    if len(points) <= 1:
        return 0
    return _path_length(points) + _segment_length(points[0], points[-1])


def _cycle_range(points: list[tuple[int, int]], prefix: list[int], i: int, j: int) -> int:
    if j <= i:
        return 0
    return prefix[j] - prefix[i] + _segment_length(points[i], points[j])


def _open_row_cover(
    rows: dict[int, list[int]], offsets: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    corners: list[tuple[int, int]] = []
    for index, y in enumerate(sorted(rows)):
        sweep = _row_sweep(sorted(rows[y]), y, offsets)
        if index % 2 == 1:
            sweep = list(reversed(sweep))
        if corners and sweep:
            corners.extend(_manhattan_connect(corners[-1], sweep[0]))
        corners.extend(sweep)
    return _remove_immediate_repeats(corners)


def _open_col_cover(
    cols: dict[int, list[int]], offsets: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Same zigzag, but along columns. Built by swapping x/y."""
    swapped = [(dy, dx) for dx, dy in offsets]
    corners: list[tuple[int, int]] = []
    for index, x in enumerate(sorted(cols)):
        sweep_swapped = _row_sweep(sorted(cols[x]), x, swapped)
        sweep = [(cy, cx) for cx, cy in sweep_swapped]
        if index % 2 == 1:
            sweep = list(reversed(sweep))
        if corners and sweep:
            corners.extend(_manhattan_connect(corners[-1], sweep[0]))
        corners.extend(sweep)
    return _remove_immediate_repeats(corners)


def _farthest_end(
    points: list[tuple[int, int]], prefix: list[int], start: int, limit: int
) -> int | None:
    """Largest end > start such that the closed piece fits in limit."""
    n = len(points)
    last_ok: int | None = None
    for end in range(start + 1, n):
        if _cycle_range(points, prefix, start, end) <= limit:
            last_ok = end
        else:
            break
    return last_ok


def _tours_needed(
    points: list[tuple[int, int]], prefix: list[int], limit: int
) -> int:
    n = len(points)
    if n <= 1:
        return 1
    used = 0
    i = 0
    while i < n - 1:
        end = _farthest_end(points, prefix, i, limit)
        if end is None:
            return n
        used += 1
        i = end
    return used


def _split_for_limit(
    points: list[tuple[int, int]], prefix: list[int], limit: int
) -> list[list[tuple[int, int]]] | None:
    n = len(points)
    if n <= 1:
        return [points]
    pieces: list[list[tuple[int, int]]] = []
    i = 0
    while i < n - 1:
        end = _farthest_end(points, prefix, i, limit)
        if end is None:
            return None
        pieces.append(points[i : end + 1])
        i = end
    return pieces


def _split_minmax(
    points: list[tuple[int, int]], k: int
) -> list[list[tuple[int, int]]]:
    """
    Cut the covering path into at most k tours.

    Cut points minimize the longest *closed* tour (path plus return home).
    """
    if k <= 1 or len(points) <= 1:
        return [points]

    prefix = _prefix_lengths(points)
    lo = 0
    for i in range(len(points) - 1):
        lo = max(lo, _cycle_range(points, prefix, i, i + 1))
    hi = _cycle_range(points, prefix, 0, len(points) - 1)
    best = hi

    while lo <= hi:
        mid = (lo + hi) // 2
        if _tours_needed(points, prefix, mid) <= k:
            best = mid
            hi = mid - 1
        else:
            lo = mid + 1

    pieces = _split_for_limit(points, prefix, best)
    if pieces is None:
        return [points]
    while len(pieces) < k:
        pieces.append([pieces[-1][-1]])
    return pieces[:k]


def _best_covering(
    rows: dict[int, list[int]],
    cols: dict[int, list[int]],
    offsets: list[tuple[int, int]],
    k: int,
) -> tuple[list[list[tuple[int, int]]], str]:
    candidates: list[tuple[str, list[tuple[int, int]]]] = []
    row_cover = _open_row_cover(rows, offsets)
    if row_cover:
        candidates.append(("row-cover", row_cover))
    col_cover = _open_col_cover(cols, offsets)
    if col_cover:
        candidates.append(("col-cover", col_cover))

    best_pieces: list[list[tuple[int, int]]] | None = None
    best_name = "row-cover"
    best_max = float("inf")
    for name, cover in candidates:
        pieces = _split_minmax(cover, k)
        worst = max(_cycle_length(p) for p in pieces)
        if worst < best_max:
            best_max = worst
            best_pieces = pieces
            best_name = name
    if best_pieces is None:
        return [[(0, 0)]], "empty"
    return best_pieces, best_name


def solve_instance(instance: CGSHOP2027Instance) -> CGSHOP2027Solution:
    """
    Build one covering path (row sweep and column sweep), split it into
    k tours, and keep the split whose longest closed tour is shorter.
    """
    region = rasterize(instance.region_to_cover)
    _, offsets = _cutter_offsets(instance)
    rows: dict[int, list[int]] = defaultdict(list)
    cols: dict[int, list[int]] = defaultdict(list)
    for x, y in region.cells():
        rows[y].append(x)
        cols[x].append(y)

    k = instance.number_of_cutters
    pieces, cover_name = _best_covering(rows, cols, offsets, k)

    tours: list[CutterTour] = []
    for piece in pieces:
        anchors = _close_tour(_remove_immediate_repeats(piece))
        if not anchors:
            x0, y0, _, _ = region.bounds
            anchors = [(x0, y0)]
        tours.append(_anchors_to_tour(anchors))

    while len(tours) < k:
        tours.append(tours[-1].model_copy(deep=True))

    return CGSHOP2027Solution(
        instance_uid=instance.instance_uid,
        tours=tours[:k],
        meta={"algorithm": "minmax-split", "cover": cover_name},
    )


__all__ = ["solve_instance", "verify_solution"]
