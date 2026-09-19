"""Min-max split: one covering tour, then cut into k tours."""

from __future__ import annotations

from collections import defaultdict

from cgshop2027_pyutils.grid import rasterize
from cgshop2027_pyutils.schemas import CGSHOP2027Instance, CGSHOP2027Solution, CutterTour

from solver.coverage import (
    _anchors_to_tour,
    _close_tour,
    _cutter_offsets,
    _remove_immediate_repeats,
    _serpentine_rows,
    verify_solution,
)


def _segment_length(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(b[0] - a[0]) + abs(b[1] - a[1])


def _split_path(points: list[tuple[int, int]], k: int) -> list[list[tuple[int, int]]]:
    """Cut a polyline into k pieces with about the same length."""
    if k <= 1 or len(points) <= 1:
        return [points]

    lengths = [
        _segment_length(points[i], points[i + 1]) for i in range(len(points) - 1)
    ]
    total = sum(lengths)
    if total == 0:
        return [points] + [points[:1] for _ in range(k - 1)]

    cuts = [total * (i + 1) / k for i in range(k - 1)]
    pieces: list[list[tuple[int, int]]] = []
    current = [points[0]]
    acc = 0.0
    cut_i = 0

    for i, length in enumerate(lengths):
        acc += length
        current.append(points[i + 1])
        if cut_i < len(cuts) and acc >= cuts[cut_i]:
            pieces.append(current)
            current = [points[i + 1]]
            cut_i += 1

    if len(current) == 1 and pieces:
        pieces[-1].append(current[0])
    else:
        pieces.append(current)

    while len(pieces) < k:
        pieces.append(list(pieces[-1]) if pieces else [(0, 0)])
    return pieces[:k]


def solve_instance(instance: CGSHOP2027Instance) -> CGSHOP2027Solution:
    """
    Cover the whole region with one sweep, then split that path into k tours
    so the longest piece is as even as we can get.
    """
    region = rasterize(instance.region_to_cover)
    _, offsets = _cutter_offsets(instance)
    rows: dict[int, list[int]] = defaultdict(list)
    for x, y in region.cells():
        rows[y].append(x)

    covering = _serpentine_rows(rows, offsets)
    if not covering:
        x0, y0, _, _ = region.bounds
        covering = [(x0, y0)]

    pieces = _split_path(covering, instance.number_of_cutters)
    tours: list[CutterTour] = []
    for piece in pieces:
        anchors = _close_tour(_remove_immediate_repeats(piece))
        if not anchors:
            x0, y0, _, _ = region.bounds
            anchors = [(x0, y0)]
        tours.append(_anchors_to_tour(anchors))

    while len(tours) < instance.number_of_cutters:
        tours.append(tours[-1].model_copy(deep=True))

    return CGSHOP2027Solution(
        instance_uid=instance.instance_uid,
        tours=tours[: instance.number_of_cutters],
        meta={"algorithm": "minmax-split"},
    )


__all__ = ["solve_instance", "verify_solution"]
