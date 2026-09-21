"""Min-max: banded coverage of compact pieces, plus a path-split fallback."""

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

Point = tuple[int, int]


def _segment_length(a: Point, b: Point) -> int:
    return abs(b[0] - a[0]) + abs(b[1] - a[1])


def _path_length(points: list[Point]) -> int:
    return sum(_segment_length(a, b) for a, b in zip(points, points[1:]))


def _cycle_length(points: list[Point]) -> int:
    points = _remove_immediate_repeats(points)
    if len(points) <= 1:
        return 0
    return _path_length(points) + _segment_length(points[0], points[-1])


def _cutter_span(offsets: list[Point]) -> tuple[int, int]:
    xs = [dx for dx, _ in offsets]
    ys = [dy for _, dy in offsets]
    return max(xs) - min(xs) + 1, max(ys) - min(ys) + 1


def _cells_to_rows(cells: set[Point]) -> dict[int, list[int]]:
    rows: dict[int, list[int]] = defaultdict(list)
    for x, y in cells:
        rows[y].append(x)
    return dict(rows)


def _cells_to_cols(cells: set[Point]) -> dict[int, list[int]]:
    cols: dict[int, list[int]] = defaultdict(list)
    for x, y in cells:
        cols[x].append(y)
    return dict(cols)


def _prefix_sums(weights: list[int]) -> list[int]:
    prefix = [0]
    for w in weights:
        prefix.append(prefix[-1] + w)
    return prefix


def _fixed_cy_span(
    cells: set[Point], offsets: list[Point], cy: int
) -> tuple[int, int] | None:
    cx_lo = float("inf")
    cx_hi = float("-inf")
    for x, y in cells:
        opts = [x - dx for dx, dy in offsets if y - dy == cy]
        if not opts:
            return None
        cx_lo = min(cx_lo, min(opts))
        cx_hi = max(cx_hi, max(opts))
    return int(cx_lo), int(cx_hi)


def _band_h_sweep(cells: set[Point], offsets: list[Point]) -> list[Point] | None:
    """One horizontal pass that covers every cell, if some center-y works."""
    cys = sorted({y - dy for _, y in cells for _, dy in offsets})
    best: list[Point] | None = None
    best_span = None
    for cy in cys:
        span = _fixed_cy_span(cells, offsets, cy)
        if span is None:
            continue
        cx_lo, cx_hi = span
        sweep = [(cx_lo, cy)] if cx_lo == cx_hi else [(cx_lo, cy), (cx_hi, cy)]
        width = cx_hi - cx_lo
        if best is None or width < best_span:
            best = sweep
            best_span = width
    return best


def _band_v_sweep(cells: set[Point], offsets: list[Point]) -> list[Point] | None:
    swapped = {(y, x) for x, y in cells}
    swapped_off = [(dy, dx) for dx, dy in offsets]
    sweep = _band_h_sweep(swapped, swapped_off)
    if sweep is None:
        return None
    return [(cy, cx) for cx, cy in sweep]


def _group_keys(keys: list[int], step: int) -> list[list[int]]:
    keys = sorted(set(keys))
    bands: list[list[int]] = []
    i = 0
    while i < len(keys):
        start = keys[i]
        band = []
        while i < len(keys) and keys[i] < start + step:
            band.append(keys[i])
            i += 1
        bands.append(band)
    return bands


def _banded_row_cover(
    cells: set[Point],
    offsets: list[Point],
    reverse: bool = False,
    flip: bool = False,
) -> list[Point]:
    _, height = _cutter_span(offsets)
    by_y: dict[int, set[Point]] = defaultdict(set)
    for cell in cells:
        by_y[cell[1]].add(cell)
    bands = _group_keys(list(by_y), max(1, height))
    if reverse:
        bands = list(reversed(bands))
    corners: list[Point] = []
    for index, band_ys in enumerate(bands):
        chunk = set()
        for y in band_ys:
            chunk |= by_y[y]
        sweep = _band_h_sweep(chunk, offsets)
        if sweep is None:
            rows = _cells_to_rows(chunk)
            sweep = []
            for y in sorted(rows):
                sweep.extend(_row_sweep(sorted(rows[y]), y, offsets))
            sweep = _remove_immediate_repeats(sweep)
        if (index % 2 == 1) != flip:
            sweep = list(reversed(sweep))
        if corners and sweep:
            corners.extend(_manhattan_connect(corners[-1], sweep[0]))
        corners.extend(sweep)
    return _remove_immediate_repeats(corners)


def _banded_col_cover(
    cells: set[Point],
    offsets: list[Point],
    reverse: bool = False,
    flip: bool = False,
) -> list[Point]:
    width, _ = _cutter_span(offsets)
    by_x: dict[int, set[Point]] = defaultdict(set)
    for cell in cells:
        by_x[cell[0]].add(cell)
    bands = _group_keys(list(by_x), max(1, width))
    if reverse:
        bands = list(reversed(bands))
    corners: list[Point] = []
    for index, band_xs in enumerate(bands):
        chunk = set()
        for x in band_xs:
            chunk |= by_x[x]
        sweep = _band_v_sweep(chunk, offsets)
        if sweep is None:
            cols = _cells_to_cols(chunk)
            swapped = [(dy, dx) for dx, dy in offsets]
            sweep = []
            for x in sorted(cols):
                raw = _row_sweep(sorted(cols[x]), x, swapped)
                sweep.extend((cy, cx) for cx, cy in raw)
            sweep = _remove_immediate_repeats(sweep)
        if (index % 2 == 1) != flip:
            sweep = list(reversed(sweep))
        if corners and sweep:
            corners.extend(_manhattan_connect(corners[-1], sweep[0]))
        corners.extend(sweep)
    return _remove_immediate_repeats(corners)


def _best_tour_for_cells(cells: set[Point], offsets: list[Point]) -> list[Point]:
    if not cells:
        return [(0, 0)]
    best: list[Point] | None = None
    best_len = float("inf")
    for builder in (_banded_row_cover, _banded_col_cover):
        for reverse in (False, True):
            for flip in (False, True):
                tour = builder(cells, offsets, reverse=reverse, flip=flip)
                if not tour:
                    continue
                closed = _close_tour(tour)
                length = _cycle_length(closed)
                if length < best_len:
                    best_len = length
                    best = closed
    if best is None:
        x, y = next(iter(cells))
        return [(x, y)]
    return best


def _split_weights(weights: list[int], k: int) -> list[tuple[int, int]]:
    """Min-max split of a sequence. Returns [start, end) index ranges."""
    n = len(weights)
    if n == 0:
        return []
    if k <= 1 or n == 1:
        return [(0, n)]
    prefix = _prefix_sums(weights)

    def needed(limit: int) -> int:
        used = 0
        i = 0
        while i < n:
            used += 1
            j = i
            while j < n and prefix[j + 1] - prefix[i] <= limit:
                j += 1
            if j == i:
                return n + 1
            i = j
        return used

    lo = max(weights)
    hi = prefix[-1]
    best = hi
    while lo <= hi:
        mid = (lo + hi) // 2
        if needed(mid) <= k:
            best = mid
            hi = mid - 1
        else:
            lo = mid + 1

    ranges: list[tuple[int, int]] = []
    i = 0
    while i < n and len(ranges) < k:
        j = i
        while j < n and prefix[j + 1] - prefix[i] <= best:
            j += 1
        if j == i:
            j = i + 1
        ranges.append((i, j))
        i = j
    if i < n and ranges:
        a, _ = ranges[-1]
        ranges[-1] = (a, n)
    while len(ranges) < k:
        ranges.append((n - 1, n) if n else (0, 0))
    return ranges[:k]


def _partition_by_axis(cells: set[Point], k: int, axis: int) -> list[set[Point]]:
    buckets: dict[int, set[Point]] = defaultdict(set)
    for x, y in cells:
        buckets[x if axis == 0 else y].add((x, y))
    keys = sorted(buckets)
    if not keys:
        return [set() for _ in range(k)]
    weights = [len(buckets[key]) for key in keys]
    groups: list[set[Point]] = []
    for a, b in _split_weights(weights, k):
        part: set[Point] = set()
        for key in keys[a:b]:
            part |= buckets[key]
        groups.append(part)
    while len(groups) < k:
        groups.append(set())
    return groups


def _grid_line(a: Point, b: Point) -> list[Point]:
    if a == b:
        return [a]
    if a[0] == b[0]:
        step = 1 if b[1] > a[1] else -1
        return [(a[0], y) for y in range(a[1], b[1] + step, step)]
    if a[1] == b[1]:
        step = 1 if b[0] > a[0] else -1
        return [(x, a[1]) for x in range(a[0], b[0] + step, step)]
    via = (b[0], a[1])
    return _grid_line(a, via)[:-1] + _grid_line(via, b)


def _assign_along_cover(
    cells: set[Point], cover: list[Point], offsets: list[Point], k: int
) -> list[set[Point]]:
    """Give each cell to the first center on the covering that sweeps it, then split."""
    remaining = set(cells)
    order: list[Point] = []
    seen: set[Point] = set()
    for a, b in zip(cover, cover[1:] + cover[:1]):
        for cx, cy in _grid_line(a, b)[:-1] or [a]:
            hit = [(cx + dx, cy + dy) for dx, dy in offsets]
            for cell in hit:
                if cell in remaining and cell not in seen:
                    seen.add(cell)
                    order.append(cell)
                    remaining.discard(cell)
        if not remaining:
            break
    order.extend(sorted(remaining))
    if not order:
        return [set() for _ in range(k)]
    weights = [1] * len(order)
    groups: list[set[Point]] = []
    for a, b in _split_weights(weights, k):
        groups.append(set(order[a:b]))
    while len(groups) < k:
        groups.append(set())
    return groups


def _tours_from_groups(
    groups: list[set[Point]], offsets: list[Point]
) -> tuple[list[list[Point]], int]:
    tours: list[list[Point]] = []
    worst = 0
    for group in groups:
        tour = _best_tour_for_cells(group, offsets) if group else [(0, 0)]
        tours.append(tour)
        worst = max(worst, _cycle_length(tour))
    return tours, worst


def _rebalance_groups(
    groups: list[set[Point]], offsets: list[Point], axis: int
) -> list[set[Point]]:
    """Move a boundary line of cells from the heaviest piece to a neighbor."""
    groups = [set(g) for g in groups]
    for _ in range(8):
        lengths = []
        for g in groups:
            lengths.append(_cycle_length(_best_tour_for_cells(g, offsets)) if g else 0)
        heavy = max(range(len(groups)), key=lambda i: lengths[i])
        improved = False
        for nb in (heavy - 1, heavy + 1):
            if nb < 0 or nb >= len(groups) or not groups[heavy]:
                continue
            keys = sorted({c[axis] for c in groups[heavy]})
            if len(keys) < 2:
                continue
            edge = keys[0] if nb < heavy else keys[-1]
            moved = {c for c in groups[heavy] if c[axis] == edge}
            if moved == groups[heavy]:
                continue
            trial_h = groups[heavy] - moved
            trial_n = groups[nb] | moved
            new_h = _cycle_length(_best_tour_for_cells(trial_h, offsets))
            new_n = _cycle_length(_best_tour_for_cells(trial_n, offsets))
            if max(new_h, new_n) < max(lengths[heavy], lengths[nb]):
                groups[heavy] = trial_h
                groups[nb] = trial_n
                improved = True
                break
        if not improved:
            break
    return groups


def solve_instance(instance: CGSHOP2027Instance) -> CGSHOP2027Solution:
    """
    Split the region into k compact pieces (by x, by y, or along a covering),
    cover each piece with cutter-height/width bands, and keep the assignment
    whose longest closed tour is shortest.
    """
    region = rasterize(instance.region_to_cover)
    _, offsets = _cutter_offsets(instance)
    cells = set(region.cells())
    k = instance.number_of_cutters
    fallback = next(iter(cells)) if cells else (0, 0)

    candidates: list[tuple[str, list[list[Point]]]] = []

    for axis, name in ((0, "x-strips"), (1, "y-strips")):
        groups = _partition_by_axis(cells, k, axis)
        groups = _rebalance_groups(groups, offsets, axis)
        tours, _ = _tours_from_groups(groups, offsets)
        candidates.append((name, tours))

    if k > 1:
        seed = _banded_row_cover(cells, offsets)
        if len(seed) >= 2:
            groups = _assign_along_cover(cells, seed, offsets, k)
            tours, _ = _tours_from_groups(groups, offsets)
            candidates.append(("cover-order", tours))
        seed = _banded_col_cover(cells, offsets)
        if len(seed) >= 2:
            groups = _assign_along_cover(cells, seed, offsets, k)
            tours, _ = _tours_from_groups(groups, offsets)
            candidates.append(("cover-order-col", tours))
    else:
        candidates.append(("one-cutter", [_best_tour_for_cells(cells, offsets)]))

    best_tours: list[list[Point]] | None = None
    best_name = "x-strips"
    best_max = float("inf")
    for name, tours in candidates:
        worst = max(_cycle_length(t) for t in tours) if tours else float("inf")
        if worst < best_max:
            best_max = worst
            best_tours = tours
            best_name = name

    if not best_tours:
        best_tours = [[fallback]]

    out: list[CutterTour] = []
    for tour in best_tours:
        anchors = _close_tour(_remove_immediate_repeats(tour))
        if not anchors:
            anchors = [fallback]
        out.append(_anchors_to_tour(anchors))
    while len(out) < k:
        out.append(out[-1].model_copy(deep=True))

    return CGSHOP2027Solution(
        instance_uid=instance.instance_uid,
        tours=out[:k],
        meta={"algorithm": "minmax-split", "cover": best_name},
    )


__all__ = ["solve_instance", "verify_solution"]
