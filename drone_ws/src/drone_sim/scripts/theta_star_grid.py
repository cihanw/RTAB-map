#!/usr/bin/env python3
"""2D Theta* grid search - pure algorithm module (NO rclpy, NOT a ROS node).

Imported by theta_star_planner.py. Testable standalone with plain numpy.

REWRITE 2026-07-11: the global planner is now purely 2D (xy). Rationale:
vertical routing in this environment has only ever caused harm - a live
incident had a 3D Theta* waypoint at z=1.8m route the drone above the
feature-rich band (objects at z=0-1.5, camera horizontal/no tilt), losing
visual odometry for 8+ minutes. Obstacle points within a configurable z band
are PROJECTED onto the xy plane; the search itself never sees z. Vertical
motion remains entirely the local planner's job (it stays 3D).

Vectorization discipline (see tasks/lessons.md - a 1.4M-point cloud once
pegged a CPU core because of Python set/tuple membership tests): batch
operations use packed-int64 keys + np.unique + np.searchsorted; plain Python
set lookups are allowed ONLY for single-cell queries inside tight search
loops, where per-call numpy overhead exceeds a hash lookup.
"""
import heapq
import math

import numpy as np

# 21 bits per axis (+/- ~1M cells) - proven sufficient by the 3D version;
# two axes now fit in 42 bits of an int64.
CELL_BITS = 21
CELL_OFFSET = 1 << (CELL_BITS - 1)
CELL_MASK = (1 << CELL_BITS) - 1

# 8-connected neighborhood: {-1,0,1}^2 minus the origin.
NEIGHBOR_OFFSETS_8 = np.array(
    [(dx, dy)
     for dx in (-1, 0, 1) for dy in (-1, 0, 1)
     if not (dx == 0 and dy == 0)],
    dtype=np.int64)


def cell_keys(coords):
    """Packs (N,2) int cell coordinates into sortable (N,) int64 keys."""
    x = (coords[:, 0].astype(np.int64) + CELL_OFFSET) & CELL_MASK
    y = (coords[:, 1].astype(np.int64) + CELL_OFFSET) & CELL_MASK
    return (x << CELL_BITS) | y


def _unpack_cell_keys(keys):
    y = (keys & CELL_MASK) - CELL_OFFSET
    x = ((keys >> CELL_BITS) & CELL_MASK) - CELL_OFFSET
    return np.stack([x, y], axis=-1)


def _pack_single(cell):
    x = (int(cell[0]) + CELL_OFFSET) & CELL_MASK
    y = (int(cell[1]) + CELL_OFFSET) & CELL_MASK
    return (x << CELL_BITS) | y


def world_to_cell(point_xy, resolution):
    """(x, y) world -> integer cell tuple. NO half-cell offset - nbv_planner
    rounds its frontier voxels the same way; an offset here would create a
    spurious residual gap between the path end and the NBV target."""
    return (int(round(float(point_xy[0]) / resolution)),
            int(round(float(point_xy[1]) / resolution)))


def cell_to_world(cell, resolution):
    return np.array([cell[0], cell[1]], dtype=np.float64) * resolution


class OccupancyGrid2D:
    """Sparse (keyed) 2D obstacle grid built by projecting a 3D obstacle
    cloud onto the xy plane.

    Built ONLY from the obstacle cloud - free/unknown space is never
    consumed. This encodes the "unknown = free" rule: a cell is blocked iff
    it is an obstacle (or within the inflation radius of one); everything
    else - known-free or never-observed alike - is traversable. Frontier
    goals sit on the known/unknown boundary by definition, so treating
    unknown as blocked would make every exploration target unreachable.

    The grid is deliberately UNBOUNDED in xy (sparse keys, open-ended
    world); search cost is bounded by the caller's max_iterations, and the
    Euclidean heuristic keeps expansion focused toward the goal.
    """

    def __init__(self, resolution, inflation_radius_m,
                 projection_z_min, projection_z_max):
        self.resolution = float(resolution)
        self.inflation_radius_cells = max(
            1, int(round(inflation_radius_m / resolution)))
        # Which obstacle points participate in the 2D world: anything the
        # airframe could plausibly occupy. Overhangs ABOVE this band do not
        # wall off floor space the drone can legally use.
        self.projection_z_min = float(projection_z_min)
        self.projection_z_max = float(projection_z_max)

        self._blocked_keys_sorted = np.zeros((0,), dtype=np.int64)
        self._blocked_keys_set = set()

        # Precomputed L2 disk inflation offsets ((2r+1)^2 grid masked to the
        # circle, ~37 cells at r=3 vs ~123 for the old 3D sphere). Disk (not
        # square) matches the local planner's distance-based repulsion
        # semantics - corners of a square would over-inflate diagonals.
        r = self.inflation_radius_cells
        offsets_1d = np.arange(-r, r + 1)
        ox, oy = np.meshgrid(offsets_1d, offsets_1d, indexing='ij')
        mask = (ox ** 2 + oy ** 2) <= r * r
        self._inflation_offsets = np.stack(
            [ox[mask], oy[mask]], axis=-1).astype(np.int64)

    def update_from_points(self, obstacle_points_xyz):
        """obstacle_points_xyz: (N,3) float64 world-frame points. Projects
        the z band onto xy, voxelizes, inflates - all vectorized."""
        if obstacle_points_xyz.shape[0] == 0:
            self._blocked_keys_sorted = np.zeros((0,), dtype=np.int64)
            self._blocked_keys_set = set()
            return
        z = obstacle_points_xyz[:, 2]
        band = (z >= self.projection_z_min) & (z <= self.projection_z_max)
        pts_xy = obstacle_points_xyz[band, :2]
        if pts_xy.shape[0] == 0:
            self._blocked_keys_sorted = np.zeros((0,), dtype=np.int64)
            self._blocked_keys_set = set()
            return

        int_coords = np.round(pts_xy / self.resolution).astype(np.int64)
        # 1D-unique on packed keys (documented ~7x faster than
        # np.unique(axis=0) row-sorting on large clouds).
        raw_keys = np.unique(cell_keys(int_coords))
        raw_coords = _unpack_cell_keys(raw_keys)

        # Vectorized disk inflation: broadcast the offset table over all
        # obstacle cells, no per-point Python loop.
        inflated = raw_coords[:, None, :] + self._inflation_offsets[None, :, :]
        flat = inflated.reshape(-1, 2)
        inflated_keys = np.unique(cell_keys(flat))

        self._blocked_keys_sorted = inflated_keys
        self._blocked_keys_set = set(inflated_keys.tolist())

    def is_blocked(self, cells):
        """(N,2) int -> (N,) bool. Batch searchsorted membership test."""
        if self._blocked_keys_sorted.shape[0] == 0:
            return np.zeros(cells.shape[0], dtype=bool)
        keys = cell_keys(cells)
        idx = np.searchsorted(self._blocked_keys_sorted, keys)
        idx = np.clip(idx, 0, len(self._blocked_keys_sorted) - 1)
        return self._blocked_keys_sorted[idx] == keys

    def is_blocked_single(self, cell):
        """Single cell, Python set lookup - the CONSCIOUS exception to the
        vectorize-everything rule: for N=1, a hash lookup beats numpy's
        array-construction overhead. Used only in search inner loops."""
        return _pack_single(cell) in self._blocked_keys_set


def line_of_sight_2d(grid, a, b):
    """Integer 2D Bresenham traversal - True iff the straight segment from
    cell a to cell b crosses no blocked cell (endpoint included). Early-exits
    on the first blocked cell. This check is what makes Theta* any-angle."""
    x0, y0 = int(a[0]), int(a[1])
    x1, y1 = int(b[0]), int(b[1])

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1

    x, y = x0, y0
    err = dx - dy
    while True:
        if grid.is_blocked_single((x, y)):
            return False
        if x == x1 and y == y1:
            return True
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


def nearest_free_cell(grid, cell, radius_cells):
    """If `cell` is blocked, returns the nearest (L2) free cell within
    radius_cells, or None if fully enclosed. Bounded search - a large radius
    would mask real problems instead of surfacing them as 'no path'."""
    if not grid.is_blocked_single(cell):
        return cell
    best = None
    best_d2 = None
    r = int(radius_cells)
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            if dx == 0 and dy == 0:
                continue
            d2 = dx * dx + dy * dy
            if d2 > r * r:
                continue
            cand = (cell[0] + dx, cell[1] + dy)
            if grid.is_blocked_single(cand):
                continue
            if best_d2 is None or d2 < best_d2:
                best = cand
                best_d2 = d2
    return best


def _heuristic(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _reconstruct_and_simplify(parent, goal_cell, grid):
    """Walks the parent chain back, then runs one greedy line-of-sight
    simplification pass. Theta*'s parent chain is already any-angle, so this
    mostly removes redundant collinear midpoints - cheap (paths are tens of
    cells at most)."""
    raw = []
    node = goal_cell
    while parent[node] != node:
        raw.append(node)
        node = parent[node]
    raw.append(node)
    raw.reverse()

    if len(raw) <= 2:
        return raw
    simplified = [raw[0]]
    i = 0
    while i < len(raw) - 1:
        j = len(raw) - 1
        while j > i + 1 and not line_of_sight_2d(grid, raw[i], raw[j]):
            j -= 1
        simplified.append(raw[j])
        i = j
    return simplified


def theta_star(grid, start_xy_world, goal_xy_world, max_iterations,
               start_snap_radius_cells=3, goal_snap_radius_m=0.4):
    """2D Theta* (Nash et al. 2007, 8-connected).

    Returns a list of (x, y) world-frame waypoints, or None if no path.
    Z is deliberately NOT this module's concern - the calling node annotates
    waypoints with the goal's z.

    goal_snap_radius_m MUST stay strictly below nbv_planner's goal_tolerance
    (0.5): if the snapped path end sat farther than that from the true NBV
    target, NBV would never detect arrival and would hold the same target
    forever - a subtle deadlock verified by analysis of the 3D version.
    """
    resolution = grid.resolution
    start = world_to_cell(start_xy_world, resolution)
    goal = world_to_cell(goal_xy_world, resolution)

    if grid.is_blocked_single(start):
        # Transient: the drone's own cell can read blocked right after an
        # inflation update. Small bounded escape, else hard failure.
        start = nearest_free_cell(grid, start, start_snap_radius_cells)
        if start is None:
            return None

    if grid.is_blocked_single(goal):
        # Frontier goals legitimately land inside inflation near walls -
        # exactly the scenario this planner exists for. Snap out, bounded.
        goal_snap_cells = max(1, int(round(goal_snap_radius_m / resolution)))
        goal = nearest_free_cell(grid, goal, goal_snap_cells)
        if goal is None:
            return None  # genuinely enclosed/unreachable

    g_score = {start: 0.0}
    parent = {start: start}
    closed = set()

    counter = 0
    open_heap = [(_heuristic(start, goal) * resolution, counter, start)]

    iterations = 0
    while open_heap and iterations < max_iterations:
        iterations += 1
        _, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            cells = _reconstruct_and_simplify(parent, current, grid)
            return [cell_to_world(c, resolution) for c in cells]
        closed.add(current)

        par = parent[current]
        for offset in NEIGHBOR_OFFSETS_8:
            neighbor = (current[0] + int(offset[0]),
                        current[1] + int(offset[1]))
            if neighbor in closed:
                continue
            if grid.is_blocked_single(neighbor):
                continue

            # --- Nash et al. path-2 / path-1 rule (the Theta* core) ---
            # If the neighbor is visible from current's PARENT, connect it
            # directly to the parent (any-angle shortcut); otherwise fall
            # back to the standard grid edge from current.
            if line_of_sight_2d(grid, par, neighbor):
                tentative_g = g_score[par] + _heuristic(par, neighbor) * resolution
                cand_parent = par
            else:
                tentative_g = g_score[current] + _heuristic(current, neighbor) * resolution
                cand_parent = current

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                parent[neighbor] = cand_parent
                counter += 1
                heapq.heappush(open_heap, (
                    tentative_g + _heuristic(neighbor, goal) * resolution,
                    counter, neighbor))

    return None  # no path / iteration cap hit (both feed the same fail path)
