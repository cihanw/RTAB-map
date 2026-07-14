#!/usr/bin/env python3
"""3D Theta* grid search - pure algorithm module (NO rclpy, NOT a ROS node).

Imported by theta_star_planner.py. Can be tested independently (with plain numpy
arrays, without ROS runtime).
"""
import heapq

import numpy as np

# Exact repetition of nbv_planner.py's _voxel_keys schema (see
# tasks/lessons.md - it was the fix for the CPU bottleneck where using Python
# set/tuple with 1.4M points locked a single core. To avoid repeating the
# same mistake, the same vectorized approach is used here as well).
VOXEL_BITS = 21
VOXEL_OFFSET = 1 << (VOXEL_BITS - 1)
VOXEL_MASK = (1 << VOXEL_BITS) - 1


def voxel_keys(voxel_coords):
    """Packs (N,3) int voxel coordinates into sortable (N,) int64 keys."""
    x = (voxel_coords[:, 0].astype(np.int64) + VOXEL_OFFSET) & VOXEL_MASK
    y = (voxel_coords[:, 1].astype(np.int64) + VOXEL_OFFSET) & VOXEL_MASK
    z = (voxel_coords[:, 2].astype(np.int64) + VOXEL_OFFSET) & VOXEL_MASK
    return (x << (2 * VOXEL_BITS)) | (y << VOXEL_BITS) | z


def _unpack_voxel_keys(keys):
    z = (keys & VOXEL_MASK) - VOXEL_OFFSET
    y = ((keys >> VOXEL_BITS) & VOXEL_MASK) - VOXEL_OFFSET
    x = ((keys >> (2 * VOXEL_BITS)) & VOXEL_MASK) - VOXEL_OFFSET
    return np.stack([x, y, z], axis=-1)


# 26-connected neighborhood: {-1,0,1}^3 \ {0,0,0}
NEIGHBOR_OFFSETS_26 = np.array(
    [(dx, dy, dz)
     for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
     if not (dx == 0 and dy == 0 and dz == 0)],
    dtype=np.int64)
NEIGHBOR_COST_26 = np.linalg.norm(NEIGHBOR_OFFSETS_26.astype(np.float64), axis=1)


class OccupancyGrid3D:
    """Sparse (keyed) 3D obstacle grid.

    Constructed ONLY from the obstacle cloud - free/unknown data
    is not consumed at all. This naturally satisfies the "unknown cell = free"
    rule: a cell is either in _blocked_keys (obstacle + inflation) or
    it is not (free - whether it's known free or unknown doesn't matter).
    """

    def __init__(self, resolution, inflation_radius_m, z_min, z_max):
        self.resolution = float(resolution)
        self.inflation_radius_cells = max(1, int(round(inflation_radius_m / resolution)))
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self._blocked_keys_sorted = np.zeros((0,), dtype=np.int64)
        self._blocked_keys_set = set()

        r = self.inflation_radius_cells
        offsets_1d = np.arange(-r, r + 1)
        ox, oy, oz = np.meshgrid(offsets_1d, offsets_1d, offsets_1d, indexing='ij')
        # Spherical (L2) mask - consistent with local_planner's distance-based
        # repulsive force semantics (to avoid unnecessarily inflating the corners of the cube).
        mask = (ox ** 2 + oy ** 2 + oz ** 2) <= r * r
        self._inflation_offsets = np.stack(
            [ox[mask], oy[mask], oz[mask]], axis=-1).astype(np.int64)

    def update_from_points(self, obstacle_points_xyz):
        """obstacle_points_xyz: (N,3) float64 world-frame points."""
        if obstacle_points_xyz.shape[0] == 0:
            self._blocked_keys_sorted = np.zeros((0,), dtype=np.int64)
            self._blocked_keys_set = set()
            return
        int_coords = np.round(obstacle_points_xyz / self.resolution).astype(np.int64)
        raw_keys = np.unique(voxel_keys(int_coords))
        raw_coords = _unpack_voxel_keys(raw_keys)

        # Vectorized spherical inflation: add the precomputed sphere-offset table
        # to each obstacle voxel, deduplicate. NO per-point Python loop.
        inflated = raw_coords[:, None, :] + self._inflation_offsets[None, :, :]
        flat = inflated.reshape(-1, 3)
        inflated_keys = np.unique(voxel_keys(flat))

        self._blocked_keys_sorted = inflated_keys
        self._blocked_keys_set = set(inflated_keys.tolist())

    def is_blocked(self, voxel_coords):
        """(N,3) int -> (N,) bool, vectorized searchsorted membership test (batch)."""
        if self._blocked_keys_sorted.shape[0] == 0:
            return np.zeros(voxel_coords.shape[0], dtype=bool)
        keys = voxel_keys(voxel_coords)
        idx = np.searchsorted(self._blocked_keys_sorted, keys)
        idx = np.clip(idx, 0, len(self._blocked_keys_sorted) - 1)
        return self._blocked_keys_sorted[idx] == keys

    def is_blocked_single(self, vox):
        """Single cell, Python set lookup (for line_of_sight_3d/search inner loops
        - faster than numpy searchsorted for N=1 case; the CONSCIOUS single
        exception to the "vectorize, don't use set" rule, see module comment)."""
        key = _pack_single(vox)
        return key in self._blocked_keys_set

    def in_bounds(self, vox):
        """ONLY z is bounded. X/Y are deliberately UNBOUNDED (grid is sparse/
        keyed - world is open-ended); search cost is limited by max_iterations,
        since the heuristic already guides towards the target, it doesn't
        pointlessly stray far away in open space."""
        z_world = vox[2] * self.resolution
        return self.z_min <= z_world <= self.z_max


def _pack_single(vox):
    x = (int(vox[0]) + VOXEL_OFFSET) & VOXEL_MASK
    y = (int(vox[1]) + VOXEL_OFFSET) & VOXEL_MASK
    z = (int(vox[2]) + VOXEL_OFFSET) & VOXEL_MASK
    return (x << (2 * VOXEL_BITS)) | (y << VOXEL_BITS) | z


def world_to_voxel(point_xyz, resolution):
    return tuple(int(v) for v in np.round(np.asarray(point_xyz) / resolution))


def voxel_to_world(vox, resolution):
    # CONSISTENT with nbv_planner.py's own rule: NO half-cell offset
    # (vox * voxel_size) - to prevent a spurious residual gap between the
    # Theta* endpoint and the original NBV target.
    return np.array(vox, dtype=np.float64) * resolution


def _sign(v):
    return (v > 0) - (v < 0)


def line_of_sight_3d(grid, a_vox, b_vox):
    """3D Bresenham/DDA voxel traversal - verifies that the straight line
    between two cells does not pass through any blocked voxel. Integer arithmetic,
    no floating point, checks EVERY passed voxel, exits early on the first
    blocked voxel. This is the core shortcut (any-angle) that distinguishes Theta* from A*."""
    x0, y0, z0 = a_vox
    x1, y1, z1 = b_vox

    dx = x1 - x0
    dy = y1 - y0
    dz = z1 - z0
    step_x, step_y, step_z = _sign(dx), _sign(dy), _sign(dz)
    adx, ady, adz = abs(dx), abs(dy), abs(dz)

    x, y, z = x0, y0, z0

    if adx == 0 and ady == 0 and adz == 0:
        return True

    ax, ay, az = 2 * adx, 2 * ady, 2 * adz

    if ax >= ay and ax >= az:
        yd = ay - adx
        zd = az - adx
        for _ in range(adx):
            if grid.is_blocked_single((x, y, z)):
                return False
            if yd >= 0:
                y += step_y
                yd -= ax
            if zd >= 0:
                z += step_z
                zd -= ax
            yd += ay
            zd += az
            x += step_x
    elif ay >= ax and ay >= az:
        xd = ax - ady
        zd = az - ady
        for _ in range(ady):
            if grid.is_blocked_single((x, y, z)):
                return False
            if xd >= 0:
                x += step_x
                xd -= ay
            if zd >= 0:
                z += step_z
                zd -= ay
            xd += ax
            zd += az
            y += step_y
    else:
        xd = ax - adz
        yd = ay - adz
        for _ in range(adz):
            if grid.is_blocked_single((x, y, z)):
                return False
            if xd >= 0:
                x += step_x
                xd -= az
            if yd >= 0:
                y += step_y
                yd -= az
            xd += ax
            yd += ay
            z += step_z

    return not grid.is_blocked_single((x1, y1, z1))


def _nearest_free_cell(grid, vox, radius):
    """If vox is blocked, searches for the nearest free cell (by Euclidean distance)
    within a radius of 'radius' cells. Returns None if not found."""
    if not grid.is_blocked_single(vox):
        return vox
    best = None
    best_dist2 = None
    r = int(radius)
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            for dz in range(-r, r + 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                cand = (vox[0] + dx, vox[1] + dy, vox[2] + dz)
                d2 = dx * dx + dy * dy + dz * dz
                if d2 > r * r:
                    continue
                if grid.is_blocked_single(cand):
                    continue
                if best_dist2 is None or d2 < best_dist2:
                    best = cand
                    best_dist2 = d2
    return best


def _heuristic(a, b):
    return float(np.linalg.norm(np.array(a, dtype=np.float64) - np.array(b, dtype=np.float64)))


def _reconstruct_and_simplify(parent, goal_vox, grid):
    raw = []
    node = goal_vox
    while parent[node] != node:
        raw.append(node)
        node = parent[node]
    raw.append(node)
    raw.reverse()

    # Theta*'s parent-pointer chain is already any-angle-shortcutted (this is
    # its main advantage over a separate path-smoothing pass after A*) - raw is usually
    # already close to minimal. Still, for safety/cleanliness, a light greedy
    # simplification pass: try to connect waypoint i to the FARTHEST waypoint
    # j>i with clear line of sight, skipping those in between.
    if len(raw) <= 2:
        return raw
    simplified = [raw[0]]
    i = 0
    while i < len(raw) - 1:
        j = len(raw) - 1
        while j > i + 1 and not line_of_sight_3d(grid, raw[i], raw[j]):
            j -= 1
        simplified.append(raw[j])
        i = j
    return simplified


def theta_star(grid, start_world, goal_world, max_iterations,
               start_snap_radius_cells=3, goal_snap_radius_m=0.4):
    """3D Theta* search (Nash et al. 2007, 26-connected generalization).

    Returns: world-frame (x,y,z) waypoint list, or None if no path is found.
    """
    resolution = grid.resolution
    start_vox = world_to_voxel(start_world, resolution)
    goal_vox = world_to_voxel(goal_world, resolution)

    if grid.is_blocked_single(start_vox):
        snapped = _nearest_free_cell(grid, start_vox, start_snap_radius_cells)
        if snapped is None:
            return None
        start_vox = snapped

    if grid.is_blocked_single(goal_vox):
        # Frontier targets can be close to obstacles (this is exactly the scenario
        # this feature solves) - "snap" to the nearest free cell.
        goal_snap_radius_cells = max(1, int(round(goal_snap_radius_m / resolution)))
        snapped = _nearest_free_cell(grid, goal_vox, goal_snap_radius_cells)
        if snapped is None:
            return None  # truly enclosed/unreachable
        goal_vox = snapped

    g_score = {start_vox: 0.0}
    parent = {start_vox: start_vox}
    closed = set()

    counter = 0
    open_heap = [(_heuristic(start_vox, goal_vox) * resolution, counter, start_vox)]

    iterations = 0
    while open_heap and iterations < max_iterations:
        iterations += 1
        _, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal_vox:
            return [voxel_to_world(v, resolution) for v in
                    _reconstruct_and_simplify(parent, current, grid)]
        closed.add(current)

        par = parent[current]
        for offset in NEIGHBOR_OFFSETS_26:
            neighbor = (current[0] + int(offset[0]),
                        current[1] + int(offset[1]),
                        current[2] + int(offset[2]))
            if neighbor in closed:
                continue
            if not grid.in_bounds(neighbor):
                continue
            if grid.is_blocked_single(neighbor):
                continue

            if line_of_sight_3d(grid, par, neighbor):
                # Path 2 (any-angle): skip current and connect directly from parent.
                tentative_g = g_score[par] + _heuristic(par, neighbor) * resolution
                candidate_parent = par
            else:
                # Path 1 (standard A* edge): connect to current.
                tentative_g = g_score[current] + _heuristic(current, neighbor) * resolution
                candidate_parent = current

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                parent[neighbor] = candidate_parent
                h = _heuristic(neighbor, goal_vox) * resolution
                counter += 1
                heapq.heappush(open_heap, (tentative_g + h, counter, neighbor))

    return None  # path not found / iteration limit exceeded
