"""Waypoint pattern generation.

Kept free of ROS imports so the shapes can be unit-tested (and reasoned about)
without spinning up a node.
"""

import math

# A 5-pointed star is a *pentagram*: five points evenly spaced around a circle
# (360/5 = 72 deg apart), but visited in an order that skips every other point.
# Skipping one point each step means advancing 2 * 72 = 144 deg per move, and
# because 2 and 5 share no common factor the path visits all five points before
# closing - that is what draws the crossing lines of a star.
#
# Walking 72 deg per step instead would visit the same five points in
# neighbour order and just draw a pentagon.
_STAR_POINTS = 5
_STAR_STEP_DEG = 144.0

# Start at +90 deg so the star has a point facing "up" (+Y in the map frame),
# which is what a person expects to see in RViz.
_STAR_START_DEG = 90.0


def star_waypoints(center_x, center_y, radius, altitude, close_loop=True):
    """Return the pentagram vertices as a list of (x, y, z) tuples.

    ``close_loop`` repeats the first vertex at the end so the drone flies the
    final edge and completes the shape, rather than stopping on the fifth point
    with one line missing.
    """
    if radius <= 0.0:
        raise ValueError(f'radius must be positive, got {radius}')

    points = []
    for i in range(_STAR_POINTS):
        angle = math.radians(_STAR_START_DEG + i * _STAR_STEP_DEG)
        points.append((
            center_x + radius * math.cos(angle),
            center_y + radius * math.sin(angle),
            altitude,
        ))

    if close_loop:
        points.append(points[0])
    return points


def distance_2d(ax, ay, bx, by):
    """Planar distance. Altitude is excluded deliberately: arrival at a
    waypoint is judged in XY because the controller holds Z separately."""
    return math.hypot(ax - bx, ay - by)


def yaw_from_quaternion(x, y, z, w):
    """Extract yaw (rotation about Z) from a quaternion, in radians."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)
