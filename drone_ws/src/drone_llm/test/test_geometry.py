import math

import pytest

from drone_llm.geometry import distance_2d, star_waypoints, yaw_from_quaternion


def test_star_has_five_distinct_vertices_and_closes():
    pts = star_waypoints(0.0, 0.0, 2.0, 1.0)
    assert len(pts) == 6                 # 5 vertices + closing repeat
    assert pts[0] == pts[-1]
    assert len({(round(x, 6), round(y, 6)) for x, y, _ in pts[:5]}) == 5


def test_vertices_lie_on_the_requested_circle():
    cx, cy, r = 3.0, -1.5, 2.5
    for x, y, _ in star_waypoints(cx, cy, r, 1.0)[:5]:
        assert math.hypot(x - cx, y - cy) == pytest.approx(r, abs=1e-9)


def test_traversal_is_a_star_not_a_pentagon():
    """The defining property: consecutive vertices are chord-length apart for
    a 144-degree step, which is strictly longer than a pentagon's 72-degree
    edge on the same circle. This is what makes the lines cross."""
    r = 2.0
    pts = star_waypoints(0.0, 0.0, r, 1.0)
    star_edge = distance_2d(*pts[0][:2], *pts[1][:2])
    pentagon_edge = 2 * r * math.sin(math.radians(36))
    assert star_edge == pytest.approx(2 * r * math.sin(math.radians(72)), abs=1e-9)
    assert star_edge > pentagon_edge


def test_altitude_is_constant_and_centre_is_respected():
    pts = star_waypoints(1.0, 2.0, 1.0, 0.8)
    assert all(z == pytest.approx(0.8) for _, _, z in pts)
    mean_x = sum(x for x, _, _ in pts[:5]) / 5
    mean_y = sum(y for _, y, _ in pts[:5]) / 5
    # Vertices are evenly spaced on the circle, so their centroid is the centre.
    assert mean_x == pytest.approx(1.0, abs=1e-9)
    assert mean_y == pytest.approx(2.0, abs=1e-9)


def test_non_positive_radius_is_rejected():
    with pytest.raises(ValueError):
        star_waypoints(0.0, 0.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        star_waypoints(0.0, 0.0, -1.0, 1.0)


def test_yaw_from_quaternion_identity_and_quarter_turn():
    assert yaw_from_quaternion(0.0, 0.0, 0.0, 1.0) == pytest.approx(0.0)
    h = math.sqrt(0.5)
    assert yaw_from_quaternion(0.0, 0.0, h, h) == pytest.approx(math.pi / 2)
