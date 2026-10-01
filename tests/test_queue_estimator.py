"""
Tests for lane-level queue estimation (run with: python -m pytest tests/).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import load_config
from src.queue_estimator import QueueEstimator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def est():
    cfg = load_config(os.path.join(ROOT, "configs", "config.yaml"))
    return QueueEstimator(cfg["video"], fps=25)


def box_at(x, y, tid, w=30, h=20):
    """Box whose bottom-centre (ground point) is at (x, y)."""
    return [x - w / 2, y - h, x + w / 2, y, tid]


def test_each_default_zone_maps_to_its_lane(est):
    for lane, poly in enumerate(est.zones):
        cx, cy = poly.reshape(-1, 2).mean(axis=0)
        assert est.lane_of((cx, cy)) == lane


def test_only_stopped_vehicles_are_queued(est):
    # lane 4 (E straight): y in [150, 212], x in [600, 960]
    for f in range(10):
        tracks = np.array([
            box_at(700, 200, 1),               # parked in the queue
            box_at(900 - 10 * f, 200, 2),      # moving at 10 px/frame = 250 px/s
        ])
        queues, vehicles = est.update(tracks, f)
    assert vehicles[4] == 2
    assert queues[4] == 1
    assert est.stopped_ids_per_lane[4] == {1}


def test_queue_mode_all_counts_every_vehicle():
    cfg = load_config(os.path.join(ROOT, "configs", "config.yaml"))
    est = QueueEstimator({**cfg["video"], "queue_mode": "all"}, fps=25)
    queues, _ = est.update(np.array([box_at(700, 200, 1), box_at(800, 200, 2)]), 0)
    assert queues[4] == 2


def test_vehicle_outside_zones_is_ignored(est):
    queues, vehicles = est.update(np.array([box_at(480, 280, 7)]), 0)   # intersection box
    assert vehicles.sum() == 0 and queues.sum() == 0
