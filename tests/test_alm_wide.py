"""Checks on the reconstruction of the light from continuous traces.

The two larger releases do not say per trial whether the light was on, so the loader
infers it from onset events, a laser trace per site and the task epoch times. That
inference is the part most likely to be silently wrong, and wrong in a way that would
still produce plausible looking numbers, so it is worth testing on its own.
"""

from __future__ import annotations

import numpy as np

from cadence.data.alm_wide import _assign_to_trials


def test_events_land_in_the_right_trial():
    starts = np.array([0.0, 10.0, 20.0, 30.0])
    stops = np.array([5.0, 15.0, 25.0, 35.0])
    ev = np.array([1.0, 12.0, 24.9, 30.0])
    assert list(_assign_to_trials(ev, starts, stops)) == [0, 1, 2, 3]


def test_events_in_the_gaps_belong_to_nobody():
    starts = np.array([0.0, 10.0])
    stops = np.array([5.0, 15.0])
    ev = np.array([-1.0, 7.0, 16.0])
    assert list(_assign_to_trials(ev, starts, stops)) == [-1, -1, -1]


def test_trial_order_does_not_matter():
    """Trials are not guaranteed to be sorted, and an unsorted table must not shift
    every event by one."""
    starts = np.array([20.0, 0.0, 10.0])
    stops = np.array([25.0, 5.0, 15.0])
    ev = np.array([21.0, 1.0, 11.0])
    assert list(_assign_to_trials(ev, starts, stops)) == [0, 1, 2]


def test_boundaries_are_inclusive_at_both_ends():
    starts = np.array([0.0, 10.0])
    stops = np.array([5.0, 15.0])
    assert list(_assign_to_trials(np.array([0.0, 5.0, 10.0, 15.0]), starts, stops)) \
        == [0, 0, 1, 1]


def test_empty_event_list_is_handled():
    out = _assign_to_trials(np.array([]), np.array([0.0]), np.array([1.0]))
    assert len(out) == 0
