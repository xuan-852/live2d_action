"""Mechanics tests only; naturalness requires actual rendered sequences and user feedback."""
import unittest
import numpy as np
from tools.expression.trajectory import ResponseGrid, fit


class TrajectoryTests(unittest.TestCase):
    def grid(self):
        return ResponseGrid([{'shoulder': s, 'swing': w, 'hand': [s / 60, w / 30]}
                             for s in (-30, 0, 30, 60) for w in (-30, 0, 30)])

    def test_interpolates_measured_nodes(self):
        grid = self.grid()
        q = np.array([[1/3, .5], [2/3, 1]])
        predicted, jac = grid.forward(q)
        np.testing.assert_allclose(predicted, [[0, 0], [.5, 1]], atol=1e-10)
        self.assertTrue(np.all(np.isfinite(jac)))

    def test_temporal_fit_not_frozen_or_out_of_bounds(self):
        grid = self.grid()
        targets = np.column_stack((np.linspace(0, .8, 16), np.zeros(16)))
        angles, report = fit(targets, grid, 1, iterations=12)
        self.assertGreater(np.ptp(angles[:, 0]), 25)
        self.assertLessEqual(np.max(angles[:, 0]), 60)
        self.assertGreaterEqual(np.min(angles[:, 0]), -30)
        self.assertLess(report['objective'][-1], report['objective'][0])

    def test_unreachable_target_keeps_error(self):
        _, report = fit(np.tile([10, 10], (4, 1)), self.grid(), 1, iterations=5)
        self.assertGreater(report['predictedMeanError'], 10)

    def test_nonfinite_targets_rejected(self):
        with self.assertRaises(ValueError):
            fit([[0, 0], [float('nan'), 0], [0, 0]], self.grid(), 1)
