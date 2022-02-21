import unittest

import numpy as np

from pareto_dib.pareto_set import ParetoSet


class PareoSetTests(unittest.TestCase):
    def setUp(self):
        self.PA = ParetoSet(tol=1e-8)

        self.PA.add((0.2, 0.1))
        self.PA.add((0., 1.))  # pareto optimal
        self.PA.add((0.25, 0.9))  # pareto optimal
        self.PA.add((0.5, 0.5))  # pareto optimal
        self.PA.add((0.75, 0.1))  # pareto optimal
        self.PA.add((1., 0.))  # pareto optimal
        self.PA.add((0., 0.))

    def test_edgecases(self):
        P = ParetoSet(tol=1e-8)

        self.assertTrue(P.add((0., 1.)))
        self.assertEqual(len(P), 1)
        self.assertTrue(P.is_pareto((0.5, 1.)))
        self.assertTrue(P.add((0.5, 1.)))
        self.assertEqual(len(P), 1)
        self.assertTrue(P.is_pareto((0.5, 1.)))
        self.assertFalse(P.add((0.5, 1.)))
        self.assertTrue(P.is_pareto((0.5, 1.)))
        self.assertTrue(P.add((0.5, 1.1)))
        self.assertEqual(len(P), 1)
        self.assertFalse(P.add((0.5, 1.)))
        self.assertFalse(P.add((0.5, 0.9)))
        self.assertEqual(len(P), 1)

    def test_ispareto(self):
        self.assertTrue(self.PA.is_pareto((0., 1.)))
        self.assertTrue(self.PA.is_pareto((0.25, 0.9)))
        self.assertTrue(self.PA.is_pareto((0.5, 0.5)))
        self.assertTrue(self.PA.is_pareto((0.75, .1)))
        self.assertTrue(self.PA.is_pareto((1., 0.)))

        self.assertTrue(self.PA.is_pareto((0.125, 0.95)))

    def test_length(self):
        self.assertEqual(len(self.PA), 5)

    def test_distance(self):
        self.assertAlmostEqual(self.PA.distance((0.5, 0.5)), 0.)
        self.assertAlmostEqual(self.PA.distance((2, 0)), 0.)
        self.assertAlmostEqual(self.PA.distance((2, 2)), 0.)
        self.assertAlmostEqual(self.PA.distance((0, 2)), 0.)
        self.assertAlmostEqual(self.PA.distance((0.9, 0.)), 0.)
        self.assertAlmostEqual(self.PA.distance((0.125, 0.95)), 0.)
        self.assertAlmostEqual(self.PA.distance((0.9, -1)), 0.1)
        self.assertAlmostEqual(self.PA.distance((-1, 0)), 1.)
        self.assertAlmostEqual(self.PA.distance((0.25, 0.5)), 0.)
        self.assertAlmostEqual(self.PA.distance((1., -0.5)), 0.)
        self.assertAlmostEqual(
            self.PA.distance(
                (0.2, 0.45)), 0.05 * np.sqrt(2))
        self.assertAlmostEqual(self.PA.distance((0, 0)),
                               np.sqrt(0.1**2 + 0.5**2))

        self.assertAlmostEqual(self.PA.distance((0.75 + 1e-4, .1 + 1e-4)),
                               0.)
        self.assertAlmostEqual(self.PA.distance((0.75 + 1e-4, .1 - 1e-4)),
                               0.)
        self.assertAlmostEqual(self.PA.distance((0.75 - 1e-4, .1 + 1e-4)),
                               0.)
        self.assertAlmostEqual(self.PA.distance((0.75 - 1e-4, .1 - 1e-4)),
                               1e-4)

    def test_contains(self):
        self.assertTrue((0., 1.) in self.PA)
        self.assertTrue((0.25, 0.9) in self.PA)
        self.assertTrue((0.5, 0.5) in self.PA)
        self.assertTrue((0.75, .1) in self.PA)
        self.assertTrue((1., 0.) in self.PA)

        self.assertFalse((1., 1.) in self.PA)
        self.assertFalse((1., 0.001) in self.PA)
        self.assertFalse((0., 0.) in self.PA)

        self.assertTrue((0.75 + 1e-10, .1 + 1e-10) in self.PA)
        self.assertTrue((0.75 + 1e-10, .1 - 1e-10) in self.PA)
        self.assertTrue((0.75 - 1e-10, .1 + 1e-10) in self.PA)
        self.assertTrue((0.75 - 1e-10, .1 - 1e-10) in self.PA)

        self.assertFalse((0.75 + 1e-6, .1 + 1e-6) in self.PA)
        self.assertFalse((0.75 + 1e-6, .1 - 1e-6) in self.PA)
        self.assertFalse((0.75 - 1e-6, .1 + 1e-6) in self.PA)
        self.assertFalse((0.75 - 1e-6, .1 - 1e-6) in self.PA)


if __name__ == "__main__":
    unittest.main()
