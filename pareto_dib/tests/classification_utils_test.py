import unittest

import numpy as np

from classification_util import merge_joint_sym


class ClassificationUtilTests(unittest.TestCase):
    def setUp(self):
        pass

    def test_merge_joint_sym(self):
        cmap = {0: {0, 2}, 1: {1, 3}}
        pxy = np.zeros((4, 4, 2))
        pxy[:, :, 0] = np.array([[1, 2, 3, 4],
                                 [2, 5, 6, 7],
                                 [3, 6, 8, 9],
                                 [4, 7, 9, 0]])
        pxy[:, :, 1] = np.array([[3, 8, 7, 6],
                                 [8, 4, 5, 9],
                                 [7, 5, 2, 1],
                                 [6, 9, 1, 0]])
        pxy /= pxy.sum()

        pxy_ans = np.zeros((2, 2, 2))
        pxy_ans[:, :, 0] = np.array([[15, 21],
                                     [21, 19]])
        pxy_ans[:, :, 1] = np.array([[19, 20],
                                     [20, 22]])
        pxy_ans /= pxy_ans.sum()

        pxy_mer = merge_joint_sym(pxy, cmap)

        self.assertEqual(pxy_ans.shape, pxy_mer.shape)
        np.testing.assert_array_almost_equal(pxy_ans, pxy_mer)


if __name__ == "__main__":
    unittest.main()
