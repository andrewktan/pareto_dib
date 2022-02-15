import numpy as np
from sortedcontainers import SortedKeyList


class ParetoSet(SortedKeyList):
    """Maintained maximal set with efficient insertion."""

    def __init__(self, tol=1e-8):
        super().__init__(key=lambda x: x[0])

        self.tol = tol

    def add(self, p):
        """Insert tuple into set if maximal in first two indices.

        Args:
            p (tuple): tuple to insert

        Returns:
            bool: True only if point is inserted

        """
        if len(self) == 0:
            super().add(p)
            return True

        if not self.is_pareto(p) or p in self:
            return False

        # remove dominated points on the left
        idx = self.bisect_left(p) - 1
        while idx + 1 < len(self) and \
                np.abs(self[idx + 1][0] - p[0]) < 1e-8:
            idx += 1
        while idx >= 0 and self[idx][1] - p[1] < 1e-8:
            self.pop(idx)
            idx -= 1

        super().add(p)
        return True

    def is_pareto(self, p):
        """Check if tuple is pareto maximal in first two indices.

        Args:
            p (tuple): tuple to insert

        Returns:
            bool: True if point is pareto maximal

        """
        # if it's the only point or it's in the set, it's maximal
        if len(self) == 0 or p in self:
            return True

        # check right for dominating points
        idx = self.bisect_left(p)

        while idx - 1 >= 0 and \
                np.abs(self[idx - 1][0] - p[0]) < 1e-8:
            idx -= 1

        if idx == len(self):
            return True
        else:
            return p[1] - self[idx][1] > 1e-8

    def __contains__(self, p):
        p_test = (p[0] - self.tol, None)
        left = self.bisect_left(p_test)

        while left < len(self) and np.abs(self[left][0] - p[0]) < self.tol:
            if np.abs(self[left][1] - p[1]) < self.tol:
                return True

            left += 1

        return False

    def __add__(self, other):
        """Merge another pareto set into self.

        Args:
            other (ParetoSet): set to merge into self

        Returns:
            ParetoSet: self

        """

        for item in other:
            self.add(item)

        return self

    def distance(self, p):
        """Given a tuple, calculate the minimum Euclidean distance to pareto
        frontier (in first two indices).

        Args:
            p (tuple): point

        Returns:
            float: minimum Euclidean distance to pareto frontier

        """
        point = np.array(p[0:2])
        dom = self.dominant_array(p)

        # distance is zero if pareto optimal
        if dom.shape[0] == 0:
            return 0.

        # add corners of all adjacent pairs
        candidates = np.zeros((dom.shape[0] + 1, 2))
        for i in range(dom.shape[0] - 1):
            candidates[i, :] = np.min(dom[[i, i + 1], :], axis=0)

        # add top and right bounds
        candidates[-1, :] = (p[0], np.max(dom[:, 1]))
        candidates[-2, :] = (np.max(dom[:, 0]), p[1])

        return np.min(np.sqrt(np.sum(np.square(candidates - point), axis=1)))

    def dominant_array(self, p):
        """Given a tuple, return the set of dominating points in the set (in
        the first two indices).

        Args:
            p (tuple): point

        Returns:
            numpy.ndarray: array of dominating points

        """
        idx = self.bisect_right(p)

        domlist = []

        while idx < len(self) and self[idx][1] > p[1]:
            domlist.append(self[idx])
            idx += 1

        return np.array([x[0:2] for x in domlist])

    def to_array(self):
        """Convert first two indices to numpy.ndarray

        Args:
            None

        Returns:
            numpy.ndarray: array of shape (len(self), 2)

        """
        A = np.zeros((len(self), 2))
        for i, tup in enumerate(self):
            A[i, :] = tup[0:2]

        return A

    def from_list(self, A):
        """Convert iterable of tuples into ParetoSet.

        Args:
            A (iterator): iterator of tuples

        Returns:
            None

        """
        for a in A:
            self.add(a)

    def to_list(self, idx=None):
        """Converts self or slice of self to list.

        Args:
            idx (int, optional): index to convert to list, if None, entire set
                is output

        Returns:
            list: list specified above

        """
        if idx:
            return [x[idx] for x in self]
        else:
            return list(self)

    def to_set(self, idx=None):
        """Converts self or slice of self to set.

        Args:
            idx (int, optional): index to convert to set, if None, entire set
                is output

        Returns:
            set: set specified above

        """
        if idx:
            return {x[idx] for x in self}
        else:
            return set(self)
