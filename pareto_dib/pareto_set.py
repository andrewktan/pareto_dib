import numpy as np
from sortedcontainers import SortedKeyList


class ParetoSet(SortedKeyList):
    """Maintained maximal set with efficient insertion."""

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, key=lambda x: x[0])

    def __init__(self, tol=1e-8):
        super().__init__()

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
                np.abs(self[idx + 1][0] - p[0]) < self.tol:
            idx += 1
        while idx >= 0 and self[idx][1] - p[1] < self.tol:
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

        if idx == len(self):
            return True
        else:
            return p[1] - self[idx][1] > self.tol

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
        """Given a tuple, calculate the minimum Euclidean distance to Pareto
        frontier (in first two indices).

        Args:
            p (tuple): point

        Returns:
            float: minimum Euclidean distance to pareto frontier

        """
        point = np.array(p[0:2])
        # distance is zero if pareto optimal
        if self.is_pareto(p):
            return 0.

        # compare with next point
        idx = self.bisect_left(p)
        min_dist = self[idx][1] - point[1]

        while self[idx][0] - point[0] < min_dist:
            if idx + 1 < len(self) and self[idx][1] > point[1]:
                corner = np.array([self[idx][0], self[idx + 1][1]])
                dist = np.sqrt(np.sum(np.square(point - corner)))
                min_dist = np.minimum(dist, min_dist)
            else:
                dist = self[idx][0] - point[0]
                min_dist = np.minimum(dist, min_dist)
                break

            idx += 1

        return min_dist

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
