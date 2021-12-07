from .sim import euclidean_dist
import typing
from numpy import ndarray


class DBC:
    def __init__(self, dataset: ndarray, min_pts: int, epsilon: float):
        self.dataset = dataset
        self.min_pts = min_pts
        self.epsilon = epsilon

    def _get_epsilon_neighbor(self, pt: int):
        neighbor = []
        for point in range(len(self.dataset)):
            if euclidean_dist(self.dataset[point], self.dataset[pt]) <= self.epsilon:
                neighbor.append(point)
        return neighbor

    def _explore_pt_neighbor(self, pt: int, pt_neighbor: typing.List[int],
                             assignments: typing.List[int],
                             assignment: int) -> typing.List[int]:
        assignments[pt] = assignment

        while pt_neighbor:
            next_pt = pt_neighbor.pop()

            if assignments[next_pt] == -1:
                # now we know it's a border point
                assignments[next_pt] = assignment

            if assignments[next_pt] == 0:
                # haven't seen this point before
                assignments[next_pt] = assignment
                next_pt_neighbor = self._get_epsilon_neighbor(next_pt)

                # this is the core point
                if len(next_pt_neighbor) >= self.min_pts:
                    pt_neighbor += next_pt_neighbor

        return assignments

    def dbscan(self) -> typing.List[int]:
        assignments = [0] * len(self.dataset)
        assignment = 0

        for P in range(len(self.dataset)):
            if assignments[P] != 0:
                continue

            pt_neighbor = self._get_epsilon_neighbor(P)

            if len(pt_neighbor) >= self.min_pts:
                # this is the core point!
                assignment += 1
                assignments = self._explore_pt_neighbor(P, pt_neighbor, assignments, assignment)
            else:
                # this one could be either border or noise
                assignments[P] = -1

        return assignments
