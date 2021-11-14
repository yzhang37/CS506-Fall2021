from functools import reduce
from collections import defaultdict
from math import inf
import random
import csv

from cs506 import sim


def point_avg(points):
    """
    Accepts a list of points, each with the same number of dimensions.
    (points can have more dimensions than 2)
    
    Returns a new point which is the center of all the points.
    """
    assert(len(points) > 0)
    size = len(points)
    sums = reduce(lambda a, b: [i + j for i, j in zip(a, b)], points)
    return [i / size for i in sums]


def update_centers(dataset, assignments):
    """
    Accepts a dataset and a list of assignments; the indexes 
    of both lists correspond to each other.
    Compute the center for each of the assigned groups.
    Return `k` centers in a list
    """
    labels = sorted(set(assignments))

    cluster_groups = dict()

    for data, assignment in zip(dataset, assignments):
        cluster_groups.setdefault(assignment, [])
        cluster_groups[assignment].append(data)

    centroids = [point_avg(cluster_groups[lbl]) for lbl in labels]
    return centroids


def assign_points(data_points, centers):
    """
    """
    assignments = []
    for point in data_points:
        shortest = inf  # positive infinity
        shortest_index = 0
        for i in range(len(centers)):
            val = distance(point, centers[i])
            if val < shortest:
                shortest = val
                shortest_index = i
        assignments.append(shortest_index)
    return assignments


def distance(a, b):
    """
    Returns the Euclidean distance between a and b
    """
    return sim.euclidean_dist(a, b)


def generate_k(dataset, k):
    """
    Given `data_set`, which is an array of arrays,
    return a random set of k points from the data_set
    """
    return random.sample(dataset, k)


def cost_function(clustering: defaultdict):
    all_sum = 0.0
    for cluster in clustering.values():
        centroid = point_avg(cluster)
        dist_vect = [distance(centroid, point) for point in cluster]
        all_sum += sum(dist_vect)
    return all_sum


def generate_k_pp(dataset, k):
    """
    Given `data_set`, which is an array of arrays,
    return a random set of k points from the data_set
    where points are picked with a probability proportional
    to their distance as per kmeans pp
    """

    # 1. Random first center point
    num_dataset = len(dataset)
    first_idx = random.choice(range(num_dataset))
    centers = [dataset[first_idx]]

    # 2. Calculate the distance between each sample and each cluster center and save the smallest distance
    dist_note = [float("inf")] * num_dataset

    for j in range(k - 1):
        # Calculate the distance between each sample and each cluster center and save the smallest distance
        for i in range(num_dataset):
            dist = distance(centers[j], dataset[i])
            if dist < dist_note[i]:
                dist_note[i] = dist

        max_val = max(dist_note)
        next_idx = dist_note.index(max_val)
        centers.append(dataset[next_idx])

    return centers


def _do_lloyds_algo(dataset, k_points):
    assignments = assign_points(dataset, k_points)
    old_assignments = None
    while assignments != old_assignments:
        new_centers = update_centers(dataset, assignments)
        old_assignments = assignments
        assignments = assign_points(dataset, new_centers)
    clustering = defaultdict(list)
    for assignment, point in zip(assignments, dataset):
        clustering[assignment].append(point)
    return clustering


def k_means(dataset, k):
    if k not in range(1, len(dataset)+1):
        raise ValueError("lengths must be in [1, len(dataset)]")
    
    k_points = generate_k(dataset, k)
    return _do_lloyds_algo(dataset, k_points)


def k_means_pp(dataset, k):
    if k not in range(1, len(dataset)+1):
        raise ValueError("lengths must be in [1, len(dataset)]")

    k_points = generate_k_pp(dataset, k)
    return _do_lloyds_algo(dataset, k_points)
