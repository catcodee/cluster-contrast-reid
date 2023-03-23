import torch
import numpy as np
from collections import defaultdict

def compute_cluster_dist2(labels_a, labels_b):
    l2idxes_a = defaultdict(set)
    l2idxes_b = defaultdict(set)
    for idx, l in enumerate(labels_a):
        if not l == -1:
            l2idxes_a[l].add(idx)
    
    for idx, l in enumerate(labels_b):
        if not l == -1:
            l2idxes_b[l].add(idx)
    
    num_labels_a = len(l2idxes_a)
    num_labels_b = len(l2idxes_b)

    similarity = torch.zeros((num_labels_a, num_labels_b))

    for i in range(num_labels_a):
        set_a = l2idxes_a[i]
        for j in range(num_labels_b):
            set_b = l2idxes_b[j]
            intersection = len(set_a & set_b)
            union = len(set_a | set_b)
            similarity[i][j] = intersection*1.0 / union

    return similarity

def compute_cluster_dist(labels_a, labels_b, num_labels_a, num_labels_b):

    dist = torch.zeros(num_labels_a, num_labels_b)
    for i in range(num_labels_a):
        set_a = labels_a == i
        for j in range(num_labels_b):
            set_b = labels_b == j
            intersection = (set_a & set_b).sum()
            union = (set_a | set_b).sum()
            dist[i][j] = float(intersection) / union

    return dist

