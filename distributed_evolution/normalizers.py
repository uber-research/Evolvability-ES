#Copyright (c) 2019 Uber Technologies, Inc.
#
#Licensed under the Uber Non-Commercial License (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at the root directory of this project. 
#
#See the License for the specific language governing permissions and
#limitations under the License.
import numpy as np

from .stats import compute_centered_ranks


def centered_rank_normalizer(x):
    if len(x.shape) == 1:
        return compute_centered_ranks(x)
    elif len(x.shape) == 2:
        ranked = np.zeros(x.shape)
        for i in range(x.shape[1]):
            ranked[:, i] = compute_centered_ranks(x[:, i])
        return ranked
    raise ValueError("X must have 1 or 2 dimensions")


def normal_normalizer(x, eps=1e-5):
    return (x - x.mean(axis=0)) / (x.std(axis=0) + eps)
