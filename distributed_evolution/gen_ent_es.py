#Copyright (c) 2019 Uber Technologies, Inc.
#
#Licensed under the Uber Non-Commercial License (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at the root directory of this project. 
#
#See the License for the specific language governing permissions and
#limitations under the License.
import scipy
import scipy.spatial.distance
import torch

from .gen_es import GenESOptimizer


class GenEntESOptimizer(GenESOptimizer):
    def __init__(self, *args, bandwidth=1.0, **kwargs):
        self.bandwidth = bandwidth
        super().__init__(*args, **kwargs)

    def update_population(self, descriptors, _returns, novelty):
        self.optimizer.zero_grad()
        novelty = self.novelty_normalizer(novelty)
        dists = scipy.spatial.distance.squareform(
            scipy.spatial.distance.pdist(novelty, "sqeuclidean")
        )
        # pylint: disable=invalid-unary-operand-type
        k = torch.tensor(
            scipy.exp(-dists / self.bandwidth ** 2),
            device=self.device,
            dtype=torch.float32,
        )

        ratio = torch.cat(
            [self.population.ratio(descriptor) for descriptor in descriptors]
        )
        p = (k * ratio).mean(dim=1)
        loss = (torch.log(p) * ratio).mean()

        loss.backward()
        self.optimizer.step()
        self.population.step()
