#Copyright (c) 2019 Uber Technologies, Inc.
#
#Licensed under the Uber Non-Commercial License (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at the root directory of this project. 
#
#See the License for the specific language governing permissions and
#limitations under the License.
import torch

from .gen_es import GenESOptimizer


class GenVarESOptimizer(GenESOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update_population(self, descriptors, _returns, novelty):
        self.optimizer.zero_grad()
        novelty = torch.tensor(
            self.novelty_normalizer(novelty), device=self.device, dtype=torch.float32
        )

        ratio = torch.cat(
            [self.population.ratio(descriptor) for descriptor in descriptors]
        )
        loss = -((novelty ** 2).sum(dim=1) * ratio).mean()

        loss.backward()
        self.optimizer.step()
        self.population.step()
