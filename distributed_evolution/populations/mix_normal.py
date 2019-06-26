#Copyright (c) 2019 Uber Technologies, Inc.
#
#Licensed under the Uber Non-Commercial License (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at the root directory of this project. 
#
#See the License for the specific language governing permissions and
#limitations under the License.
import pickle

import numpy as np
import torch
import torch.utils.checkpoint

from ..noise_module import noise


class ProbRatio(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mus, sigma, descriptor, decode_fn):
        ctx.save_for_backward(mus)
        ctx.sigma = sigma
        ctx.descriptor = descriptor
        ctx.decode_fn = decode_fn
        return torch.ones(1, dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad_output):
        mus, = ctx.saved_tensors
        theta = ctx.decode_fn(ctx.descriptor)
        epsilons = [theta - mu for mu in mus]
        grads = torch.stack(
            [
                (epsilon / ctx.sigma ** 2)
                / (
                    1
                    + sum(
                        torch.exp(
                            -0.5
                            * (other.dot(other) - epsilon.dot(epsilon))
                            / ctx.sigma ** 2
                        )
                        for other in epsilons
                        if other is not epsilon
                    )
                )
                * grad_output
                for epsilon in epsilons
            ]
        )
        return (grads, None, None, None)


class MixtureNormal(torch.nn.Module):
    def __init__(
        self, init_mus, sigma=0.02, sigma_decay=1.0, sigma_limit=0.001, device="cpu"
    ):
        super().__init__()
        self.mus = torch.nn.Parameter(
            torch.tensor(init_mus, device=device, dtype=torch.float32)
        )
        self.dims = len(init_mus[0])
        self.sigma = sigma
        self.sigma_decay = sigma_decay
        self.sigma_limit = sigma_limit
        self.random_state = np.random.RandomState()
        self.device = device

    def seed(self, seed):
        self.random_state.seed(seed)

    def step(self):
        self.sigma *= self.sigma_decay
        if self.sigma < self.sigma_limit:
            self.sigma = self.sigma_limit

    def forward(self, descriptor):
        return self.ratio(descriptor)

    def ratio(self, descriptor):
        return ProbRatio.apply(self.mus, self.sigma, descriptor, self.decode)

    # def log_prob(self, descriptor):
    #     # def compute_score(mu):
    #     #     epsilon = (self.decode(descriptor) - mu) / self.sigma
    #     #     return -0.5 * (
    #     #         self.dims * float(np.log(2 * np.pi))
    #     #         + self.sigma ** self.dims
    #     #         + epsilon.dot(epsilon)
    #     #     )

    #     # return torch.utils.checkpoint.checkpoint(compute_score, self.mu)
    #     return LogProb.apply(self.mu, self.sigma, descriptor, self.decode)

    def sample(self, batch_size):
        assert batch_size % 2 == 0
        n_epsilons = batch_size // 2
        noise_inds = np.asarray(
            [
                noise.sample_index(self.random_state, len(self.mus[0]))
                for _ in range(n_epsilons)
            ],
            dtype="int",
        )
        dist_inds = np.random.randint(len(self.mus), size=n_epsilons)
        descriptors = [
            (noise_idx, dist_idx, 1)
            for noise_idx, dist_idx in zip(noise_inds, dist_inds)
        ] + [
            (noise_idx, dist_idx, -1)
            for noise_idx, dist_idx in zip(noise_inds, dist_inds)
        ]
        thetas = (self.decode(descriptor) for descriptor in descriptors)
        return descriptors, thetas

    def decode(self, descriptor):
        noise_idx, dist_idx, direction = descriptor
        epsilon = torch.tensor(
            noise.get(noise_idx, len(self.mus[0])), device=self.device
        )
        with torch.no_grad():
            return self.mus[dist_idx] + direction * self.sigma * epsilon

    def eval_theta(self):
        return [self.mus[i] for i in range(len(self.mus))]

    def save(self, fnm):
        with open(fnm, "wb") as f:
            pickle.dump(
                {
                    "init_mus": self.mus.cpu().detach().numpy(),
                    "sigma": self.sigma,
                    "sigma_decay": self.sigma_decay,
                    "sigma_limit": self.sigma_limit,
                    "device": self.device,
                },
                f,
            )

    @classmethod
    def load(cls, fnm):
        with open(fnm, "rb") as f:
            return cls(**pickle.load(f))
