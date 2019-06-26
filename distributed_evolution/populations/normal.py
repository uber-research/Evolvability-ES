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

# class LogProb(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, mu, sigma, descriptor, decode_fn):
#         ctx.save_for_backward(mu)
#         ctx.sigma = sigma
#         ctx.descriptor = descriptor
#         ctx.decode_fn = decode_fn
#         with torch.no_grad():
#             epsilon = (decode_fn(descriptor) - mu) / sigma
#             d = len(mu)
#             return -0.5 * (
#                 d * float(np.log(2 * np.pi)) + sigma ** d + epsilon.dot(epsilon)
#             )

#     @staticmethod
#     def backward(ctx, grad_output):
#         mu, = ctx.saved_tensors
#         grad = (ctx.decode_fn(ctx.descriptor) - mu) / ctx.sigma ** 2 * grad_output
#         return (grad, None, None, None)


class ProbRatio(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mu, sigma, descriptor, decode_fn):
        ctx.save_for_backward(mu)
        ctx.sigma = sigma
        ctx.descriptor = descriptor
        ctx.decode_fn = decode_fn
        return torch.ones(1, dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad_output):
        mu, = ctx.saved_tensors
        grad = (ctx.decode_fn(ctx.descriptor) - mu) / ctx.sigma ** 2 * grad_output
        return (grad, None, None, None)


class Normal(torch.nn.Module):
    def __init__(
        self, init_mu, sigma=0.02, sigma_decay=1.0, sigma_limit=0.001, device="cpu"
    ):
        super().__init__()
        self.mu = torch.nn.Parameter(torch.tensor(init_mu, device=device))
        self.dims = len(init_mu)
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
        return ProbRatio.apply(self.mu, self.sigma, descriptor, self.decode)

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
                noise.sample_index(self.random_state, len(self.mu))
                for _ in range(n_epsilons)
            ],
            dtype="int",
        )
        descriptors = [(idx, 1) for idx in noise_inds] + [
            (idx, -1) for idx in noise_inds
        ]
        thetas = (self.decode(descriptor) for descriptor in descriptors)
        return descriptors, thetas

    def decode(self, descriptor):
        noise_idx, direction = descriptor
        epsilon = torch.tensor(noise.get(noise_idx, len(self.mu)), device=self.device)
        with torch.no_grad():
            return self.mu + direction * self.sigma * epsilon

    def eval_theta(self):
        return self.mu

    def save(self, fnm):
        with open(fnm, "wb") as f:
            pickle.dump(
                {
                    "init_mu": self.mu.cpu().detach().numpy(),
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
