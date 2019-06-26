#Copyright (c) 2019 Uber Technologies, Inc.
#
#Licensed under the Uber Non-Commercial License (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at the root directory of this project. 
#
#See the License for the specific language governing permissions and
#limitations under the License.
import itertools
import json
import logging
import pickle
import time
from collections import namedtuple
from pprint import pprint

import distributed
import numpy as np
import torch

from .normalizers import centered_rank_normalizer

StepStats = namedtuple(
    "StepStats",
    [
        "po_returns_mean",
        "po_returns_median",
        "po_returns_std",
        "po_returns_max",
        "po_returns_min",
        "po_novelties_mean",
        "po_novelties_var",
        "po_novelties_median",
        "po_novelties_max",
        "po_novelties_min",
        "po_len_mean",
        "po_len_std",
        "episodes_this_step",
        "timesteps_this_step",
        "time_elapsed_this_step",
    ],
)

EvalStats = namedtuple(
    "EvalStats",
    [
        "eval_returns_mean",
        "eval_returns_median",
        "eval_novelties_mean",
        "eval_novelties_median",
        "eval_novelties_max",
        "eval_novelties_min",
        "eval_returns_std",
        "eval_len_mean",
        "eval_len_std",
        "eval_n_episodes",
    ],
)
OverallEvalStats = namedtuple(
    "OverallEvalStats",
    [
        "eval_returns_mean",
        "eval_returns_median",
        "eval_novelties_mean",
        "eval_novelties_median",
        "eval_novelties_max",
        "eval_novelties_min",
        "eval_returns_std",
        "eval_len_mean",
        "eval_len_std",
        "eval_n_episodes",
        "time_elapsed",
    ],
)

POResult = namedtuple(
    "POResult", ["descriptors", "returns", "lengths", "novelties", "extra"]
)
EvalResult = namedtuple("EvalResult", ["returns", "lengths", "novelties"])

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def run_po_batch(seed, niche, population, batch_size, extra):
    population.seed(seed)

    descriptors, thetas = population.sample(batch_size)

    niche.seed(int(seed))
    returns, lengths, novelties, new_extra = niche.rollout_batch(
        thetas, batch_size, extra
    )

    return POResult(
        returns=returns,
        descriptors=descriptors,
        lengths=lengths,
        novelties=novelties,
        extra=new_extra,
    )


def collect_po_results(po_results):
    descriptors = list(itertools.chain(*[r.descriptors for r in po_results]))
    returns = np.concatenate([r.returns for r in po_results])
    lengths = np.concatenate([r.lengths for r in po_results])
    novelties = np.concatenate([r.novelties for r in po_results])
    extra = [r.extra for r in po_results]
    return descriptors, returns, lengths, novelties, extra


def run_eval_batch(seed, niche, population, batch_size, extra):
    niche.seed(int(seed))
    eval_theta = population.eval_theta()
    if not isinstance(eval_theta, list):
        eval_theta = [eval_theta]
    results = []
    for theta in eval_theta:
        returns, lengths, novelties, _extra = niche.rollout_batch(
            (theta for _ in range(batch_size)), batch_size, extra, eval_mode=True
        )

        results.append(
            EvalResult(returns=returns, lengths=lengths, novelties=novelties)
        )
    return results


def collect_eval_results(eval_results):
    n_evals = len(eval_results[0])
    eval_returns = [
        np.concatenate([r[i].returns for r in eval_results]) for i in range(n_evals)
    ]
    eval_lengths = [
        np.concatenate([r[i].lengths for r in eval_results]) for i in range(n_evals)
    ]
    eval_novelties = [
        np.concatenate([r[i].novelties for r in eval_results]) for i in range(n_evals)
    ]
    return eval_returns, eval_lengths, eval_novelties


class GenESOptimizer:
    def __init__(
        self,
        client,
        population,
        optimizer,
        niche,
        batch_size,
        pop_size,
        eval_batch_size,
        evals_per_step,
        share_seed=False,
        returns_normalizer=None,
        novelty_normalizer=None,
        log_fnm="log.json",
        seed=42,
        device="cpu",
        optim_id=0,
    ):
        """
        client: Dask client
        niche: Subclass of Niche (must override `serialize` and `unserialize`)
        batch_size: number of pseudo-offspring to send to each worker at a time
            Must be divisible by 2 (because we work with antithetic pairs)
        pop_size: number of pseudo-offspring (including antitheses).
            Must be divisible by batch_size
        eval_batch_size: number times each worker should
            evaluate current theta at a time
        evals_per_step: total number of times current theta
            should be evaluated at each step
        share_seed: use same environment seed for antithetic pairs
            (should decrease variance)
        returns_normalizer: normalizer for returns
            (centered_rank_normalizer by default)
        novelty_normalizer: normalizer for novelty vectors
            (centered_rank_normalizer by default)
        seed: seed for shared random state (yields deterministic training)
        optim_id: name for this optimizer (only used in logging statements)
        """

        logger.info("Creating optimizer {}...".format(optim_id))
        self.optim_id = optim_id
        self.client = client
        self.device = device

        self.population = population
        # print(self.theta)
        logger.info(
            "Optimizer {} optimizing {} parameters".format(
                optim_id,
                np.prod([len(param.view(-1)) for param in population.parameters()]),
            )
        )
        self.niche = niche
        self.optimizer = optimizer

        assert pop_size % batch_size == 0
        assert batch_size % 2 == 0
        self.pop_size = pop_size
        self.batch_size = batch_size

        assert evals_per_step % eval_batch_size == 0
        self.eval_batch_size = eval_batch_size
        self.evals_per_step = evals_per_step

        self.share_seed = share_seed

        self.random_state = np.random.RandomState(seed)

        if returns_normalizer is None:
            returns_normalizer = centered_rank_normalizer
        self.returns_normalizer = returns_normalizer
        if novelty_normalizer is None:
            novelty_normalizer = centered_rank_normalizer
        self.novelty_normalizer = novelty_normalizer

        logger.info("Optimizer {} created!".format(optim_id))

        self.log_fnm = log_fnm
        self.log_dict = {}

        self.submit_deps()

    def submit_deps(self):
        [self.niche_fut] = self.client.scatter([self.niche])
        # self.client.replicate(self.niche_fut, 10)

    def start_chunk(self, runner, pop_size, batch_size, priority=0):
        n_batches = pop_size // batch_size
        logger.info(
            "Optimizer {} spawning {} batches of size {}".format(
                self.optim_id, n_batches, batch_size
            )
        )

        rs_seeds = self.random_state.randint(np.int32(2 ** 31 - 1), size=n_batches)

        # print(self.theta[:10])

        extra = self.niche.get_extra()
        [pop_fut, extra_fut] = self.client.scatter([self.population, extra])
        # self.client.replicate(theta_fut, 10)

        futures = self.client.map(
            runner,
            rs_seeds,
            itertools.repeat(self.niche_fut),
            itertools.repeat(pop_fut),
            itertools.repeat(batch_size),
            itertools.repeat(extra_fut),
            pure=False,
            priority=priority,
        )
        return futures

    def get_chunk(self, futures):
        return self.client.gather(futures)

    # def update_population(self, descriptors, returns, _novelty, batch_size=500):
    #     self.optimizer.zero_grad()
    #     returns = torch.tensor(self.returns_normalizer(returns), device=self.device)
    #     for start in range(0, len(descriptors), batch_size):
    #         end = start + batch_size
    #         zs = self.population.decode(descriptors[start:end])
    #         loss = sum(
    #             -returns[i] * self.population(z) for i, z in zip(range(start, end), zs)
    #         ) / len(descriptors)
    #         loss.backward()
    #     self.optimizer.step()
    #     self.population.step()

    def update_population(self, descriptors, returns, _novelty):
        # returns = np.array([-5, 10])
        self.optimizer.zero_grad()
        returns = torch.tensor(self.returns_normalizer(returns), device=self.device)
        # loss = -self.population.log_prob(descriptors[0])
        # loss.backward()
        # print(self.population.mu.grad[0])
        # print(
        #     (self.population.decode(descriptors[0]) - self.population.mu)
        #     / self.population.sigma ** 2
        # )

        ratio = torch.cat(
            [self.population.ratio(descriptor) for descriptor in descriptors]
        )
        loss = -(returns * ratio).mean()

        # loss = -sum(
        #     rets * self.population.log_prob(descriptor)
        #     for rets, descriptor in zip(returns, descriptors)
        # ) / len(descriptors)
        loss.backward()
        # print(self.population.mu.grad[0])
        # print(
        #     sum(
        #         (self.population.decode(descriptor)[0] - self.population.mu[0]) * ret
        #         for descriptor, ret in zip(descriptors, returns)
        #     )
        #     / (len(descriptors) * self.population.sigma ** 2)
        # )
        self.optimizer.step()
        self.population.step()

    def start_eval(self):
        step_t_start = time.time()

        futures = self.start_chunk(
            run_eval_batch, self.evals_per_step, self.eval_batch_size
        )

        return futures, step_t_start

    def get_eval(self, res):
        futures, step_t_start = res
        eval_results = self.get_chunk(futures)
        step_t_end = time.time()

        stats = []
        all_returns = []
        all_lengths = []
        all_novelties = []
        all_returns, all_lengths, all_novelties = collect_eval_results(eval_results)
        n_episodes = len(all_returns) * len(all_returns[0])
        logger.info(
            "get_eval {} finished running {} episodes, {} timesteps".format(
                self.optim_id, n_episodes, sum(l.sum() for l in all_lengths)
            )
        )

        for eval_returns, eval_lengths, eval_novelties in zip(
            all_returns, all_lengths, all_novelties
        ):
            stats.append(
                EvalStats(
                    eval_returns_mean=eval_returns.mean(),
                    eval_returns_median=np.median(eval_returns),
                    eval_returns_std=eval_returns.std(),
                    eval_novelties_mean=eval_novelties.mean(),
                    eval_novelties_median=np.median(eval_novelties),
                    eval_novelties_max=eval_novelties.max(),
                    eval_novelties_min=eval_novelties.min(),
                    eval_len_mean=eval_lengths.mean(),
                    eval_len_std=eval_lengths.std(),
                    eval_n_episodes=len(eval_returns),
                )
            )

        all_returns = np.concatenate(all_returns)
        all_lengths = np.concatenate(all_lengths)
        all_novelties = np.concatenate(all_novelties)

        overall_stats = OverallEvalStats(
            eval_returns_mean=all_returns.mean(),
            eval_returns_median=np.median(all_returns),
            eval_returns_std=all_returns.std(),
            eval_novelties_mean=all_novelties.mean(),
            eval_novelties_median=np.median(all_novelties),
            eval_novelties_max=all_novelties.max(),
            eval_novelties_min=all_novelties.min(),
            eval_len_mean=all_lengths.mean(),
            eval_len_std=all_lengths.std(),
            eval_n_episodes=n_episodes,
            time_elapsed=step_t_end - step_t_start,
        )
        return stats, overall_stats

    def start_step(self):
        step_t_start = time.time()

        futures = self.start_chunk(run_po_batch, self.pop_size, self.batch_size)

        return futures, step_t_start

    def get_step(self, res, show_progress=False):
        futures, step_t_start = res
        if show_progress:
            distributed.progress(futures)
            print()
        step_results = self.get_chunk(futures)

        descriptors, po_returns, po_lengths, po_novelties, extra = collect_po_results(
            step_results
        )
        episodes_this_step = 2 * len(po_returns)
        timesteps_this_step = po_lengths.sum()

        logger.info(
            "Optimizer {} finished running {} episodes, {} timesteps".format(
                self.optim_id, episodes_this_step, timesteps_this_step
            )
        )

        self.niche.update_extra(extra)

        self.update_population(descriptors, po_returns, po_novelties)
        logger.info("Optimizer {} finished updating population".format(self.optim_id))

        step_t_end = time.time()

        return (
            descriptors,
            po_returns,
            po_novelties,
            StepStats(
                po_returns_mean=po_returns.mean(),
                po_returns_median=np.median(po_returns),
                po_returns_std=po_returns.std(),
                po_returns_max=po_returns.max(),
                po_returns_min=po_returns.min(),
                po_novelties_mean=po_novelties.mean(),
                po_novelties_median=np.median(po_novelties),
                po_novelties_var=po_novelties.var(),
                po_novelties_max=po_novelties.max(),
                po_novelties_min=po_novelties.min(),
                po_len_mean=po_lengths.mean(),
                po_len_std=po_lengths.std(),
                episodes_this_step=episodes_this_step,
                timesteps_this_step=timesteps_this_step,
                time_elapsed_this_step=step_t_end - step_t_start,
            ),
        )

    def log_pair(self, key, val):
        self.log_dict[key] = val

    def dump_log(self):
        log_dict = {k: np.asscalar(np.array(v)) for k, v in self.log_dict.items()}
        pprint(log_dict)
        with open(self.log_fnm, "a") as f:
            f.write(json.dumps(log_dict) + "\n")
        self.log_dict = {}

    def log(self, named_tup, suffix=""):  # pylint: disable=no-self-use
        for key, val in named_tup._asdict().items():
            self.log_pair(key + suffix, val)

    def checkpoint(self, local_path, save_fn, iteration):  # pylint: disable=no-self-use
        if local_path is not None:
            local_path = local_path.format(iteration=iteration)
            save_fn(local_path)

    def save_po(  # pylint: disable=no-self-use
        self, po_indices, po_returns, po_novelties
    ):
        def save_fn(local_path):
            with open(local_path, "wb") as f:
                pickle.dump((po_indices, po_returns, po_novelties), f)

        return save_fn

    def optimize(
        self,
        n_iterations,
        niche_path=None,
        pop_path=None,
        po_path=None,
        show_progress=False,
    ):
        logger.info(
            "Optimizer {} running for {} iterations".format(self.optim_id, n_iterations)
        )

        episodes_so_far = 0
        timesteps_so_far = 0
        t_start = time.time()

        def run_step(submit_deps=False):
            if submit_deps:
                self.submit_deps()
            step_futures = self.start_step()
            eval_futures = self.start_eval()

            po_indices, po_returns, po_novelties, step_stats = self.get_step(
                step_futures, show_progress=show_progress
            )
            eval_stats, overall_eval_stats = self.get_eval(eval_futures)
            return (
                step_stats,
                po_indices,
                po_returns,
                po_novelties,
                eval_stats,
                overall_eval_stats,
            )

        for iteration in range(n_iterations):
            logger.info("=" * 20 + " Iteration {} ".format(iteration) + "=" * 20)
            if not isinstance(self.population.eval_theta(), list):
                print(self.population.eval_theta()[:10])
            # print(self.niche.stats.mean)

            # save population before so rollouts matche up

            self.checkpoint(niche_path, self.niche.save, iteration)
            self.checkpoint(pop_path, self.population.save, iteration)

            step_stats, po_indices, po_returns, po_novelties, eval_stats, overall_eval_stats = (
                run_step()
            )

            self.checkpoint(
                po_path, self.save_po(po_indices, po_returns, po_novelties), iteration
            )

            episodes_so_far += step_stats.episodes_this_step
            timesteps_so_far += step_stats.timesteps_this_step

            self.log(step_stats)
            self.log(overall_eval_stats)
            for i, stats in enumerate(eval_stats):
                self.log(stats, suffix="_" + str(i))

            self.log_pair("timesteps_so_far", timesteps_so_far)
            self.log_pair("episodes_so_far", episodes_so_far)
            self.log_pair("time_elapsed_so_far", time.time() - t_start)
            self.log_pair("iteration", iteration)

            self.dump_log()
