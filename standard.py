# Copyright (c) 2019 Uber Technologies, Inc.
#
#Licensed under the Uber Non-Commercial License (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at the root directory of this project. 
#
#See the License for the specific language governing permissions and
#limitations under the License.

import click
import torch
from dask import distributed

from distributed_evolution import populations
from distributed_evolution.gen_es import GenESOptimizer
from distributed_evolution.niches import TorchGymNiche
from distributed_evolution.normalizers import (
    centered_rank_normalizer,
    normal_normalizer,
)
from distributed_evolution.policies.cheetah import CheetahPolicy


@click.command()
@click.option("--learning-rate", default=0.01)
@click.option("--batch-size", default=10)
@click.option("--pop-size", default=10000)
@click.option("--l2-coeff", default=0.005)
@click.option("--noise-std", default=0.02)
@click.option("--n-iterations", default=100)
@click.option("--n-rollouts", default=1)
@click.option("--noise-decay", default=1.0)
@click.option("--action-noise", default=0.00)
@click.option("--env-deterministic / --env-stochastic", default=True)
@click.option("--evals-per-step", default=100)
@click.option("--eval-batch-size", default=10)
@click.option("--returns-normalization", default="centered_ranks")
@click.option("--single-threaded / --multi-threaded", default=False)
@click.option("--progress / --no-progress", default=False)
@click.option("--gpu-mem-frac", default=0.2)
@click.option("--device", default="cpu")
@click.option("--seed", default=42)
@click.option("--niche-path", default="niche.ckpt")
@click.option("--log-fnm", default="log.json")
@click.option("--pop-path", default="pop.ckpt")
@click.option("--po-path", default="po_gen_{iteration}.ckpt")
@click.option("--env-name", default="cheetah")
def main(
    learning_rate,
    batch_size,
    pop_size,
    l2_coeff,
    noise_std,
    noise_decay,
    n_iterations,
    n_rollouts,
    env_deterministic,
    returns_normalization,
    evals_per_step,
    eval_batch_size,
    single_threaded,
    log_fnm,
    gpu_mem_frac,
    action_noise,
    progress,
    seed,
    niche_path,
    pop_path,
    po_path,
    device,
    env_name,
):
    cluster = distributed.LocalCluster(
        n_workers=4, processes=True, threads_per_worker=1
    )
    client = distributed.Client(cluster)

    def make_env():
        from distributed_evolution.envs import CheetahEnv, AntEnv

        if env_name == "cheetah":
            Env = CheetahEnv
        elif env_name == "ant":
            Env = AntEnv
        else:
            raise Exception("Invalid env_name")

        if env_deterministic:
            return Env(seed=seed)
        return Env()

    env = make_env()
    print(env.action_space)

    model = CheetahPolicy(
        env.observation_space.shape,
        env.action_space.shape[0],
        env.action_space.low,
        env.action_space.high,
        ac_noise_std=action_noise,
        seed=seed,
        gpu_mem_frac=gpu_mem_frac,
        single_threaded=single_threaded,
    )

    niche = TorchGymNiche(
        make_env, model, ts_limit=1000, n_train_rollouts=n_rollouts, n_eval_rollouts=1
    )

    population = populations.Normal(
        model.get_theta(), noise_std, sigma_decay=noise_decay, device=device
    )
    optim = torch.optim.Adam(
        population.parameters(), lr=learning_rate, weight_decay=l2_coeff
    )

    if returns_normalization == "centered_ranks":
        returns_normalizer = centered_rank_normalizer
    elif returns_normalization == "normal":
        returns_normalizer = normal_normalizer
    else:
        raise ValueError("Invalid returns normalizer {}".format(returns_normalizer))

    optimizer = GenESOptimizer(
        client,
        population,
        optim,
        niche,
        batch_size=batch_size,
        pop_size=pop_size,
        eval_batch_size=eval_batch_size,
        evals_per_step=evals_per_step,
        returns_normalizer=returns_normalizer,
        seed=seed,
        device=device,
        log_fnm=log_fnm,
    )
    optimizer.optimize(
        n_iterations,
        show_progress=progress,
        pop_path=pop_path,
        niche_path=niche_path,
        po_path=po_path,
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
