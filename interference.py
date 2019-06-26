#Copyright (c) 2019 Uber Technologies, Inc.
#
#Licensed under the Uber Non-Commercial License (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at the root directory of this project. 
#
#See the License for the specific language governing permissions and
#limitations under the License.

import os
import re

import dill as pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from matplotlib.colors import LogNorm
from scipy.spatial.distance import pdist, squareform


def fun(x):
    return 5 * np.sin(0.2 * x) * np.sin(20 * x)


def max_var():
    npop = 500  # population size
    sigma = 0.5  # noise standard deviation
    alpha = 0.03  # learning rate

    mus = []
    variances = []

    # start the optimization
    mu = 1.0
    for _ in range(2000):
        epsilons = sigma * np.random.randn(npop)
        pop = mu + epsilons
        bcs = np.apply_along_axis(fun, 0, pop)

        var = bcs.var()
        print("Mu: {}, BC variance: {}".format(mu, var))
        mus.append(mu)
        variances.append(var)

        # standardize the rewards to have a gaussian distribution
        # A = (R - np.mean(R)) / np.std(R)
        # bcs = np.power(bcs, 2) + 2 * bcs.mean() * bcs
        bcs = (bcs - bcs.mean()) / bcs.std()

        # perform the parameter update. The matrix multiply below
        # is just an efficient way to sum up all the rows of the noise matrix N,
        # where each row N[j] is weighted by A[j]
        # mu = mu + alpha / (npop * (sigma ** 2)) * np.dot(bcs.T, epsilons)

        mu = mu + alpha / (npop * (sigma ** 2)) * np.dot(np.power(bcs, 2), epsilons)

        # mu = mu + alpha / (npop * (sigma ** 2)) * np.dot(
        #     bcs + 2 * bcs.mean() * bcs, epsilons
        # )
        # mu = pop[np.argmax(np.abs(bcs))]

    return mus


def kern(b):
    k_sigma = 1.0  # kernel standard deviation
    pairwise_sq_dists = squareform(pdist(b.reshape(-1, 1), "sqeuclidean"))
    # pylint: disable=invalid-unary-operand-type
    k = scipy.exp(-pairwise_sq_dists / k_sigma ** 2)
    return k


def max_ent():
    npop = 500  # population size
    sigma = 0.5  # noise standard deviation
    k_sigma = 1.0  # kernel standard deviation
    alpha = 0.10  # learning rate

    mus = []
    variances = []
    entropies = []

    # start the optimization
    mu = 1.0
    for _ in range(2000):
        epsilons = sigma * np.random.randn(npop)
        pop = mu + epsilons
        bcs = np.apply_along_axis(fun, 0, pop)

        var = bcs.var()
        mus.append(mu)
        variances.append(var)

        # standardize the rewards to have a gaussian distribution
        # A = (R - np.mean(R)) / np.std(R)
        # bcs = np.power(bcs, 2) + 2 * bcs.mean() * bcs
        standard_bcs = (bcs - bcs.mean()) / bcs.std()

        # perform the parameter update. The matrix multiply below
        # is just an efficient way to sum up all the rows of the noise matrix N,
        # where each row N[j] is weighted by A[j]
        # mu = mu + alpha / (npop * (sigma ** 2)) * np.dot(bcs.T, epsilons)

        # mu = mu + alpha / (npop * (sigma ** 2)) * np.dot(
        #     bcs + 2 * bcs.mean() * bcs, epsilons
        # )
        # mu = pop[np.argmax(np.abs(bcs))]

        k = kern(standard_bcs)
        # p_b = k.mean(axis=1)
        log_pi = epsilons / (sigma ** 2)
        # sep_term = -(k.dot(log_pi) / (npop * p_b)).mean()
        # down_term = -np.log(p_b).dot(log_pi) / npop
        # grads = sep_term + down_term
        grads = -((k / k.mean(axis=0)).mean(axis=1) + np.log(k.mean(axis=1))).dot(
            log_pi
        )
        mu += alpha * grads / npop

        entropy = -np.log(kern(bcs).mean(axis=1)).mean()
        entropies.append(entropy)
        print("Mu: {}, BC variance: {}, Entropy: {}".format(mu, var, entropy))

    return mus


def plot():
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    mpl.style.use("seaborn-muted")

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, sharey=False, sharex=False, figsize=(10, 3)
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.3, left=0.08, right=0.98)

    ax1.grid()
    ax1.set_ylabel("Behavior")
    ax1.set_xlabel("Genome")
    ax1.text(
        0.5,
        -0.35,
        "(a) Interference Pattern",
        horizontalalignment="center",
        transform=ax1.transAxes,
        fontsize=15,
    )

    xs = np.linspace(0, 15.7, 5000)
    ys = np.apply_along_axis(fun, 0, xs)
    ax1.plot(xs, ys, linewidth=0.3)
    ax1.axvline(x=np.pi / (2 * 0.2), color="gray", linestyle="dashed")

    ax2.grid()
    ax2.set_ylabel("Genome")
    ax2.set_xlabel("Generation")
    ax2.axhline(y=np.pi / (2 * 0.2), color="gray", linestyle="dashed")
    ax2.text(
        0.5,
        -0.35,
        "(b) MaxVar Evolvability ES",
        horizontalalignment="center",
        transform=ax2.transAxes,
        fontsize=15,
    )
    ax2.plot(max_var())

    ax3.grid()
    ax3.set_ylabel("Genome")
    ax3.set_xlabel("Generation")
    ax3.axhline(y=np.pi / (2 * 0.2), color="gray", linestyle="dashed")
    ax3.text(
        0.5,
        -0.35,
        "(c) MaxEnt Evolvability ES",
        horizontalalignment="center",
        transform=ax3.transAxes,
        fontsize=15,
    )
    ax3.plot(max_ent())

    # plt.show()

    plt.savefig("../gecco_figures/{}.pdf".format("interference"))
    plt.clf()


if __name__ == "__main__":
    plot()
