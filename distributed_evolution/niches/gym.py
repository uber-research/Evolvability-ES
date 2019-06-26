#Copyright (c) 2019 Uber Technologies, Inc.
#
#Licensed under the Uber Non-Commercial License (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at the root directory of this project. 
#
#See the License for the specific language governing permissions and
#limitations under the License.
import logging

import dill as pickle
import numpy as np
from distributed.protocol.serialize import register_serialization

from ..stats import RunningStats
from .core import Niche

logger = logging.getLogger(__name__)


def sample_states(env, batch_size):
    batch = []
    state = env.reset()
    for _ in range(batch_size):
        state, _reward, done, _info = env.step(env.action_space.sample())
        batch.append(state)
        if done:
            state = env.reset()
    return batch


class GymNiche(Niche):
    def __init__(
        self,
        make_env,
        model,
        ts_limit,
        ob_stats_prob=0.01,
        n_train_rollouts=1,
        n_eval_rollouts=1,
    ):
        self.make_env = make_env
        self.env = make_env()
        self.model = model
        self.ts_limit = ts_limit
        self.ob_stats_prob = ob_stats_prob
        self.n_train_rollouts = n_train_rollouts
        self.n_eval_rollouts = n_eval_rollouts
        self.random_state = np.random.RandomState()
        self.stats = (
            RunningStats(self.env.observation_space.shape)
            if self.model.needs_stats
            else None
        )

    def get_theta(self):
        return self.model.get_theta()

    def set_theta(self, theta):
        self.model.set_theta(theta)

    def seed(self, seed):
        self.random_state.seed(seed)
        self.env.seed(seed)
        self.model.seed(seed)

    def save(self, fnm):
        with open(fnm, "wb") as f:
            pickle.dump((self.serialize(), self.model.get_theta(), self.stats), f)

    @classmethod
    def load(cls, fnm):
        with open(fnm, "rb") as f:
            frames, theta, stats = pickle.load(f)
        niche = cls.deserialize(*frames)
        niche.set_theta(theta)
        niche.stats = stats
        return niche

    def rollout(
        self, theta=None, stats=None, eval_mode=False, render=False, prerender=False
    ):
        if theta is not None:
            self.set_theta(theta)
        if self.model.needs_stats:
            if stats is None:
                stats = self.stats
            self.model.set_stats(stats.mean, stats.std)
        n_rollouts = self.n_train_rollouts if not eval_mode else self.n_eval_rollouts
        total_reward = 0.0
        n_steps = 0.0
        states = []
        if render and prerender:
            self.env.render()
        for _ in range(n_rollouts):
            self.model.reset()
            state = self.env.reset()
            states.append(state)
            for _ in range(self.ts_limit):
                n_steps += 1
                if render:
                    self.env.render()
                action = self.model.act([state])[0]
                state, reward, done, _info = self.env.step(action)
                states.append(state)
                total_reward += reward
                if done:
                    break
            if not eval_mode:
                total_reward += self.model.supp_fitness()
        return total_reward / n_rollouts, n_steps / n_rollouts, states

    def rollout_batch(self, thetas, batch_size, stats=None, eval_mode=False):
        returns = np.zeros(batch_size)
        lengths = np.zeros(batch_size, dtype="int")
        new_stats = (
            RunningStats(self.env.observation_space.shape)
            if self.model.needs_stats
            else None
        )

        for i, theta in enumerate(thetas):
            returns[i], lengths[i], states = self.rollout(
                theta, stats, eval_mode=eval_mode
            )
            # print("On step {}".format(i))
            if self.model.needs_stats and self.random_state.rand() < self.ob_stats_prob:
                # print("Updating stats on step {}".format(i))
                states = np.array(states)
                new_stats.increment(
                    states.sum(axis=0), np.square(states).sum(axis=0), len(states)
                )

        return returns, lengths, new_stats

    def update_extra(self, stats_pos, stats_neg):
        if self.model.needs_stats:
            for stats in stats_pos + stats_neg:
                self.stats.update(stats)

    def get_extra(self):
        return self.stats

    def serialize(self):
        niche_frame = pickle.dumps(
            (
                self.make_env,
                self.ts_limit,
                self.ob_stats_prob,
                self.n_train_rollouts,
                self.n_eval_rollouts,
                self.model.__class__,
            )
        )
        model_frames = self.model.serialize()
        return {}, [niche_frame] + model_frames

    @classmethod
    def deserialize(cls, _header, frames):
        tup = pickle.loads(frames[0])

        make_env = tup[0]
        ts_limit = tup[1]
        ob_stats_prob = tup[2]
        n_train_rollouts = tup[3]
        n_eval_rollouts = tup[4]
        Model = tup[5]

        model = Model.deserialize(frames[1:])

        return cls(
            make_env, model, ts_limit, ob_stats_prob, n_train_rollouts, n_eval_rollouts
        )


register_serialization(GymNiche, GymNiche.serialize, GymNiche.deserialize)
