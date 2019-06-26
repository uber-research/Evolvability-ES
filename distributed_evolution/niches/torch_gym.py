#Copyright (c) 2019 Uber Technologies, Inc.
#
#Licensed under the Uber Non-Commercial License (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at the root directory of this project. 
#
#See the License for the specific language governing permissions and
#limitations under the License.
import time

import dill as pickle
import numpy as np
from distributed.protocol.serialize import register_serialization

from ..stats import RunningStats
from .gym import GymNiche


class TorchGymNiche(GymNiche):
    def __init__(
        self,
        make_env,
        model,
        ts_limit,
        ob_stats_prob=0.01,
        n_train_rollouts=1,
        n_eval_rollouts=1,
    ):
        super().__init__(
            make_env,
            model,
            ts_limit=ts_limit,
            ob_stats_prob=ob_stats_prob,
            n_train_rollouts=n_train_rollouts,
            n_eval_rollouts=n_eval_rollouts,
        )
        self.novelty_shape = self.env.novelty_space.shape

    def update_extra(self, stats):
        if self.model.needs_stats:
            for stat in stats:
                self.stats.update(stat)

    def save(self, fnm):
        with open(fnm, "wb") as f:
            pickle.dump((self.serialize(), self.stats), f)

    @classmethod
    def load(cls, fnm):
        with open(fnm, "rb") as f:
            frames, stats = pickle.load(f)
        niche = cls.deserialize(*frames)
        niche.stats = stats
        return niche

    def rollout(
        self,
        theta=None,
        stats=None,
        eval_mode=False,
        render=False,
        prerender=False,
        sleep=0.0,
        ts_limit=None,
    ):
        if theta is not None:
            self.set_theta(theta.cpu().detach().numpy())
        if self.model.needs_stats:
            if stats is None:
                stats = self.stats
            self.model.set_stats(stats.mean, stats.std)
        total_reward = 0.0
        n_steps = 0
        states = []
        if render and prerender:
            self.env.render()
        self.model.reset()
        state = self.env.reset()
        states.append(state)
        novelty = np.zeros(self.novelty_shape)
        n_rollouts = self.n_train_rollouts if not eval_mode else self.n_eval_rollouts
        for _ in range(n_rollouts):
            for _ in range(self.ts_limit if ts_limit is None else ts_limit):
                n_steps += 1
                if render:
                    self.env.render()
                    if sleep > 0:
                        time.sleep(sleep)
                action = self.model.act([state])[0]
                state, reward, done, _info = self.env.step(action)
                states.append(state)
                total_reward += reward
                if done:
                    break
            if not eval_mode:
                total_reward += self.model.supp_fitness()
            novelty += self.env.novelty()
        return (
            total_reward / n_rollouts,
            n_steps / n_rollouts,
            states,
            novelty / n_rollouts,
        )

    def rollout_batch(self, thetas, batch_size, stats=None, eval_mode=False):
        returns = np.zeros(batch_size)
        lengths = np.zeros(batch_size, dtype="int")
        new_stats = (
            RunningStats(self.env.observation_space.shape)
            if self.model.needs_stats
            else None
        )
        novelties = np.zeros((batch_size,) + self.env.novelty_space.shape)

        for i, theta in enumerate(thetas):
            returns[i], lengths[i], states, novelties[i] = self.rollout(
                theta, stats, eval_mode=eval_mode
            )
            if self.model.needs_stats and self.random_state.rand() < self.ob_stats_prob:
                states = np.array(states)
                new_stats.increment(
                    states.sum(axis=0), np.square(states).sum(axis=0), len(states)
                )

        return returns, lengths, novelties, new_stats

    def serialize(self):
        niche_frame = pickle.dumps(
            (self.make_env, self.ts_limit, self.ob_stats_prob, self.model.__class__)
        )
        model_frames = self.model.serialize()
        return {}, [niche_frame] + model_frames

    @classmethod
    def deserialize(cls, _header, frames):
        tup = pickle.loads(frames[0])

        make_env = tup[0]
        ts_limit = tup[1]
        ob_stats_prob = tup[2]
        Model = tup[3]

        model = Model.deserialize(frames[1:])

        return cls(make_env, model, ts_limit=ts_limit, ob_stats_prob=ob_stats_prob)


register_serialization(
    TorchGymNiche, TorchGymNiche.serialize, TorchGymNiche.deserialize
)
