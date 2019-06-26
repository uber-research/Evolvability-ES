#Copyright (c) 2019 Uber Technologies, Inc.
#
#Licensed under the Uber Non-Commercial License (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at the root directory of this project. 
#
#See the License for the specific language governing permissions and
#limitations under the License.
import gym

from .bullet import BulletEnv

# class CheetahEnv(gym.Env):
#     def __init__(self):
#         self.env = gym.make("HalfCheetah-v2")
#         self.action_space = self.env.action_space
#         self.observation_space = self.env.observation_space

#     def reset(self):
#         return self.env.reset()

#     def render(self, mode="human"):
#         self.env.render(mode)

#     def step(self, action):
#         return self.env.step(action)

#     def seed(self, seed):
#         self.env.seed(seed)

#     def novelty(self):
#         mass = np.expand_dims(self.env.unwrapped.model.body_mass, 1)
#         xpos = self.env.unwrapped.sim.data.xipos
#         center = np.sum(mass * xpos, 0) / np.sum(mass)
#         return center[0]


class CheetahEnv(BulletEnv):
    def __init__(self, seed=None):
        super().__init__("HalfCheetahBulletEnv-v0", seed)
        self.novelty_space = gym.spaces.Box(shape=(1,), low=-100, high=100)

    def step(self, action):
        state, _reward, done, info = self.env.step(action)
        reward = self.env.unwrapped.robot.body_xyz[0] if done else 0.0
        return state, reward, done, info

    def novelty(self):
        return self.env.unwrapped.robot.body_xyz[0]
