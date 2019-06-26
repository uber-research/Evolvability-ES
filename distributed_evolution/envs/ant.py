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


class AntEnv(BulletEnv):
    def __init__(self, seed=None):
        super().__init__("AntBulletEnv-v0", seed)
        self.novelty_space = gym.spaces.Box(shape=(2,), low=-100, high=100)
        self.coord = "x"
        self.direction = "+"

    def step(self, action):
        state, _reward, done, info = self.env.step(action)
        x, y, _z = self.env.unwrapped.robot.body_xyz
        # dist = (x ** 2 + y ** 2) ** 0.5
        if done:
            reward = x if self.coord == "x" else y
            if self.direction == "-":
                reward = -reward
        else:
            reward = 0.0
        return state, reward, done, info

    def novelty(self):
        return self.env.unwrapped.robot.body_xyz[:2]

    def set_target(self, coord, direction):
        self.coord = coord
        self.direction = direction
