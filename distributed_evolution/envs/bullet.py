#Copyright (c) 2019 Uber Technologies, Inc.
#
#Licensed under the Uber Non-Commercial License (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at the root directory of this project. 
#
#See the License for the specific language governing permissions and
#limitations under the License.

import gym
import numpy as np
import pybullet_envs  # pylint: disable=unused-import

import pybullet


class BulletEnv(gym.Env):
    def __init__(self, env_id, seed=None):
        self.env = gym.make(env_id)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self._seed = seed

    def reset(self):
        if self._seed is not None:
            self.env.seed(self._seed)
        return self.env.reset()

    def render(self, mode="human"):
        if mode == "human":
            return self.env.render(mode)

        env = self.env.unwrapped
        base_pos = [0, 0, 0]
        base_pos = env.robot.body_xyz

        # pylint: disable=protected-access
        view_matrix = env._p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=env._cam_dist,
            yaw=env._cam_yaw,
            pitch=env._cam_pitch,
            roll=0,
            upAxisIndex=2,
        )
        proj_matrix = env._p.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(env._render_width) / env._render_height,
            nearVal=0.1,
            farVal=100.0,
        )
        (_, _, px, _, _) = env._p.getCameraImage(
            width=env._render_width,
            height=env._render_height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,  # pylint: disable=c-extension-no-member
        )
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def step(self, action):
        state, _reward, done, info = self.env.step(action)
        reward = self.env.unwrapped.robot.body_xyz[0] if done else 0.0
        return state, reward, done, info

    def seed(self, seed=None):
        self.env.seed(seed)

    def novelty(self):
        return self.env.unwrapped.robot.body_xyz[0]
