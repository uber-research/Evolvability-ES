#Copyright (c) 2019 Uber Technologies, Inc.
#
#Licensed under the Uber Non-Commercial License (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at the root directory of this project. 
#
#See the License for the specific language governing permissions and
#limitations under the License.
# import logging

# import dill as pickle
# import numpy as np
# from distributed.protocol.serialize import register_serialization

# from .core import Niche

# logger = logging.getLogger(__name__)


# class MultiGymNiche(Niche):
#     def __init__(self, make_envs, model, ts_limit):
#         self.make_envs = make_envs
#         self.envs = make_envs()
#         self.model = model
#         self.ts_limit = ts_limit
#         self.random_state = np.random.RandomState()

#     def seed(self, seed):
#         self.random_state.seed(seed)
#         for env in self.envs:
#             env.seed(self.random_state.randint(np.int32(2 ** 31 - 1)))
#         self.model.seed(seed)

#     def rollout(self, theta, eval_mode=False, render=False):
#         self.model.set_theta(theta)

#         fitnesses = np.zeros(len(self.envs))
#         states = [env.reset() for env in self.envs]
#         dones = [False] * len(self.envs)
#         self.model.reset()

#         n_steps = 0
#         for _ in range(self.ts_limit):
#             actions = self.model.act(states)
#             assert len(actions) == len(self.envs)
#             for i, (env, action, done) in enumerate(zip(self.envs, actions, dones)):
#                 if not done:
#                     n_steps += 1
#                     if render:
#                         env.render()
#                     state, reward, done, _ = env.step(action)
#                     fitnesses[i] += reward
#                     if not done:
#                         states[i] = state
#                     dones[i] = done
#             if all(dones):
#                 break

#         return sum(fitnesses) / len(fitnesses), n_steps / len(self.envs)

#     def serialize(self):
#         niche_frame = pickle.dumps(
#             (self.make_envs, self.ts_limit, self.model.__class__)
#         )
#         model_frames = self.model.serialize()
#         return {}, [niche_frame] + model_frames

#     @staticmethod
#     def deserialize(_header, frames):
#         tup = pickle.loads(frames[0])

#         make_envs = tup[0]
#         ts_limit = tup[1]
#         Model = tup[2]

#         model = Model.deserialize(frames[1:])

#         return MultiGymNiche(make_envs, model, ts_limit)


# register_serialization(
#     MultiGymNiche, MultiGymNiche.serialize, MultiGymNiche.deserialize
# )
