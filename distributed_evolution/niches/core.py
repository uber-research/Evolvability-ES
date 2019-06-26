#Copyright (c) 2019 Uber Technologies, Inc.
#
#Licensed under the Uber Non-Commercial License (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at the root directory of this project. 
#
#See the License for the specific language governing permissions and
#limitations under the License.
import numpy as np


class Niche:
    def rollout_batch(self, thetas, batch_size, extra, eval_mode=False):
        returns = np.zeros(batch_size)
        lengths = np.zeros(batch_size, dtype="int")

        for i, theta in enumerate(thetas):
            returns[i], lengths[i] = self.rollout(theta, extra, eval_mode=eval_mode)

        return returns, lengths

    def update_extra(self, extra_pos, extra_neg):
        pass

    def get_extra(self):  # pylint: disable=no-self-use
        return None

    def set_theta(self, theta):
        raise NotImplementedError()

    def get_theta(self):
        raise NotImplementedError()

    def save(self, fnm):
        raise NotImplementedError()

    def load(self, fnm):
        raise NotImplementedError()

    def seed(self, seed):
        raise NotImplementedError()

    def serialize(self):
        raise NotImplementedError()

    @staticmethod
    def deserialize(_header, _frames):
        raise NotImplementedError()

    def rollout(self, theta, extra, eval_mode=False, render=False):
        raise NotImplementedError()


# def serialize(niche):
#     header, frames = niche.serialize()
#     return header, [pickle.dumps(niche.__class__)] + frames


# def deserialize(header, frames):
#     NicheClass = pickle.loads(frames[0])
#     return NicheClass.deserialize(header, frames[1:])


# register_serialization(Niche, serialize, deserialize)
