#Copyright (c) 2019 Uber Technologies, Inc.
#
#Licensed under the Uber Non-Commercial License (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at the root directory of this project. 
#
#See the License for the specific language governing permissions and
#limitations under the License.
import pickle


class Policy:
    needs_stats = False

    def seed(self, seed):
        pass

    def act(self, states):
        raise NotImplementedError()

    def reset(self):
        pass

    def supp_fitness(self):  # pylint: disable=no-self-use
        return 0.0

    def set_theta(self, theta):
        raise NotImplementedError()

    def _serialize(self, *args, **kwargs):  # pylint: disable=no-self-use
        frame = pickle.dumps((args, kwargs))
        return [frame]

    @classmethod
    def deserialize(cls, frames):
        args, kwargs = pickle.loads(frames[0])
        return cls(*args, **kwargs)
