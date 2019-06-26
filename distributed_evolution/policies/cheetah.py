#Copyright (c) 2019 Uber Technologies, Inc.
#
#Licensed under the Uber Non-Commercial License (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at the root directory of this project. 
#
#See the License for the specific language governing permissions and
#limitations under the License.
import numpy as np
import tensorflow as tf

from .tf import TFPolicy


class CheetahPolicy(TFPolicy):
    needs_stats = True

    def __init__(
        self,
        ob_space_shape,
        num_actions,
        action_low,
        action_high,
        ac_noise_std=0.01,
        hidden_dims=(256, 256),
        gpu_mem_frac=0.2,
        single_threaded=False,
        theta=None,
        seed=42,
    ):
        self.ob_space_shape = ob_space_shape
        self.num_actions = num_actions
        self.action_low = action_low
        self.action_high = action_high
        self.ac_noise_std = ac_noise_std
        self.hidden_dims = hidden_dims
        self.gpu_mem_frac = gpu_mem_frac

        assert len(ob_space_shape) == 1
        assert np.all(np.isfinite(action_low)) and np.all(
            np.isfinite(action_high)
        ), "Action bounds required"

        self.nonlin = tf.tanh

        self.random_state = np.random.RandomState(seed)

        print('Creating policy...')
        tf.reset_default_graph()
        graph = tf.Graph()
        with graph.as_default():
            np.random.seed(seed)
            tf.set_random_seed(seed)
            with tf.variable_scope(type(self).__name__) as scope:
                # Observation normalization
                ob_mean = tf.get_variable(
                    "ob_mean",
                    ob_space_shape,
                    tf.float32,
                    tf.constant_initializer(np.nan),
                    trainable=False,
                )
                ob_std = tf.get_variable(
                    "ob_std",
                    ob_space_shape,
                    tf.float32,
                    tf.constant_initializer(np.nan),
                    trainable=False,
                )
                self.in_mean = tf.placeholder(tf.float32, ob_space_shape)
                self.in_std = tf.placeholder(tf.float32, ob_space_shape)

                self.assign_ob_stats = [
                    tf.assign(ob_mean, self.in_mean),
                    tf.assign(ob_std, self.in_std),
                ]

                # Policy network
                self.observation = tf.placeholder(
                    tf.float32, [None] + list(ob_space_shape)
                )
                self.action = self._make_net(
                    tf.clip_by_value((self.observation - ob_mean) / ob_std, -5.0, 5.0)
                )

        super().__init__(
            graph,
            scope,
            gpu_mem_frac=gpu_mem_frac,
            single_threaded=single_threaded,
            theta=theta,
        )

    def seed(self, seed):
        self.random_state.seed(seed)

    def set_stats(self, ob_mean, ob_std):
        self.sess.run(
            self.assign_ob_stats, feed_dict={self.in_mean: ob_mean, self.in_std: ob_std}
        )

    def _make_net(self, o):
        x = o
        for ilayer, hd in enumerate(self.hidden_dims):
            x = self.nonlin(dense(x, hd, "l{}".format(ilayer), normc_initializer(1.0)))

        a = dense(x, self.num_actions, "out", normc_initializer(0.01))

        return a

    def act(self, states):
        a = self.sess.run(self.action, feed_dict={self.observation: states})
        if self.ac_noise_std != 0:
            a += self.random_state.randn(*a.shape) * self.ac_noise_std
        return a

    def serialize(self):
        # NOTE: not serializing theta; needs to be passed separately
        return super()._serialize(
            self.ob_space_shape,
            self.num_actions,
            self.action_low,
            self.action_high,
            self.ac_noise_std,
            self.hidden_dims,
            self.gpu_mem_frac,
        )


def _normalize(x, std):
    def py_func_init(out):
        shape = out.shape
        out = np.reshape(out, [-1, shape[-1]])
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        out = np.reshape(out, shape)
        return out

    return x.assign(tf.py_func(py_func_init, [x], tf.float32))


def dense(x, size, name, weight_init=None, bias=True, std=1.0):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=weight_init)

    w.reinitialize = _normalize(w, std=std)

    ret = tf.matmul(x, w)
    if bias:
        b = tf.get_variable(name + "/b", [size], initializer=tf.zeros_initializer)

        b.reinitialize = b.assign(tf.zeros_like(b))
        # b = tf.Print(b, [b, w], name + 'last_bias,w=' )
        return ret + b
    return ret


def normc_initializer(std=1.0):
    def _initializer(
        shape, dtype=None, partition_info=None
    ):  # pylint: disable=unused-argument
        def py_func_init():
            out = np.random.randn(np.prod(shape[:-1]), shape[-1]).astype(np.float32)
            out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
            out = np.reshape(out, shape)
            return out

        result = tf.py_func(py_func_init, [], tf.float32)
        result.set_shape(shape)
        return result

    return _initializer
