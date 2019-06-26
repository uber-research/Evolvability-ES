#Copyright (c) 2019 Uber Technologies, Inc.
#
#Licensed under the Uber Non-Commercial License (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at the root directory of this project. 
#
#See the License for the specific language governing permissions and
#limitations under the License.
import logging

import numpy as np
import tensorflow as tf

from .core import Policy

logger = logging.getLogger(__name__)


def var_shape(x):
    out = [k.value for k in x.get_shape()]
    assert all(
        isinstance(a, int) for a in out
    ), "shape function assumes that shape is fully known"
    return out


def numel(x):
    return int(np.prod((var_shape(x))))


class SetFromFlat(object):
    def __init__(self, var_list, dtype=tf.float32):
        assigns = []
        shapes = list(map(var_shape, var_list))
        total_size = np.sum([int(np.prod((shape))) for shape in shapes])

        self.theta = theta = tf.placeholder(dtype, [total_size])
        start = 0
        assigns = []
        for (shape, v) in zip(shapes, var_list):
            size = int(np.prod((shape)))
            assigns.append(tf.assign(v, tf.reshape(theta[start : start + size], shape)))
            start += size
        assert start == total_size
        self.op = tf.group(*assigns)

    def __call__(self, sess, theta):
        sess.run(self.op, feed_dict={self.theta: theta})


class GetFlat(object):
    def __init__(self, var_list):
        self.op = tf.concat([tf.reshape(v, [numel(v)]) for v in var_list], 0)

    def __call__(self, sess):
        return sess.run(self.op)


class TFPolicy(Policy):
    def __init__(
        self, graph, scope, gpu_mem_frac=0.2, single_threaded=False, theta=None
    ):
        # np.random.seed(seed)
        self.graph = graph
        self.scope = scope
        with graph.as_default():
            self.all_variables = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, self.scope.name
            )

            self.trainable_variables = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, self.scope.name
            )
            self.num_params = sum(
                int(np.prod(v.get_shape().as_list())) for v in self.trainable_variables
            )
            self._setfromflat = SetFromFlat(self.trainable_variables)
            self._getflat = GetFlat(self.trainable_variables)

            # self._setallfromflat = SetFromFlat(self.all_variables)
            # self._getallflat = GetFlat(self.all_variables)

        logger.info("Trainable variables ({} parameters)".format(self.num_params))
        for v in self.trainable_variables:
            shp = v.get_shape().as_list()
            logger.info("- {} shape:{} size:{}".format(v.name, shp, np.prod(shp)))
        logger.info("All variables")
        for v in self.all_variables:
            shp = v.get_shape().as_list()
            logger.info("- {} shape:{} size:{}".format(v.name, shp, np.prod(shp)))

        config = tf.ConfigProto()
        # pylint: disable=no-member
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = gpu_mem_frac
        if single_threaded:
            config.inter_op_parallelism_threads = 1
            config.intra_op_parallelism_threads = 1
        self.sess = tf.Session(graph=graph, config=config)
        self.initialize()
        if theta is not None:
            self.set_theta(theta)

    def act(self, states):
        raise NotImplementedError()

    def initialize(self):
        for v in self.all_variables:
            self.sess.run(v.initializer)

    def set_theta(self, theta):
        self._setfromflat(self.sess, theta)

    def get_theta(self):
        return self._getflat(self.sess)
