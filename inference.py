import tensorflow as tf
import numpy as np
import config
import fetch_data
import network as nn
import os
import time
import util
import network as nn


def restore_ae(path):
    tf.reset_default_graph()
    graph = tf.get_default_graph()
    graph_handle = util.find_graph(path)
    net = nn.NetWork(64, 8)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        graph_handle.restore(sess, tf.train.latest_checkpoint(config.path_e))
        print('Model restored')

        inputs = fetch_data.get_volume('D:\PhD\AI_FEM64', batch_size=1, scaling_factor=1)
        print(np.shape(inputs['velocity:0']))
        inputs['Train/step:0'] = 0

        v = sess.run(net.y, feed_dict=inputs)




