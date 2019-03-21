import tensorflow as tf
import numpy as np
import config
import fetch_data
import network as nn
import os
import time
import util
import matplotlib.pyplot as plt


def restore_ae(data, graph_path, grid, frame=-1):
    tf.reset_default_graph()
    graph = tf.get_default_graph()
    graph_handle = util.find_graph(graph_path)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        graph_handle.restore(sess, tf.train.latest_checkpoint(config.path_e))
        inputs = fetch_data.get_volume(config.benchmark_data, time_idx=frame, batch_size=1, scaling_factor=1)
        inputs['Train/step:0'] = 0

        v = sess.run(graph.get_tensor_by_name('Decoder/decoder:0'), feed_dict=inputs)
        util.contour(v, inputs['velocity:0'])






