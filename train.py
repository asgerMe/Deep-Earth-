import tensorflow as tf
import numpy as np
import config
import fetch_data
import network as nn
import os

import helper_functions as hf

def train_network():

    tf.reset_default_graph()

    net = nn.NetWork(32, param_state_size=config.param_state_size)

    if config.train_integrator_network:
        int_net = nn.IntegratorNetwork(param_state_size=config.param_state_size, sequence_length=30)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        writer = tf.summary.FileWriter(config.tensor_board)

        writer.add_graph(sess.graph)
        saver = tf.train.Saver(tf.global_variables())

        store_integrator_loss = 0

        for i in range(config.training_runs):
            inputs = fetch_data.get_volume(config.data_path, 1)
            loss, _ = sess.run([net.loss, net.train], inputs)

            if i % config.f_integrator_network == 0 and config.train_integrator_network:
                input_sequence, input_0 = fetch_data.get_volume(config.data_path, 1, sequential=True, sequence_length=30)

                parameter_encodings, label_encodings = sess.run([net.encoded_sdf, net.full_encoding], input_sequence)
                start_encoding = sess.run([net.full_encoding], input_0)
                integrator_feed_dict = {'label_encodings:0': label_encodings, 'parameter_encodings:0': parameter_encodings,
                                        'start_encoding:0': start_encoding, 'phase:0': True}

                _, loss, merged_int = sess.run([int_net.train_int, int_net.loss_int, int_net.merged_int], integrator_feed_dict)
                store_integrator_loss = merged_int
                print('Integrator Loss', loss)

            if config.save_freq:
                if config.meta_graphs and i % config.save_freq == 0 and config.save_freq != 0 and config.save_freq != 0:
                    saver.save(sess, os.path.join(config.meta_graphs, "trained_model.ckpt"))
                    print('Saving graph')

            if config.f_tensorboard:
                if i % config.f_tensorboard == 0 and config.f_tensorboard != 0:
                    merged = sess.run(net.merged, inputs)
                    writer.add_summary(merged, i)
                    writer.add_summary(store_integrator_loss, i)

                    print('Saving tensorboard')

            if not i % 10:
                print('Encoder Loss:', loss)
                print('Training Run', i)
