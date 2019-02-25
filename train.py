import tensorflow as tf
import numpy as np
import config
import fetch_data
import network as nn
import os
import deploy

import helper_functions as hf

def train_network():

    tf.reset_default_graph()

    net = nn.NetWork(config.data_size, param_state_size=config.param_state_size)

    if config.train_integrator_network:
        int_net = nn.IntegratorNetwork(param_state_size=config.param_state_size)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        writer = tf.summary.FileWriter(config.tensor_board)
        writer.add_graph(sess.graph)
        meta_graph_def = tf.train.export_meta_graph(filename=os.path.join(config.tensor_board, 'my-model.meta'))

        saver = tf.train.Saver(tf.global_variables())
        store_integrator_loss_tb = 0
        store_integrator_loss = -1
        for i in range(config.training_runs):
            inputs = fetch_data.get_volume(config.data_path, 1)
            loss, _ = sess.run([net.loss, net.train], inputs)

            if config.f_integrator_network:
                if i % config.f_integrator_network == 0:
                    input_sequence, input_0 = fetch_data.get_volume(config.data_path, 1, sequential=True, sequence_length=config.sequence_length)
                    parameter_encodings, label_encodings = sess.run([net.encoded_sdf, net.full_encoding], input_sequence)

                    start_encoding = sess.run([net.full_encoding], input_0)

                    integrator_feed_dict = {'label_encodings:0': label_encodings, 'parameter_encodings:0': parameter_encodings,
                                        'start_encoding:0': start_encoding, 'phase:0': True, "sequence_length:0": config.sequence_length}

                    _, int_loss, merged_int = sess.run([int_net.train_int, int_net.loss_int, int_net.merged_int], integrator_feed_dict)
                    store_integrator_loss_tb = merged_int
                    store_integrator_loss = int_loss
                    print('Training integrator')

            if config.save_freq and os.path.isdir(config.meta_graphs):
                if config.meta_graphs and i % config.save_freq == 0 and config.save_freq != 0 and config.save_freq != 0:
                    saver.save(sess, os.path.join(config.meta_graphs, "trained_model.ckpt"))
                    print('Saving graph')


            if config.f_tensorboard and os.path.isdir(config.tensor_board):
                if i % config.f_tensorboard == 0 and config.f_tensorboard != 0:
                    merged = sess.run(net.merged, inputs)
                    writer.add_summary(merged, i)
                    writer.add_summary(store_integrator_loss_tb, i)

                    print('Saving tensorboard')

            if i % (config.f_tensorboard*10) == 0 and config.f_tensorboard != 0:  # Record execution stats
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                _ = sess.run([net.train],
                                      feed_dict=inputs,
                                      options=run_options,
                                      run_metadata=run_metadata)
                writer.add_run_metadata(run_metadata, 'step%d' % i)



            if not i % 10:
                print('Training Run', i, '//  Encoder Loss:', loss, '//  Integrator Loss', store_integrator_loss)

