import tensorflow as tf
import numpy as np
import config
import fetch_data
import network as nn
import os
import time
import util


def train_network():
    tf.reset_default_graph()
    net = nn.NetWork(config.data_size, param_state_size=config.param_state_size)

    init = tf.global_variables_initializer()
    SCF = 1#fetch_data.get_scaling_factor(config.data_path)

    with tf.Session() as sess:
        sess.run(init)

        sub_dir = os.path.join(config.tensor_board, time.strftime("%Y%m%d-%H%M%S"))
        sub_dir_test = os.path.join(config.tensor_board, 'test'+time.strftime("%Y%m%d-%H%M%S"))
        os.mkdir(sub_dir)
        writer = tf.summary.FileWriter(sub_dir)
        writer_test = tf.summary.FileWriter(sub_dir_test)
        writer.add_graph(sess.graph)

        saver = tf.train.Saver(tf.global_variables())
        store_integrator_loss_tb = 0
        store_integrator_loss = -1

        for i in range(config.training_runs):

            inputs = fetch_data.get_volume(config.data_path, batch_size=config.batch_size, scaling_factor=SCF)
            inputs['Train/step:0'] = i

            if i % config.f_tensorboard == 0 and config.f_tensorboard != 0 and os.path.isdir(config.tensor_board):
                loss, lr, merged, _ = sess.run([net.loss, net.lr, net.merged, net.train], inputs)
                writer.add_summary(merged, i)
            else:
                loss, lr, _ = sess.run([net.loss, net.lr, net.train], inputs)


            if os.path.isdir(config.path_e) and i % config.save_freq == 0:
                saver.save(sess, os.path.join(config.path_e, time.strftime("%Y%m%d-%H%M%S") + "_trained_model.ckpt"))
                print('Saving graph')

            if i % 500 == 0 and os.path.isdir(config.tensor_board):
                inputs_ci = fetch_data.get_volume('D:\output\circular_ice', 1, scaling_factor=SCF)
                inputs_ci['Train/step:0'] = i

                loss, merged = sess.run([net.loss, net.merged], inputs_ci)
                writer_test.add_summary(merged, i)
                test_field = sess.run(net.y, inputs_ci)
                np.save(os.path.join(config.test_field_path, 'train_field' + time.strftime("%Y%m%d-%H%M%S")), test_field)

                # Record execution stats
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                _ = sess.run([net.train],
                                      feed_dict=inputs,
                                      options=run_options,
                                      run_metadata=run_metadata)
                writer.add_run_metadata(run_metadata, 'step%d' % i)

            if not i % 10:
                print('Training Run', i, 'Learning Rate', lr,'//  Encoder Loss:', loss, '//  Integrator Loss', store_integrator_loss)

            util.create_gif_encoder("D:/output/circular_ice/", sess, net, i=i, save_frequency=config.save_gif, SCF=SCF)


def train_integrator():
    tf.reset_default_graph()
    graph = tf.get_default_graph()
    graph_handle = 0
    int_net = 0

    if config.train_integrator_network:
        if not config.conv:
            int_net = nn.IntegratorNetwork(param_state_size=config.param_state_size, sdf_state_size=config.sdf_state)
        else:
            int_net = nn.Convo_IntegratorNetwork(config.data_size,param_state_size= config.param_state_size, sdf_state_size=config.sdf_state)

    init = tf.global_variables_initializer()

    SCF = 1#fetch_data.get_scaling_factor(config.data_path)

    for file in os.listdir(config.path_e):
        print(file)
        if file.endswith('.ckpt.meta'):
            try:
                graph_handle = tf.train.import_meta_graph(os.path.join(config.path_e, file))
                print(graph)
                break
            except IOError:
                print('Cant import graph')
                exit()

    with tf.Session() as sess:
        sess.run(init)

        sub_dir = os.path.join(config.tensor_board, 'integrator_' + time.strftime("%Y%m%d-%H%M%S"))
        os.mkdir(sub_dir)
        writer = tf.summary.FileWriter(sub_dir)
        writer.add_graph(sess.graph)

        saver = tf.train.Saver(tf.global_variables())
        store_integrator_loss_tb = 0
        store_integrator_loss = -1

        graph_handle.restore(sess, tf.train.latest_checkpoint(config.path_e))
        print('Model restored')

        full_encoding = graph.get_tensor_by_name("Latent_State/full_encoding:0")
        encoded_sdf = graph.get_tensor_by_name("Boundary_conditions/encoded_sdf:0")


        for i in range(config.training_runs):
            input_sequence, input_0 = fetch_data.get_volume(config.data_path, 1, sequential=True, sequence_length=config.sequence_length)

            start_encoding = sess.run([full_encoding], input_0)
            parameter_encodings, label_encodings = sess.run([encoded_sdf, full_encoding], input_sequence)

            integrator_feed_dict = {'label_encodings:0': label_encodings,
                                    'parameter_encodings:0': parameter_encodings,
                                    'start_encoding:0': start_encoding, 'phase:0': True,
                                    "sequence_length:0": config.sequence_length}


            if i % config.f_tensorboard == 0 and config.f_tensorboard != 0 and os.path.isdir(config.tensor_board):
                _, int_loss, merged_int = sess.run([int_net.train_int, int_net.loss_int, int_net.merged_int], integrator_feed_dict)
                writer.add_summary(merged_int, i)
            else:
                _, int_loss = sess.run([int_net.train_int, int_net.loss_int], integrator_feed_dict)


            if not i % 10:
                print('Training Run', i, 'Learning Rate', config.lr_max, '//  Encoder Loss:', -1, '//  Integrator Loss', int_loss)

            if config.save_freq and os.path.isdir(config.path_i):
                if config.meta_graphs and i % config.save_freq == 0 and config.save_freq > 2:
                    saver.save(sess, os.path.join(config.path_i, "trained_integrator_model.ckpt"))
                    print('Saving graph')

            util.create_gif_integrator(sess, int_net, graph, roll_out = 2000, i=i, save_frequency=config.save_gif, SCF=SCF)