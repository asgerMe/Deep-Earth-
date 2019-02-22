import tensorflow as tf
import numpy as np
import config
import os
import fetch_data


def create_sequence(graph_path=config.save_path, roll_out_length=0):
    tf.reset_default_graph()
    feed_dict = fetch_data.get_volume(config.data_path, 1)

    encoder = config.encoder_name + '.meta'
    integrator = ''
    if roll_out_length != 0:
        integrator = config.integrator_name + '.meta'

    path_encoder = os.path.join(graph_path, encoder)
    path_integrator = os.path.join(graph_path, integrator)

    try:
        saver_encoder = tf.train.import_meta_graph(path_encoder)
    except:
        print('No meta graph in path')
        exit()


    #with tf.Session() as sess:
    #    graph = tf.get_default_graph()
    #    sub_path = "./weights/"

    #    saver_encoder.restore(sess, tf.train.latest_checkpoint(sub_path))
    #    print('model restored')

    #    placeholder = graph.get_tensor_by_name("inputframe:0")
    #    phase = graph.get_tensor_by_name("phase:0")
    #    op_to_restore = graph.get_tensor_by_name("output:0")

    #    test_batch = [data_in]

    #    for i in range(future_roll_out_length):

    #        feed_dict = {placeholder: test_batch, phase: [0]}
    #        case = sess.run(op_to_restore, feed_dict=feed_dict)
    #        case_stack = np.transpose(case,[0,3,1,2])

    #        test_batch = np.hstack([test_batch, case_stack])
    #        test_batch = [test_batch[0][1:][:][:]]

    #        # Displaying for later
    #        case = np.squeeze(case)

    #        span_x = 128 - gen_frame_x
    #        span_y = 128 - gen_frame_y

    #        pad_xmin = int(np.floor(span_x / 2))
    #        pad_xmax = int(np.ceil(span_x / 2))

    #        pad_ymin = int(np.floor(span_y / 2))
    #        pad_ymax = int(np.ceil(span_y / 2))

    #        tfb = case[pad_xmin:(pad_xmin + np.shape(lat_f)[0]),pad_ymin:(pad_ymin +np.shape(lat_f)[1])]
    #        predictions[i][:][:] = tfb

    #    return predictions

