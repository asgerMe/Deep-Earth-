import tensorflow as tf
import numpy as np
import config
import os
import fetch_data
import matplotlib.pyplot as plt
from matplotlib import animation
import imageio

def deploy_network():
    tf.reset_default_graph()
    feed_dict = fetch_data.get_volume(config.data_path, 1)

    for file in os.listdir(config.meta_graphs):
        print(file)
        if file.endswith('.ckpt.meta'):
            try:
                graph_handle = tf.train.import_meta_graph(os.path.join(config.meta_graphs, file))
                print(graph_handle)
                break
            except IOError:
                print('Cant import graph')
                exit()


    with tf.Session() as sess:
        graph = tf.get_default_graph()
        sub_path = "./weights/"

        graph_handle.restore(sess, tf.train.latest_checkpoint(config.meta_graphs))
        print('Model restored')
        search_dir = ''

        if os.path.isdir(config.alt_dir):
            search_dir = config.alt_dir
        else:
            search_dir = config.data_path

        input_sequence, input_0 = fetch_data.get_volume(search_dir, 1, sequential=True,
                                                        sequence_length=config.sequence_length,
                                                        inference=True)


        full_encoding = graph.get_tensor_by_name("Latent_State/full_encoding:0")
        encoded_sdf = graph.get_tensor_by_name("Boundary_conditions/encoded_sdf:0")
        start_encoding = sess.run([full_encoding], input_0)
        parameter_encodings, label_encodings = sess.run([encoded_sdf, full_encoding], input_sequence)

        integrator_feed_dict = {'label_encodings:0': label_encodings, 'parameter_encodings:0': parameter_encodings,
                                'start_encoding:0': start_encoding, 'phase:0': True,
                                "sequence_length:0": config.sequence_length}



        roll_out = graph.get_tensor_by_name("Integrater_Network/next_encoding:0")
        encoded_states = sess.run(roll_out, integrator_feed_dict)
        decoder = graph.get_tensor_by_name("Decoder/decoder:0")
        field_sequence = sess.run(decoder, feed_dict={'sdf:0': input_sequence['sdf:0'], 'Latent_State/full_encoding:0': np.squeeze(encoded_states)})
        gt_sequence = sess.run(decoder, feed_dict={'sdf:0': input_sequence['sdf:0'], 'Latent_State/full_encoding:0': np.squeeze(label_encodings)})

        plt.close('all')
        images = []
        vf_seq = input_sequence['velocity:0']

        for i in range(np.shape(field_sequence)[0]):
            rec = np.linalg.norm(np.squeeze(field_sequence[i, :, 16, :, :]), axis=2)
            gt = np.linalg.norm(np.squeeze(gt_sequence[i, :, 16, :, :]), axis=2)
            vf = np.linalg.norm(np.squeeze(vf_seq[i, :, 16, :, :]), axis=2)
            error = np.abs(rec - gt)

            stacked = np.uint8(255.0*np.concatenate(((rec - np.min(rec))/np.max(rec - np.min(rec)), (gt - np.min(gt))/np.max(gt - np.min(gt)), (vf - np.min(vf))/np.max(vf - np.min(vf)), (error - np.min(error))/np.max(error - np.min(error))), axis=1))

            images.append(stacked)

        print('gif saved at:', os.path.join(config.output_dir, 'field_evo.gif'))
        path = os.path.join(config.output_dir, 'field_evo.gif')
        imageio.mimsave(path, images)



