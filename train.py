import tensorflow as tf
import numpy as np
import config
import fetch_data
import network as nn

tf.reset_default_graph()

net = nn.NetWork(32, param_state_size=16)
int_net = nn.IntegratorNetwork(param_state_size=16, sequence_length=30)
init = tf.initialize_all_variables()
#test_input = fetch_data.get_volume(config.data_path, 1, sequence_length=100, sequential=True)

with tf.Session() as sess:
    sess.run(init)

    for i in range(config.training_runs):
        #input = fetch_data.get_volume(config.data_path, 1)
        #loss, _= sess.run([net.loss, net.train], input)

        if i % 1 == 0:
            input_sequence, input_0 = fetch_data.get_volume(config.data_path, 1, sequential=True, sequence_length=30)

            parameter_encodings, label_encodings = sess.run([net.encoded_sdf, net.full_encoding], input_sequence)
            start_encoding = sess.run([net.full_encoding], input_0)
            integrator_feed_dict = {'label_encodings:0': label_encodings, 'parameter_encodings:0': parameter_encodings,
                                    'start_encoding:0': start_encoding}

            encodings = sess.run(int_net.loss, integrator_feed_dict)
            print(np.shape(encodings))

        #print('Loss:', loss)
        print('Training Run', i)

    #input=fetch_data.get_volume(config.data_path, sequential=True, sequence_length=30)
    #sess.run(int_net.full_encoding, input)