import config
import fetch_data
import os
import imageio
import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

def find_graph(path):
    if not len(os.listdir(path)):
        print('No valid graph files')

    for file in os.listdir(path):
        print(file)
        if file.endswith('.ckpt.meta'):
            try:
                graph_handle = tf.train.import_meta_graph(os.path.join(config.path_e, file))
                print('Importing graph')
                return graph_handle

            except IOError:
                print('Cant import graph')
                return 0




def cosine_annealing(step, max_step, lr_min, lr_max):
    g_lr = tf.Variable(lr_max, name='g_lr')
    g_lr_update = tf.assign(g_lr, lr_min + 0.5 * (lr_max - lr_min) * (
                tf.cos(tf.cast(step, tf.float32) * np.pi / max_step) + 1), name='g_lr_update')
    return g_lr_update


def get_multihot():
    grid_dict = ''
    index = 0
    values = 0
    dense_shape = 0

    if True:
        files = os.listdir(config.grid_dir)

        print('Searching for grid in', config.grid_dir)
        for file in files:
            if file.endswith('.npz'):
                print('Found grid:', file)
                grid_dict = np.load(os.path.join(config.grid_dir, file))
                break

        if grid_dict == '':
            print('No grid file found in:', config.grid_dir,
                 '... Supply grid file .npz here or change to dir with valid grid using -g')
            exit()

        index = []
        values = []
        dense_shape = [np.int64(pow(config.data_size, 3)), np.shape(grid_dict['prim_points'])[0]]

        c = 0

        for i in grid_dict['point_prims']:
            linear_index = grid_dict['linear_index'][c]
            pt_volume = grid_dict['point_volume'][c]
            for j in i:
                index.append([linear_index, j])
                values.append(np.float32(1.0/pt_volume))
            c += 1

    return np.array(index, dtype=np.int64), np.array(values, dtype=np.float32), np.array(dense_shape, dtype=np.int64), grid_dict

def differential_kernel():
        kernel = np.asarray([[[0, 1, 0], [1, 1, 1], [0, 1, 0]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[0, 1, 0], [1, 1, 1], [0, 1, 0]]])
        kernel = np.expand_dims(kernel, axis=3)
        kernel = np.expand_dims(kernel, axis=4)
        kernel = tf.constant(kernel, dtype=tf.float32)

        return kernel

def trilinear_interpolation_kernel():
    kernel = np.asarray(
        [[[1.0 / 32, 1.0 / 16, 1.0 / 32], [1.0 / 16, 1.0 / 8, 1.0 / 16], [1.0 / 32, 1.0 / 16, 1.0 / 32]],
         [[1.0 / 16, 1.0 / 8, 1.0 / 16], [1.0 / 8, 1.0 / 4, 1.0 / 8], [1.0 / 16, 1.0 / 8, 1.0 / 16]],
         [[1.0 / 32, 1.0 / 16, 1.0 / 32], [1.0 / 16, 1.0 / 8, 1.0 / 16], [1.0 / 32, 1.0 / 16, 1.0 / 32]]])

    kernel = np.expand_dims(kernel, axis=3)
    kernel = np.expand_dims(kernel, axis=4)
    kernel = tf.constant(kernel, dtype=tf.float32)

    return kernel


def get_name_ext(net = 'AE'):

    name = ''
    dc = "DL"

    if config.conv:
        dc = "CL"

    diff = 'off'
    floss = 'off'

    if config.fem_loss:
        floss = 'on'

    if config.use_fem:
        diff = 'on'

    name = net + '___EncoderArc___' + dc + '_LSS_' + str(config.param_state_size)\
               + '_SBB_' + str(config.sb_blocks) + '_BFS_' + str(config.n_filters) +\
               '_FEMCONV_' + diff + '_FEMLoss_' + floss

    return str(name)

def create_gif_encoder(path, sess, net, i = 0, gif_length=2000, save_frequency = 5000, SCF = 1, restore=False):
    viridis = cm.get_cmap('inferno', 12)
    if not i % save_frequency:
        search_dir = path
        try:
            MOVIE = []
            diff_MOVIE = []
            print('Generating gif')
            for F in range(gif_length):
                try:
                    test_input = fetch_data.get_volume(search_dir, batch_size=1, time_idx=F, scaling_factor=SCF)
                    if not test_input:
                        break
                except IndexError:
                    print('index out of range -> creating gif with stashed frames')
                    break

                network = net.y
                diff = net.d_labels

                reconstructed_vel = sess.run(network, feed_dict=test_input)
                image = np.linalg.norm(np.squeeze(reconstructed_vel[0, :, 16, :, :]), axis=2)
                image_gt = np.linalg.norm(np.squeeze(test_input['velocity:0'][0, :, 16, :, :]), axis=2)
                image_diffs = sess.run(diff, feed_dict=test_input)

                test_input['velocity:0'] *= 0

                reconstructed_vel0 = sess.run(network, feed_dict=test_input)
                image0 = np.linalg.norm(np.squeeze(reconstructed_vel0[0, :, 16, :, :]), axis=2)

                full_image = np.concatenate((image_gt, image, image0), axis=1)
                diff_image = []
                for diffs_num in range(np.shape(image_diffs)[4]):

                    diff = np.squeeze(image_diffs[0, :, 16, :, diffs_num])

                    if np.amax(diff) > 0:
                        diff = (np.uint8(255*viridis((diff - np.amin(diff)) / (np.amax(diff) - np.amin(diff)))))
                    else:
                        diff = viridis(np.uint8(diff))

                    if diffs_num > 0:
                        diff_image = np.abs(np.concatenate((diff_image, diff), axis=1))
                    else:
                        diff_image = np.abs(diff)


                MOVIE.append(full_image)
                diff_MOVIE.append(diff_image)



            path = os.path.join(config.gif_path, 'test_vel_field_' + get_name_ext() + str(i) + '.gif')
            MOVIE = np.uint8(255 * viridis( (MOVIE - np.amin(MOVIE)) / (np.amax(MOVIE) - np.amin(MOVIE))))
            imageio.mimwrite(path, MOVIE)

            diff_path = os.path.join(config.gif_path, 'diff_vel_field_' + get_name_ext() +  str(i) + '.gif')

            imageio.mimwrite(diff_path, diff_MOVIE)

            print('gif saved at:', os.path.join(config.gif_path, 'test_vel_field_' + get_name_ext() + str(i) + '.gif'))

        except OSError:
            print('No valid .npy test file in output dir / alternative dir not found')

def create_gif_integrator(sess, net, autoencoder_graph, roll_out, i = 0, gif_length=500, save_frequency = 5000, SCF = 1, restore=False):

    if not i % save_frequency and i > 1:
        search_dir = config.data_path
        if os.path.isdir(config.alt_dir):
            search_dir = config.alt_dir
        try:
            video_length = gif_length
            MOVIE = []

            sdf = autoencoder_graph.get_tensor_by_name("Boundary_conditions/encoded_sdf:0")
            full_encoding = autoencoder_graph.get_tensor_by_name("Latent_State/full_encoding:0")
            reconstructed_v = autoencoder_graph.get_tensor_by_name("Decoder/decoder:0")
            v_next = ''
            next_encoding = 0
            for F in range(gif_length):

                try:
                    input_i = fetch_data.get_volume(search_dir, batch_size=1, time_idx=F, scaling_factor=SCF)
                except IndexError:
                    print('index out of range -> creating gif with stashed frames')
                    break

                if F == 0:
                    full_enc = sess.run(full_encoding, feed_dict=input_i)
                else:
                    full_enc = next_encoding
                try:
                    input_next = fetch_data.get_volume(search_dir, batch_size=1, time_idx=(1+F), scaling_factor=SCF)
                except IndexError:
                    print('index out of range -> creating gif with stashed frames')
                    break

                next_encoded_sdf = sess.run(sdf, feed_dict=input_next)

                integrator_feed_dict = {'parameter_encodings:0': next_encoded_sdf,  'start_encoding:0': [full_enc], "sequence_length:0": 1}

                next_encoding = sess.run(net.full_encoding, feed_dict=integrator_feed_dict)
                

                v_next = sess.run(reconstructed_v,  feed_dict={'sdf:0': input_next['sdf:0'], 'Latent_State/full_encoding:0': np.squeeze(next_encoding, axis=1)})

                image = np.linalg.norm(np.squeeze(v_next[0, :, 16, :, :]), axis=2)
                MOVIE.append(image)

            path = os.path.join(config.gif_path, 'integrator_vel_field_' + str(i) + '.gif')
            MOVIE = np.uint8(255 * (MOVIE - np.amin(MOVIE)) / (np.amax(MOVIE) - np.amin(MOVIE)))
            imageio.mimwrite(path, MOVIE)

            print('gif saved at:', os.path.join(config.gif_path, 'integrator_vel_field_' +str(i)+ '.gif'))
        except OSError:
            print('No valid .npy test file in output dir / alternative dir not found')



def create_dirs(clear=False):
    if (os.path.isdir(config.data_path)):
        output_path = os.path.join(config.data_path, 'network_output')

        grid_path = os.path.join(config.data_path, 'grid')
        config.grid_dir = grid_path
        graphs_path = os.path.join(output_path, 'graphs')

        integrator_graph = os.path.join(graphs_path, 'integrator')
        autoencoder_graph = os.path.join(graphs_path, 'autoencoder')
        test_fields = os.path.join(output_path, 'fields')
        gifs = os.path.join(output_path, 'gifs')

        tensorboard_path = os.path.join(output_path, 'tensorboard')
        bmd = os.path.join(output_path, 'bench_mark_data')

        if not os.path.isdir(output_path):
            print('Creating Folders')
            os.mkdir(output_path)
        config.output_dir = output_path

        if not os.path.isdir(bmd):
            print('Creating Folders')
            os.mkdir(bmd)
        config.benchmark_data = bmd



        if not os.path.isdir(grid_path):
            print('creating', grid_path)
            os.mkdir(grid_path)
        config.grid_dir = grid_path

        if not os.path.isdir(graphs_path):
            print('creating', graphs_path)
            os.mkdir(graphs_path)

        if not os.path.isdir(integrator_graph):
            print('creating', integrator_graph)
            os.mkdir(integrator_graph)
        config.path_i = integrator_graph

        if not os.path.isdir(autoencoder_graph):
            print('creating', autoencoder_graph)
            os.mkdir(autoencoder_graph)
        config.path_e = autoencoder_graph

        if not os.path.isdir(test_fields):
            print('creating', test_fields)
            os.mkdir(test_fields)
        config.test_field_path = test_fields

        if not os.path.isdir(gifs):
            print('creating', gifs)
            os.mkdir(gifs)
        config.gif_path = gifs

        if not os.path.isdir(tensorboard_path):
            print('creating', tensorboard_path)
            os.mkdir(tensorboard_path)
        config.tensor_board = tensorboard_path

