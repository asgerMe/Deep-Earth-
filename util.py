import config
import fetch_data
import os
import imageio
import numpy as np
import tensorflow as tf


def cosine_annealing(step, max_step, lr_min, lr_max):
    g_lr = tf.Variable(lr_max, name='g_lr')
    g_lr_update = tf.assign(g_lr, lr_min + 0.5 * (lr_max - lr_min) * (
                tf.cos(tf.cast(step, tf.float32) * np.pi / max_step) + 1), name='g_lr_update')
    return g_lr_update


def trilinear_interpolation_kernel():
    kernel = np.asarray(
        [[[1.0 / 32, 1.0 / 16, 1.0 / 32], [1.0 / 16, 1.0 / 8, 1.0 / 16], [1.0 / 32, 1.0 / 16, 1.0 / 32]],
         [[1.0 / 16, 1.0 / 8, 1.0 / 16], [1.0 / 8, 1.0 / 4, 1.0 / 8], [1.0 / 16, 1.0 / 8, 1.0 / 16]],
         [[1.0 / 32, 1.0 / 16, 1.0 / 32], [1.0 / 16, 1.0 / 8, 1.0 / 16], [1.0 / 32, 1.0 / 16, 1.0 / 32]]])

    kernel = np.expand_dims(kernel, axis=3)
    kernel = np.expand_dims(kernel, axis=4)
    kernel = tf.constant(kernel, dtype=tf.float32)

    return kernel


def create_gif(sess, net, i = 0):
    if not i % 1000 and i > 100:
        search_dir = config.output_dir
        if os.path.isdir(config.alt_dir):
            search_dir = config.alt_dir
        try:
            video_length = 2000
            MOVIE = []

            for F in range(video_length):
                try:
                    test_input = fetch_data.get_volume(search_dir, batch_size=1, time_idx=F, scaling_factor=SCF)
                except IndexError:
                    print('index out of range -> creating gif with stashed frames')
                    break

                reconstructed_vel = sess.run(net.y, feed_dict=test_input)
                image = np.squeeze(reconstructed_vel[0, :, 16, :, :])
                image = np.uint8(255 * (image - np.min(image)) / np.max(image - np.min(image)))
                MOVIE.append(image)

            path = os.path.join(config.output_dir, 'test_vel_field_' + str(i) + '.gif')
            imageio.mimwrite(path, MOVIE)
            print('gif saved at:', os.path.join(config.output_dir, 'test_vel_field_' + str(i) + '.gif'))

        except OSError:
            print('No valid .npy test file in output dir / alternative dir not found')

