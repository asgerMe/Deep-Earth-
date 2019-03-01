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



def create_gif_encoder(sess, net, i = 0, gif_length=2000, save_frequency = 5000, SCF = 1):
    if not i % save_frequency:
        search_dir = config.data_path
        if os.path.isdir(config.alt_dir):
            search_dir = config.alt_dir
        try:
            video_length = gif_length
            MOVIE = []

            for F in range(video_length):
                try:
                    test_input = fetch_data.get_volume(search_dir, batch_size=1, time_idx=F, scaling_factor=SCF)
                except IndexError:
                    print('index out of range -> creating gif with stashed frames')
                    break

                reconstructed_vel = sess.run(net.y, feed_dict=test_input)
                image = np.linalg.norm(np.squeeze(reconstructed_vel[0, :, 16, :, :]), axis=2)
                image_gt = np.linalg.norm(np.squeeze(test_input['velocity:0'][0, :, 16, :, :]), axis=2)

                full_image = np.concatenate((image_gt, image), axis=1)
                MOVIE.append(full_image)

            path = os.path.join(config.output_dir, 'test_vel_field_' + str(i) + '.gif')
            MOVIE = np.uint8(255 * (MOVIE - np.amin(MOVIE)) / (np.amax(MOVIE) - np.amin(MOVIE)))
            imageio.mimwrite(path, MOVIE)
            print('gif saved at:', os.path.join(config.output_dir, 'test_vel_field_' + str(i) + '.gif'))

        except OSError:
            print('No valid .npy test file in output dir / alternative dir not found')



