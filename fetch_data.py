import numpy as np
import os
import config
import matplotlib.pyplot as plt


def get_volume(data_path, batch_size=1, external_encoding=None, sequential=False, sequence_length=1):

    list_files = os.listdir(data_path)
    batch = []

    if not sequential:
        for i in range(batch_size):
            try:
                random_file_name = list_files[np.random.randint(0, np.size(list_files))]
                full_path = os.path.join(data_path, random_file_name)
            except ValueError:
                print('no suitable files in path')
                exit()

            try:
                data = np.load(full_path, mmap_mode='r+')
                n_voxels = np.shape(data)[2]
                voxel_dim = int(np.ceil(pow(n_voxels, 1 / 3)))

                dim1 = np.shape(data)[0]
                random_time_index = np.random.randint(0, dim1)

                data = data[random_time_index, :, :]
                batch.append(data)

            except IndexError:
                print('could not open file')
                continue

    if sequential:
        for i in range(batch_size):
            try:
                random_file_name = list_files[np.random.randint(0, np.size(list_files)-1)]
                full_path = os.path.join(data_path, random_file_name)
            except ValueError:
                print('no suitable files in path, at least two files needed !')
                exit()

            try:
                data = np.load(full_path, mmap_mode='r+')
                dim1 = np.shape(data)[0]
                random_time_index = np.random.randint(0, dim1)

                batch = data[random_time_index:random_time_index+sequence_length, :, :]
                padding = np.zeros([sequence_length-np.shape(batch)[0], np.shape(batch)[1], np.shape(batch)[2],
                                  np.shape(batch)[3], np.shape(batch)[4]])

                batch = np.concatenate([batch, padding], axis=0)

            except IndexError:
                print('could not open file')
                continue


    batch = np.asarray(batch)
    sdf = np.expand_dims(batch[:, 0, :, :, :], 4)
    velocity = np.moveaxis(batch[:, 1:4, :, :, :], 1, 4)


    feed_dict = {'sdf:0': sdf, 'velocity:0': velocity}
    if not sequential:
        return feed_dict
    else:
        feed_dict_0 = {'sdf:0': [sdf[0, :, :, :, :]], 'velocity:0': [velocity[0, :, :, :, :]]}
        return feed_dict, feed_dict_0


def get_random_sample(name, batch_size = 1, voxel_side_length=32):
    data_array = np.random.rand(1, voxel_side_length, voxel_side_length, voxel_side_length, 3)
    feed_dict = {name: data_array}
    return data_array

def viz_data_slice(x, field_index = 0):
    half_size = int(np.ceil(np.shape(x)[3]/2))
    n_fields = int(np.shape(x)[4])
    fidx = np.clip(int(field_index), 0, n_fields)
    d2_tensor = x[0, :, :, half_size, fidx]
    plt.imshow(d2_tensor, interpolation='bilinear')
    plt.show()




