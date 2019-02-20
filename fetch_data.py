import numpy as np
import os
import config
import matplotlib.pyplot as plt
#this is a test

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

    voxels = np.shape(batch)[2]
    if np.log2(voxels) != np.ceil(np.log2(voxels)):
        new_voxels = pow(2, np.ceil(np.log2(voxels)))
        padding = int(new_voxels - voxels)
        batch = np.pad(batch, ((0, 0), (0, 0), (padding, 0), (padding, 0), (padding, 0)), mode='constant')

    sdf = np.expand_dims(batch[:, 0, :, :, :], 4)
    velocity = np.moveaxis(batch[:, 1:4, :, :, :], 1, 4)

    feed_dict = {'sdf:0': sdf, 'velocity:0': velocity}
    if not sequential:
        return feed_dict
    else:
        feed_dict_0 = {'sdf:0': [sdf[0, :, :, :, :]], 'velocity:0': [velocity[0, :, :, :, :]]}
        return feed_dict, feed_dict_0

def get_grid_diffs(file_name, data_path=config.data_path):

    full_path = os.path.join(data_path, file_name)

    try:
        data = np.load(full_path, mmap_mode='r+')
        voxels = np.shape(data)[2]
        if np.log2(voxels) != np.ceil(np.log2(voxels)):
            new_voxels = pow(2, np.ceil(np.log2(voxels)))
            padding = int(new_voxels - voxels)
            data = np.pad(data, ((0, 0), (0, 0), (padding, 0), (padding, 0), (padding, 0)), mode='constant')
            data = np.moveaxis(data, 1, 4)

        return data

    except IndexError:
        print('could not open file')
        return 0



