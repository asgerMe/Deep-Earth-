import numpy as np
import os
import config
import glob

def get_scaling_factor(data_path):
    list_files = os.listdir(data_path)
    print('Searching for scaling factor in', len(list_files), 'files')
    scaling_factor = 1
    if len(list_files):
        for file in list_files:
            try:
                print(file)
                data = np.load(os.path.join(data_path, file), mmap_mode='r')
                max_v = np.amax(data[:, 1:4, :, :, :])
                print(np.shape(data))
                print(max_v)
                if abs(max_v) > scaling_factor:
                    scaling_factor = abs(max_v)
            except IOError:
                continue
    print('Largest v component found:', scaling_factor)
    return scaling_factor


def get_volume(data_path, batch_size=1, time_idx = -1, sequential=False, sequence_length=1, inference=False, scaling_factor=1):

    os.chdir(data_path)
    list_files = glob.glob('*.npy')

    batch = []
    random_file_name = ''
    if not sequential:
        for i in range(batch_size):
            for attempts in range(1000):
                full_path=''
                try:
                    if time_idx == -1:
                        random_file_name = list_files[np.random.randint(0, np.size(list_files))]
                    else:
                        random_file_name = list_files[0]

                    full_path = os.path.join(data_path, random_file_name)
                    print(full_path)
                except ValueError:
                    print('no .npy files in path', data_path)
                    return 0

                try:
                    data = np.load(full_path, mmap_mode='r')
                    dim1 = np.shape(data)[0]

                    if time_idx == -1:
                        time_idx = np.random.randint(0, dim1)

                    data_slice = data[time_idx, :, :]
                    batch.append(data_slice)
                    break

                except:
                    continue

    if sequential:
        for i in range(batch_size):
            try:
                random_file_name = list_files[np.random.randint(0, np.size(list_files))]
                full_path = os.path.join(data_path, random_file_name)
            except ValueError:
                print('no suitable files in path, at least two files needed !')
                exit()

            try:
                data = np.load(full_path, mmap_mode='r')
                dim1 = np.shape(data)[0]

                if not inference:
                    random_time_index = np.random.randint(0, dim1)
                else:
                    random_time_index = 0

                batch = data[random_time_index:random_time_index+sequence_length, :, :]

                padding = np.zeros([sequence_length-np.shape(batch)[0], np.shape(batch)[1], np.shape(batch)[2],
                                  np.shape(batch)[3], np.shape(batch)[4]])

                batch = np.concatenate([batch, padding], axis=0)

            except IndexError:
                continue


    batch = np.asarray(batch)

    voxels = np.shape(batch)[2]
    if np.log2(voxels) != np.ceil(np.log2(voxels)):
        new_voxels = pow(2, np.ceil(np.log2(voxels)))
        padding = int(new_voxels - voxels)
        batch = np.pad(batch, ((0, 0), (0, 0), (padding, 0), (padding, 0), (padding, 0)), mode='constant')

    sdf = np.expand_dims(batch[:, 0, :, :, :], 4)
    velocity = np.moveaxis(batch[:, 1:4, :, :, :], 1, 4)

    feed_dict = {'sdf:0': sdf, 'labels:0': velocity / scaling_factor, 'velocity:0': velocity / scaling_factor}
    if not sequential:
        return feed_dict
    else:
        feed_dict_0 = {'sdf:0': [sdf[0, :, :, :, :]], 'labels:0': [velocity[0, :, :, :, :] / scaling_factor], 'velocity:0': [velocity[0, :, :, :, :] / scaling_factor]}
        return feed_dict, feed_dict_0





