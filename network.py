import tensorflow as tf
import numpy as np
import fetch_data as fd
import config

class layers:

    def __init__(self):
        self.grid = tf.constant(fd.get_grid_diffs(config.grid_file, config.data_path), dtype=tf.float32)

    def get_kernel(self):
        kernel = np.asarray([[[0, 1, 0], [1, 1, 1], [0, 1, 0]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[0, 1, 0], [1, 1, 1], [0, 1, 0]]])
        kernel = np.expand_dims(kernel, axis=3)
        kernel = np.expand_dims(kernel, axis=4)
        kernel = tf.constant(kernel, dtype=tf.float32)

        return kernel

    def SB(self, x, j, n_filters=16, n_blocks=1, upsample=True):
        output = x
        for i in range(n_blocks):
            if(upsample):
                weight_name = 'SB_BLOCK_UP' + str(i) + str(j)
            else:
                weight_name = 'SB_BLOCK_DW' + str(i) + str(j)

            conv_layer = tf.layers.conv3d(inputs=output,
                                        filters=n_filters,
                                        strides=(1, 1, 1),
                                        kernel_size=(3, 3, 3),
                                        activation=tf.nn.leaky_relu,
                                        padding='SAME',
                                        name=weight_name)
            output = conv_layer

        if x.get_shape().as_list()[4] == n_filters:
            output += x

        return output

    def BB(self, x, bb_blocks=1, n_filters=16, FEM=False, upsample=True):
        for q in range(bb_blocks):
            if not FEM:
                x = self.SB(x, q, n_filters=n_filters, n_blocks=bb_blocks, upsample=upsample)
            else:
                x = self.geometric_SB(x, q, upsample=upsample)

            if upsample:
                x = self.upsample(x)
            else:
                x = self.downsample(x)
        return x

    def upsample(self, x, interpolation=1):
        tensor_list_x = []
        tensor_list_y = []
        xdim = x.get_shape().as_list()[1]
        ydim = x.get_shape().as_list()[2]
        zdim = x.get_shape().as_list()[2]

        unstack_x = tf.unstack(x, axis=1)
        for i in unstack_x:
            tensor_list_x.append(tf.image.resize_images(i, [ydim*2, zdim*2], method=interpolation))

        x = tf.stack(tensor_list_x, axis=1)

        unstack_y = tf.unstack(x, axis=2)
        for i in unstack_y:
            tensor_list_y.append(tf.image.resize_images(i, [xdim*2, zdim*2], method=interpolation))

        x = tf.stack(tensor_list_y, axis=2)

        return x

    def downsample(self, x, interpolation=1):
        tensor_list_x = []
        tensor_list_y = []
        xdim = x.get_shape().as_list()[1]
        ydim = x.get_shape().as_list()[2]
        zdim = x.get_shape().as_list()[2]

        unstack_x = tf.unstack(x, axis=1)
        for i in unstack_x:
            tensor_list_x.append(tf.image.resize_images(i, [tf.cast(ydim/2, dtype=tf.int32), tf.cast(zdim/2, dtype=tf.int32)], method=interpolation))

        x = tf.stack(tensor_list_x, axis=1)

        unstack_y = tf.unstack(x, axis=2)
        for i in unstack_y:
            tensor_list_y.append(tf.image.resize_images(i, [tf.cast(xdim/2, dtype=tf.int32), tf.cast(zdim/2, dtype=tf.int32)], method=interpolation))

        x = tf.stack(tensor_list_y, axis=2)

        return x

    def central_difference(self, x):
        x_stack = tf.unstack(x, axis=4)
        dx_kernel = tf.transpose(tf.constant([[[[[1.0, 0.0, -1.0]]]]]), perm=[4, 1, 2, 3, 0])
        dy_kernel = tf.transpose(tf.constant([[[[[1.0, 0.0, -1.0]]]]]), perm=[0, 4, 2, 3, 1])
        dz_kernel = tf.transpose(tf.constant([[[[[1.0, 0.0, -1.0]]]]]), perm=[0, 1, 4, 3, 2])

        d_stack = []
        for slices in x_stack:
            dx = tf.nn.conv3d(tf.expand_dims(slices, axis=4), strides=(1, 1, 1, 1, 1), filter=dx_kernel, padding='SAME',
                              name='dx')
            dy = tf.nn.conv3d(tf.expand_dims(slices, axis=4), strides=(1, 1, 1, 1, 1), filter=dy_kernel, padding='SAME',
                              name='dy')
            dz = tf.nn.conv3d(tf.expand_dims(slices, axis=4), strides=(1, 1, 1, 1, 1), filter=dz_kernel, padding='SAME',
                              name='dz')
            d_stack.append(dx)
            d_stack.append(dy)
            d_stack.append(dz)

        return tf.squeeze(tf.stack(d_stack, axis=4), axis=5)

    def FEM_diffential(self, x):

        shape_x = x.get_shape().as_list()[1]
        shape_y = x.get_shape().as_list()[2]
        shape_z = x.get_shape().as_list()[3]
        data_channels = x.get_shape().as_list()[4]

        grid_channels = self.grid.get_shape().as_list()[4]

        x = tf.einsum('qxyzk,qxyzt->qxyzkt', x, self.grid)
        x = tf.reshape(x, shape=(-1, shape_x, shape_y, shape_z, data_channels * grid_channels))

        unstack_output = tf.unstack(x, axis=4)
        output_list = []
        for i in unstack_output:
            i = tf.expand_dims(i, axis=4)
            slice_conv = tf.nn.conv3d(input=i,
                                      filter=self.get_kernel(),
                                      padding='SAME',
                                      strides=(1, 1, 1, 1, 1))

            output_list.append(slice_conv)

        x = tf.squeeze(tf.stack(output_list, axis=4), axis=5)
        return x


    def geometric_SB(self, x, j, n_blocks=1, n_filters=16, upsample=True, kernel_width = 3, activation = None):

        shape_x = x.get_shape().as_list()[1]
        shape_y = x.get_shape().as_list()[2]
        shape_z = x.get_shape().as_list()[3]
        data_channels = x.get_shape().as_list()[4]

        grid_x = self.grid.get_shape().as_list()[1]
        grid_y = self.grid.get_shape().as_list()[2]
        grid_z = self.grid.get_shape().as_list()[3]
        grid_channels = self.grid.get_shape().as_list()[4]

        ratio_x = grid_x / shape_x
        ratio_y = grid_y / shape_y
        ratio_z = grid_z / shape_z

        max_subs = int(np.log2(np.max([ratio_x, ratio_y, ratio_z])))
        _grid = self.grid
        for i in range(max_subs):
            _grid = tf.nn.avg_pool3d(input=_grid, ksize=(1, 1, 1, 1, 1), padding='VALID', strides=(1, 2, 2, 2, 1))

        output = x
        output = tf.einsum('qxyzk,qxyzt->qxyzkt', output, _grid)
        output = tf.reshape(output, shape=(-1, shape_x, shape_y, shape_z, data_channels*grid_channels))
        unstack_output = tf.unstack(output, axis=4)
        output_list = []
        for i in unstack_output:
            i = tf.expand_dims(i, axis=4)
            slice_conv = tf.nn.conv3d(input=i,
                                  filter=self.get_kernel(),
                                  padding='SAME',
                                  strides=(1, 1, 1, 1, 1))

            output_list.append(slice_conv)

        output = tf.squeeze(tf.stack(output_list, axis=4), axis=5)
        output = tf.concat((x, output), axis=4)

        for i in range(n_blocks):
            if upsample:
                weight_name = 'SB_BLOCK_UP' + str(i) + str(j)
            else:
                weight_name = 'SB_BLOCK_DW' + str(i) + str(j)

            output = tf.layers.conv3d(inputs=output,
                                        filters=n_filters,
                                        strides=(1, 1, 1),
                                        kernel_size=(kernel_width, kernel_width, kernel_width),
                                        activation=activation,
                                        padding='SAME',
                                        name=weight_name)

        if data_channels == n_filters:
            output += x
        return output

class IntegratorNetwork:

    def __init__(self, param_state_size=8, sequence_length = 30):

        self.y = tf.placeholder(dtype=tf.float32, shape=(sequence_length, 2*param_state_size), name='label_encodings')

        self.full_encoding = tf.placeholder(dtype=tf.float32, shape=(1, 1, 2*param_state_size), name='start_encoding')
        self.parm_encodings = tf.placeholder(dtype=tf.float32, shape=(sequence_length, param_state_size), name= 'parameter_encodings')
        training = tf.placeholder(dtype=tf.bool, name='phase')
        self.list_of_encodings = self.full_encoding

        def body(encoding, idx, list_of_encodings, parm_encodings):
            f1 = tf.contrib.layers.batch_norm(tf.nn.dropout(tf.layers.dense(encoding, 1024, activation=tf.nn.elu), rate=0.1), is_training=training)
            f2 = tf.contrib.layers.batch_norm(tf.nn.dropout(tf.layers.dense(f1, 512, activation=tf.nn.elu), rate=0.1), is_training=training)
            T =  tf.contrib.layers.batch_norm(tf.nn.dropout(tf.layers.dense(f2, param_state_size, activation=tf.nn.elu), rate=0.1), is_training=training)

            encoding = tf.slice(encoding,[0, 0, 0],[-1, -1, param_state_size])
            sliced_parm_encoding = tf.slice(parm_encodings, [tf.cast(idx, dtype=tf.int32), 0], [1, -1])

            sliced_parm_encoding = tf.expand_dims(sliced_parm_encoding, axis=0)
            encoding = tf.concat((encoding, sliced_parm_encoding), axis=2)

            encoding = encoding + tf.pad(T, tf.constant([[0, 0], [0, 0], [0, param_state_size]]), 'CONSTANT')
            idx = idx + 1
            list_of_encodings = tf.concat((list_of_encodings, encoding), axis=1)

            return encoding, idx, list_of_encodings, parm_encodings

        def condition(encoding, idx, list_of_encodings, parm_encodings):
            return tf.less(idx, sequence_length)

        idx = tf.constant(0)
        self.full_encoding, idx, self.list_of_encodings, self.parm_encodings = tf.while_loop(condition, body, [self.full_encoding, idx, self.list_of_encodings, self.parm_encodings],
                                                                      shape_invariants=[tf.TensorShape([1, 1, 2*param_state_size]),
                                                                                        idx.get_shape(),
                                                                                        tf.TensorShape([1, None, 2*param_state_size]),
                                                                                        tf.TensorShape([sequence_length, param_state_size])])

        self.list_of_encodings = tf.slice(self.list_of_encodings, [0, 1, 0], [-1, -1, -1])

        self.loss = tf.reduce_mean(tf.square(self.list_of_encodings - tf.expand_dims(self.y, axis=0)))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train = tf.train.AdamOptimizer(0.0001).minimize(self.loss)


class NetWork(layers):

    def __init__(self, voxel_side, param_state_size=8):
        super(NetWork, self).__init__()

        self.param_state_size = param_state_size
        self.vx_log2 = np.log2(voxel_side)
        self.q = self.vx_log2 - np.log2(param_state_size)

        x = tf.placeholder(dtype=tf.float32, shape=(None,  voxel_side,  voxel_side,  voxel_side, 3), name='velocity')
        sdf = tf.placeholder(dtype=tf.float32, shape=(None,  voxel_side,  voxel_side,  voxel_side, 1), name='sdf')
        c = self.encoder_network(x)

        with tf.variable_scope('boundary_conditions'):
            self.encoded_sdf = self.encoder_network(sdf)

        self.full_encoding = tf.concat((c, self.encoded_sdf), axis=1)

        y = self.decoder_network(self.full_encoding)

        if not config.use_fem:
            dy = self.central_difference(y)
            dx = self.central_difference(x)
        else:
            dy = self.FEM_diffential(y)
            dx = self.FEM_diffential(x)

        self.l2_loss_v = tf.reduce_mean(tf.square(y - x))
        self.l2_loss_dv = tf.reduce_mean(tf.square(dy - dx))

        self.loss = self.l2_loss_v + self.l2_loss_dv
        self.train = tf.train.AdamOptimizer(0.0001).minimize(self.loss)

    def decoder_network(self, x, sb_blocks=1, n_filters=128):

        output_dim = pow(self.param_state_size, 3)*n_filters
        x = tf.layers.dense(x, output_dim, activation=tf.nn.leaky_relu)
        x = tf.reshape(x, shape=(-1, self.param_state_size, self.param_state_size, self.param_state_size, n_filters))

        x = self.BB(x, int(self.q), FEM=config.use_fem, n_filters=n_filters)
        x = tf.layers.conv3d(x,   strides=(1, 1, 1),
                                  kernel_size=(3, 3, 3),
                                  filters=3, padding='SAME',
                                  activation=tf.nn.leaky_relu,
                                  name='output_convolution')
        return x

    def encoder_network(self, x,  sb_blocks=1, n_filters=128):

        x = self.BB(x, int(self.q), FEM=config.use_fem, upsample=False, n_filters=n_filters)
        x = tf.contrib.layers.flatten(x)
        c = tf.layers.dense(x, self.param_state_size, activation=tf.nn.leaky_relu)
        return c


    def tensor_board(self, on = False):
        if on:
            tf.summary.scalar('Loss', self.loss)
            tf.summary.scalar('Field Loss', self.l2_loss_v)
            tf.summary.scalar('Gradient Loss', self.l2_loss_dv)
            merged = tf.summary.merge_all()
            return merged
