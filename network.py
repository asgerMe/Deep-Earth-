import tensorflow as tf
import numpy as np
import fetch_data as fd
import config

class train_schedule:

    @staticmethod
    def cosine_annealing(step, max_step, lr_min, lr_max):
        g_lr = tf.Variable(lr_max, name='g_lr')
        g_lr_update = tf.assign(g_lr, lr_min + 0.5 * (lr_max - lr_min) * ( tf.cos(tf.cast(step, tf.float32) * np.pi / max_step) + 1), name='g_lr_update')
        return g_lr_update

class layers:

    def __init__(self):
        self.grid = tf.constant(fd.get_grid_diffs(config.grid_file, config.data_path), dtype=tf.float32)

    def differential_kernel(self):
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
                                        name=weight_name,  kernel_initializer=tf.contrib.layers.xavier_initializer() )
            output = conv_layer

        if x.get_shape().as_list()[4] == n_filters:
            output += x

        return output

    def BB(self, x, bb_blocks=1, sb_blocks=1, n_filters=16, FEM=False, upsample=True):
        for q in range(bb_blocks):
            if not FEM:
                x = self.SB(x, q, n_filters=n_filters, n_blocks=sb_blocks, upsample=upsample)
            else:
                x = self.geometric_SB(x, q, n_blocks=sb_blocks, upsample=upsample)

            if upsample:
                x = self.upsample(x)
            else:
                x = self.downsample(x)
        return x

    def trilinear_interpolation_kernel(self):
        kernel = np.asarray(
            [[[1.0/32, 1.0/16, 1.0/32], [1.0/16, 1.0/8, 1.0/16], [1.0/32, 1.0/16, 1.0/32]],
             [[1.0/16, 1.0/8, 1.0/16], [1.0/8, 1.0/4, 1.0/8], [1.0/16, 1.0/8, 1.0/16]],
             [[1.0/32, 1.0/16, 1.0/32], [1.0/16, 1.0/8, 1.0/16], [1.0/32, 1.0/16, 1.0/32]]])

        kernel = np.expand_dims(kernel, axis=3)
        kernel = np.expand_dims(kernel, axis=4)
        kernel = tf.constant(kernel, dtype=tf.float32)

        return kernel


    def upsample(self, x, interpolation=1):


        xdim = x.get_shape().as_list()[1]
        ydim = x.get_shape().as_list()[2]
        zdim = x.get_shape().as_list()[3]
        wdim = x.get_shape().as_list()[4]

        x = tf.reshape(x, [-1, xdim, ydim, zdim])
        bdim = tf.shape(x)[0]

        x = tf.expand_dims(x, axis=4)

        x = tf.nn.conv3d_transpose(x, output_shape=[bdim, 2*xdim, 2*ydim, 2*zdim, 1], strides=(1, 2, 2, 2, 1), filter=self.trilinear_interpolation_kernel(), padding='SAME', name = 'upsample')
        x = tf.reshape(x, [-1, int(2*xdim), int(2*ydim), int(2*zdim), int(wdim)])
        return x

    def downsample(self, x, interpolation=1):

        xdim = x.get_shape().as_list()[1]
        ydim = x.get_shape().as_list()[2]
        zdim = x.get_shape().as_list()[3]
        wdim = x.get_shape().as_list()[4]

        x = tf.reshape(x, [-1, xdim, ydim, zdim])
        x = tf.expand_dims(x, axis=4)

        x = tf.nn.conv3d(x, strides=(1, 2, 2, 2, 1), filter=self.trilinear_interpolation_kernel(), padding='SAME', name='downsample')
        x = tf.reshape(x, [-1, int(xdim/2), int(ydim/2), int(zdim/2), int(wdim)])
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
            _grid = self.downsample(_grid)

        output = x
        output = tf.einsum('qxyzk,qxyzt->qxyzkt', output, _grid)
        output = tf.reshape(output, shape=(-1, shape_x, shape_y, shape_z, data_channels*grid_channels))
        unstack_output = tf.unstack(output, axis=4)
        output_list = []
        for i in unstack_output:
            i = tf.expand_dims(i, axis=4)
            slice_conv = tf.nn.conv3d(input=i,
                                  filter=self.differential_kernel(),
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

    def __init__(self, param_state_size=8):

        self.y = tf.placeholder(dtype=tf.float32, shape=(None, 2*param_state_size), name='label_encodings')
        sequence_length = tf.placeholder(dtype=tf.int32, name='sequence_length')
        self.full_encoding = tf.placeholder(dtype=tf.float32, shape=(1, 1, 2*param_state_size), name='start_encoding')
        self.parm_encodings = tf.placeholder(dtype=tf.float32, shape=(None, param_state_size), name= 'parameter_encodings')
        training = tf.placeholder(dtype=tf.bool, name='phase')

        with tf.variable_scope('Integrater_Network'):
            self.list_of_encodings = self.full_encoding

            def body(encoding, idx, list_of_encodings, parm_encodings):
                f1 = tf.nn.dropout(tf.layers.dense(encoding, 1024, activation=tf.nn.elu), keep_prob=0.9)
                f2 = tf.nn.dropout(tf.layers.dense(f1, 512, activation=tf.nn.elu), keep_prob=0.9)
                T = tf.nn.dropout(tf.layers.dense(f2, param_state_size, activation=tf.nn.elu), keep_prob=0.9)

                encoding = tf.slice(encoding,[0, 0, 0], [-1, -1, param_state_size])
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
                                                                                            tf.TensorShape([None, param_state_size])])


            self.list_of_encodings = tf.slice(self.list_of_encodings, [0, 1, 0], [-1, -1, -1], name='next_encoding')

            self.loss_int = tf.reduce_mean(tf.square(self.list_of_encodings - tf.expand_dims(self.y, axis=0)))
            self.merged_int = tf.summary.scalar('Integrator Loss', self.loss_int)
            self.train_int = tf.train.AdamOptimizer(beta1=0.5).minimize(self.loss_int)


class NetWork(layers, train_schedule):

    def __init__(self, voxel_side, param_state_size=8):
        super(NetWork, self).__init__()

        self.param_state_size = param_state_size
        self.vx_log2 = np.log2(voxel_side)
        self.q = self.vx_log2 - np.log2(param_state_size)

        x = tf.placeholder(dtype=tf.float32, shape=(None,  voxel_side,  voxel_side,  voxel_side, 3), name='velocity')
        sdf = tf.placeholder(dtype=tf.float32, shape=(None,  voxel_side,  voxel_side,  voxel_side, 1), name='sdf')

        with tf.variable_scope('Boundary_conditions'):
            self.encoded_sdf = tf.identity(self.encoder_network(sdf, sb_blocks=1, n_filters=8), name='encoded_sdf')

        with tf.variable_scope('Encoder'):
            c = tf.identity(self.encoder_network(x, sb_blocks=config.sb_blocks, n_filters=config.n_filters), name= 'encoded_field')

        with tf.variable_scope('Latent_State'):
            self.full_encoding = tf.tanh(tf.concat((c, self.encoded_sdf), axis=1, name='full_encoding'))

        with tf.variable_scope('Decoder'):
            y = tf.identity(self.decoder_network(self.full_encoding, sb_blocks=config.sb_blocks, n_filters=config.n_filters), name='decoder')

        with tf.variable_scope('Differentiate'):
            if not config.use_fem:
                dy = self.central_difference(y)
                dx = self.central_difference(x)
            else:
                dy = self.FEM_diffential(y)
                dx = self.FEM_diffential(x)

        with tf.variable_scope('Loss_Estimation'):
            self.l2_loss_v = tf.reduce_mean(tf.reduce_sum(tf.square(y - x), axis= 4))
            self.l2_loss_dv = tf.reduce_mean(tf.reduce_sum(tf.square(dy - dx), axis=4))

            self.loss = self.l2_loss_v + self.l2_loss_dv

        with tf.variable_scope('Train'):
            self.step = tf.placeholder(dtype=tf.int32, name='step')
            self.lr = train_schedule.cosine_annealing(lr_max=0.0001, lr_min=0.000025, max_step= 5000, step=self.step)
            self.train = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5).minimize(self.loss)

        tf.summary.scalar('Encoder Loss', self.loss)
        tf.summary.scalar('Encoder Field Loss', self.l2_loss_v)
        tf.summary.scalar('Encoder Gradient Loss', self.l2_loss_dv)
        self.merged = tf.summary.merge_all()

    def decoder_network(self, x, sb_blocks=1, n_filters=128):
        output_dim = pow(self.param_state_size, 3)*n_filters
        x = tf.layers.dense(x, output_dim, activation=tf.nn.leaky_relu, kernel_initializer= tf.contrib.layers.xavier_initializer())
        x = tf.reshape(x, shape=(-1, self.param_state_size, self.param_state_size, self.param_state_size, n_filters))

        x = self.BB(x, int(self.q), FEM=config.use_fem, sb_blocks=sb_blocks, n_filters=n_filters)
        x = tf.layers.conv3d(x,   strides=(1, 1, 1),
                                  kernel_size=(3, 3, 3),
                                  filters=3, padding='SAME',
                                  name='output_convolution')
        return x

    def encoder_network(self, x, sb_blocks=1, n_filters=128):
        x = self.BB(x, int(self.q), FEM=config.use_fem, upsample=False, sb_blocks=sb_blocks, n_filters=n_filters)
        x = tf.contrib.layers.flatten(x)
        c = tf.layers.dense(x, self.param_state_size, activation=tf.nn.leaky_relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
        return c


