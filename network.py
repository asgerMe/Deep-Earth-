import tensorflow as tf
import numpy as np
import fetch_data as fd
import config
import util
import os

class layers:

    def __init__(self):

        if config.fem_loss or config.use_fem:
            index, value, shape, grid = util.get_multihot()
            self.multi_hot = tf.SparseTensorValue(index, value, shape)
            self.grid_dict = grid

        self.Test = 100

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
                                        kernel_size=(1, 1, 1),
                                        activation=tf.nn.leaky_relu,
                                        padding='SAME',
                                        name=weight_name)
            output = conv_layer

        if x.get_shape().as_list()[4] == n_filters:
            output += x

        return output

    def BB(self, x, bb_blocks=1, sb_blocks=1, n_filters=16, FEM=False, upsample=True):

        for q in range(bb_blocks):
            if not upsample:
                filters = (1 + q) * n_filters
            else:
                filters = n_filters

            x = self.SB(x, q, n_filters=filters, n_blocks=sb_blocks, upsample=upsample)
            if upsample:
                if not config.resample:
                    with tf.name_scope('nearest_neighbour_interpolation'):
                        x = self.resample_nearest_neighbour(x, 2.0)
                else:
                    with tf.name_scope('Tri_linear_interpolation'):
                        x = self.upsample(x)
            else:
                x = tf.layers.conv3d(x, strides=(2, 2, 2),
                                     kernel_size=(3, 3, 3),
                                     filters= (2+q)*n_filters, padding='SAME',
                                     name='down_sample_' + str(q))

        return x


    def upsample(self, x, interpolation=1):

        xdim = x.get_shape().as_list()[1]
        ydim = x.get_shape().as_list()[2]
        zdim = x.get_shape().as_list()[3]
        wdim = x.get_shape().as_list()[4]

        x = tf.reshape(x, [-1, xdim, ydim, zdim])
        bdim = tf.shape(x)[0]

        x = tf.expand_dims(x, axis=4)

        x = tf.nn.conv3d_transpose(x, output_shape=[bdim, 2*xdim, 2*ydim, 2*zdim, 1], strides=(1, 2, 2, 2, 1), filter=util.trilinear_interpolation_kernel(), padding='SAME', name = 'upsample')
        x = tf.reshape(x, [-1, int(2*xdim), int(2*ydim), int(2*zdim), int(wdim)])
        return x

    def downsample(self, x, interpolation=1):

        xdim = x.get_shape().as_list()[1]
        ydim = x.get_shape().as_list()[2]
        zdim = x.get_shape().as_list()[3]
        wdim = x.get_shape().as_list()[4]

        x = tf.reshape(x, [-1, xdim, ydim, zdim])
        x = tf.expand_dims(x, axis=4)

        x = tf.nn.conv3d(x, strides=(1, 2, 2, 2, 1), filter=util.trilinear_interpolation_kernel(), padding='SAME', name='downsample')
        x = tf.reshape(x, [-1, int(xdim/2), int(ydim/2), int(zdim/2), int(wdim)])
        return x

    def resample_nearest_neighbour(self, x, scale):

        d = int(x.get_shape().as_list()[1])
        h = int(x.get_shape().as_list()[2])
        w = int(x.get_shape().as_list()[3])
        c = int(x.get_shape().as_list()[4])


        hw = tf.reshape(tf.transpose(x, [0, 2, 3, 1, 4]), [-1, int(h), int(w), int(d * c)])
        h *= scale
        w *= scale

        hw = tf.image.resize_nearest_neighbor(hw, (int(h), int(w)))
        hw = tf.reshape(hw, [-1, int(h), int(w), int(d), int(c)])

        dh = tf.reshape(tf.transpose(hw, [0,3,1,2,4]), [-1,int(d),int(h),int(w*c)])
        d *= scale
        dh = tf.image.resize_nearest_neighbor(dh, (int(d), int(h)))

        return tf.reshape(dh, [-1, int(d), int(h), int(w), int(c)])


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

    def fem_differentiate(self, x):
        data_size = x.get_shape().as_list()[1]
        flatten_prim_idx = tf.constant(self.grid_dict['prim_points'], dtype=tf.int64)
        n_filters = x.get_shape().as_list()[4]
        inverse_jacobians = tf.constant(np.reshape(self.grid_dict['inv_j'], newshape=(-1, 4, 4)), dtype=tf.float32)

        flat_output = tf.reshape(x, shape=(tf.shape(x)[0], -1, tf.shape(x)[4]))

        prim_aligned_features = tf.gather(flat_output, flatten_prim_idx, axis=1)

        differentiate = tf.einsum('jqk, ijqt -> ijkt', inverse_jacobians, prim_aligned_features)

        primitive_features = tf.reshape(differentiate, shape=(-1, tf.shape(inverse_jacobians)[0], 4 * n_filters))
        b = tf.reshape(primitive_features, shape=(tf.shape(primitive_features)[1], -1))

        accumulate_features_on_points = tf.sparse_tensor_dense_matmul(self.multi_hot, b)

        out = tf.reshape(accumulate_features_on_points, shape=(-1, data_size, data_size, data_size, 4 * n_filters))
        out = tf.concat((out, x), axis=4)
        return out

    def differentiate_features(self, x, n_filters = 128, name=''):

        with tf.variable_scope('Differentiate_Features' + name):

            x_s = tf.layers.conv3d(x, strides=(1, 1, 1),
                                kernel_size=(1, 1, 1),
                                filters=n_filters, padding='SAME',
                                name='diff_convolution_x_s',
                                activation=tf.nn.leaky_relu)

            out = tf.layers.conv3d(x, strides=(1, 1, 1),
                                kernel_size=(1, 1, 1),
                                filters=n_filters, padding='SAME',
                                name='diff_convolution_in',
                                activation=tf.nn.leaky_relu)
            if config.use_fem:
                out = self.fem_differentiate(out)

            out = tf.layers.conv3d(out, strides=(1, 1, 1),
                                kernel_size=(1, 1, 1),
                                filters=n_filters, padding='SAME',
                                name='diff_convolution_out',
                                activation = tf.nn.leaky_relu)

            out = x_s + out
            out = tf.nn.leaky_relu(out)

            return out

class IntegratorNetwork:

    def __init__(self, param_state_size=8, sdf_state_size=8):

        self.y = tf.placeholder(dtype=tf.float32, shape=(None, sdf_state_size  + param_state_size), name='label_encodings')
        sequence_length = tf.placeholder(dtype=tf.int32, name='sequence_length')
        self.full_encoding = tf.placeholder(dtype=tf.float32, shape=(1, 1, sdf_state_size + param_state_size), name='start_encoding')
        self.parm_encodings = tf.placeholder(dtype=tf.float32, shape=(None, sdf_state_size), name= 'parameter_encodings')
        training = tf.placeholder(dtype=tf.bool, name='phase')

        with tf.variable_scope('Integrater_Network'):
            self.list_of_encodings = self.full_encoding

            def body(encoding, idx, list_of_encodings, parm_encodings):
                f1 = tf.nn.dropout(tf.layers.dense(encoding, 1024, activation=tf.nn.elu, name='layer_1'), keep_prob=0.9)
                f2 = tf.nn.dropout(tf.layers.dense(f1, 512, activation=tf.nn.elu, name='layer_2'), keep_prob=0.9)
                T = tf.layers.dense(f2, param_state_size, activation=tf.nn.elu, name='next_encoding')


                encoding = tf.slice(encoding, [0, 0, 0], [-1, -1, param_state_size])
                sliced_parm_encoding = tf.slice(parm_encodings, [tf.cast(idx, dtype=tf.int32), 0], [1, -1])

                sliced_parm_encoding = tf.expand_dims(sliced_parm_encoding, axis=0)
                encoding = tf.concat((encoding, sliced_parm_encoding), axis=2)

                encoding = encoding + tf.pad(T, tf.constant([[0, 0], [0, 0], [0, sdf_state_size]]), 'CONSTANT')
                idx = idx + 1
                list_of_encodings = tf.concat((list_of_encodings, encoding), axis=1)

                return encoding, idx, list_of_encodings, parm_encodings

            def condition(encoding, idx, list_of_encodings, parm_encodings):
                return tf.less(idx, sequence_length)

            idx = tf.constant(0)
            self.full_encoding, idx, self.list_of_encodings, self.parm_encodings = tf.while_loop(condition, body, [self.full_encoding, idx, self.list_of_encodings, self.parm_encodings],
                                                                        shape_invariants=[tf.TensorShape([1, 1, sdf_state_size + param_state_size]),
                                                                                            idx.get_shape(),
                                                                                            tf.TensorShape([1, None, sdf_state_size + param_state_size]),
                                                                                            tf.TensorShape([None, sdf_state_size])])


            self.list_of_encodings = tf.slice(self.list_of_encodings, [0, 0, 0], [-1, tf.shape(self.list_of_encodings)[1] -1, -1], name='next_encoding')

            self.loss_int = tf.reduce_mean(tf.square(self.list_of_encodings - tf.expand_dims(self.y, axis=0)))
            self.merged_int = tf.summary.scalar('Integrator Loss', self.loss_int)
            self.train_int = tf.train.AdamOptimizer(learning_rate=config.lr_max).minimize(self.loss_int)

class Convo_IntegratorNetwork:

    def __init__(self, voxel_side, param_state_size=8, sdf_state_size=8):

        self.y = tf.placeholder(dtype=tf.float32, shape=(None, voxel_side, voxel_side, voxel_side, sdf_state_size + param_state_size),
                                    name='label_encodings')
        sequence_length = tf.placeholder(dtype=tf.int32, name='sequence_length')
        self.full_encoding = tf.placeholder(dtype=tf.float32, shape=(1, voxel_side, voxel_side, voxel_side, sdf_state_size + param_state_size),
                                                name='start_encoding')
        self.parm_encodings = tf.placeholder(dtype=tf.float32, shape=(None, voxel_side, voxel_side, voxel_side, sdf_state_size),
                                                 name='parameter_encodings')
        training = tf.placeholder(dtype=tf.bool, name='phase')

        with tf.variable_scope('Integrater_Network'):
            self.list_of_encodings = self.full_encoding

            def body(encoding, idx, list_of_encodings, parm_encodings):


                x = tf.layers.conv3d(encoding, strides=(1, 1, 1),
                                     kernel_size=(3, 3, 3),
                                     filters=32 * param_state_size, padding='SAME',
                                     name='integrator0',
                                     activation=tf.nn.leaky_relu)
                x = tf.layers.conv3d(x, strides=(1, 1, 1),
                                     kernel_size=(3, 3, 3),
                                     filters=16 * param_state_size, padding='SAME',
                                     name='integrator1',
                                     activation=tf.nn.leaky_relu)
                x = tf.layers.conv3d(x, strides=(1, 1, 1),
                                     kernel_size=(3, 3, 3),
                                     filters=8 * param_state_size, padding='SAME',
                                     name='integrator2',
                                     activation=tf.nn.leaky_relu)
                T = tf.layers.conv3d(x, strides=(1, 1, 1),
                                     kernel_size=(1, 1, 1),
                                     filters=param_state_size, padding='SAME',
                                     name='new_v_latent',
                                     activation=tf.nn.leaky_relu)

                encoding = tf.slice(encoding, [0, 0, 0, 0, 0], [-1, -1, -1, -1, param_state_size])
                sliced_parm_encoding = tf.slice(parm_encodings, [tf.cast(idx, dtype=tf.int32), 0, 0, 0, 0], [1, -1, -1, -1, -1])

                sliced_parm_encoding = tf.expand_dims(sliced_parm_encoding, axis=0)
                encoding = tf.concat((encoding, sliced_parm_encoding), axis=2)

                encoding = encoding + tf.pad(T, tf.constant([[0, 0], [0, 0], [0, 0], [0, 0], [0, sdf_state_size]]), 'CONSTANT')
                idx = idx + 1
                list_of_encodings = tf.concat((list_of_encodings, encoding), axis=1)

                return encoding, idx, list_of_encodings, parm_encodings

            def condition(encoding, idx, list_of_encodings, parm_encodings):
                return tf.less(idx, sequence_length)

            idx = tf.constant(0)
            self.full_encoding, idx, self.list_of_encodings, self.parm_encodings = tf.while_loop(condition, body, [
            self.full_encoding, idx, self.list_of_encodings, self.parm_encodings], shape_invariants=[tf.TensorShape([1, 1,sdf_state_size + param_state_size]),
                                                                                                         idx.get_shape(), tf.TensorShape([1, None, sdf_state_size + param_state_size]),
                                                                                                         tf.TensorShape([None, sdf_state_size])])

            self.list_of_encodings = tf.slice(self.list_of_encodings, [0, 0, 0],
                                                  [-1, tf.shape(self.list_of_encodings)[1] - 1, -1],
                                                  name='next_encoding')

            self.loss_int = tf.reduce_mean(tf.square(self.list_of_encodings - tf.expand_dims(self.y, axis=0)))
            self.merged_int = tf.summary.scalar('Integrator Loss', self.loss_int)
            self.train_int = tf.train.AdamOptimizer(learning_rate=config.lr_max).minimize(self.loss_int)




class NetWork(layers):

    def __init__(self, voxel_side, param_state_size=8):
        super(NetWork, self).__init__()

        self.param_state_size = param_state_size
        self.vx_log2 = np.log2(voxel_side)
        self.q = self.vx_log2 - np.log2(param_state_size)

        self.x = tf.placeholder(dtype=tf.float32, shape=(None,  voxel_side,  voxel_side,  voxel_side, 3), name='velocity')
        sdf = tf.placeholder(dtype=tf.float32, shape=(None,  voxel_side,  voxel_side,  voxel_side, 1), name='sdf')
        self.labels = tf.placeholder(dtype=tf.float32, shape=(None,  voxel_side,  voxel_side,  voxel_side, 3), name='labels')

        with tf.variable_scope('Boundary_conditions'):
            self.encoded_sdf = tf.identity(self.encoder_network_sdf(sdf, sb_blocks=1, n_filters=8, output=config.sdf_state), name='encoded_sdf')

        with tf.variable_scope('Encoder'):
            c = tf.identity(self.encoder_network(self.x, sb_blocks=config.sb_blocks, n_filters=config.n_filters), name= 'encoded_field')

        with tf.variable_scope('Latent_State'):
            if not config.conv:
                self.full_encoding = (tf.identity(tf.concat((c, self.encoded_sdf), axis=1), name='full_encoding'))
            else:
                self.full_encoding = tf.identity(tf.concat((c, self.encoded_sdf), axis=4), name='full_encoding')

        with tf.variable_scope('Decoder'):
            self.y = tf.identity(self.decoder_network(self.full_encoding, sdf, sb_blocks=config.sb_blocks, n_filters=config.n_filters), name='decoder')

        with tf.variable_scope('Differentiate'):
            if not config.fem_loss:
                dy = self.central_difference(self.y)
                self.d_labels = self.central_difference(self.labels)
            else:
                dy = self.fem_differentiate(self.labels)
                self.d_labels = self.fem_differentiate(self.y)

        with tf.variable_scope('Loss_Estimation'):
            self.l2_loss_v = tf.reduce_mean(tf.abs(self.y - self.labels))
            self.l2_loss_dv = tf.reduce_mean(tf.abs(dy - self.d_labels))

            self.loss = 0.2*self.l2_loss_dv + self.l2_loss_v

        with tf.variable_scope('Train'):
            self.step = tf.placeholder(dtype=tf.int32, name='step')
            self.lr = util.cosine_annealing(lr_max=config.lr_max, lr_min=config.lr_min, max_step= config.period, step=self.step)
            self.train = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5).minimize(self.loss)

        tf.summary.scalar('Encoder Loss', self.loss)
        tf.summary.scalar('Encoder Field Loss', self.l2_loss_v)
        tf.summary.scalar('Encoder Gradient Loss', self.l2_loss_dv)
        tf.summary.histogram('Latent State', self.full_encoding)
        self.merged = tf.summary.merge_all()

    def decoder_network(self, x, sdf, sb_blocks=1, n_filters=128):

        if not config.conv:
            output_dim = pow(self.param_state_size, 3)*n_filters
            x = tf.layers.dense(x, output_dim, activation=tf.nn.leaky_relu, name='decoder_dense')
            x = tf.reshape(x, shape=(-1, self.param_state_size, self.param_state_size, self.param_state_size, n_filters), name='decoder_reshape')

        x = tf.layers.conv3d(x, strides=(1, 1, 1),
                             kernel_size=(3, 3, 3),
                             filters=n_filters, padding='SAME',
                             name='latent_output_convolution')

        x = self.BB(x, int(self.q), FEM=config.use_fem, sb_blocks=sb_blocks, n_filters=n_filters)

        sdf = tf.layers.conv3d(sdf, strides=(1, 1, 1),
                             kernel_size=(3, 3, 3),
                             filters=n_filters, padding='SAME',
                             name='output_convolution_sdf')

        x = tf.identity(tf.concat((x, sdf), axis=4), name='merge_sdf')


        x = self.differentiate_features(x, n_filters=n_filters, name='2')

        x = tf.layers.conv3d(x,   strides=(1, 1, 1),
                                  kernel_size=(3, 3, 3),
                                  filters=3, padding='SAME',
                                  name='output_convolution')
        return x

    def encoder_network(self, x, sb_blocks=1, n_filters=128):


        x = self.differentiate_features(x, n_filters=n_filters, name= '1')

        x = self.BB(x, int(self.q), FEM=config.use_fem, upsample=False, sb_blocks=sb_blocks, n_filters=n_filters)

        if not config.conv:
            x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(x, self.param_state_size, activation=tf.nn.leaky_relu)
        else:
            x = tf.layers.conv3d(x, strides=(1, 1, 1),
                                 kernel_size=(3, 3, 3),
                                 filters=config.param_state_size, padding='SAME',
                                 name='latent_convolution',
                                 activation=tf.nn.leaky_relu)
        return x


    def encoder_network_sdf(self, x, sb_blocks=1, n_filters=8, output=8):

        x = self.BB(x, int(self.q), FEM=config.use_fem, upsample=False, sb_blocks=sb_blocks, n_filters=n_filters)
        if not config.conv:
             x = tf.contrib.layers.flatten(x)
             x = tf.layers.dense(x, output, activation=tf.nn.leaky_relu)

        else:
            x = tf.layers.conv3d(x, strides=(1, 1, 1),
                                 kernel_size=(1, 1, 1),
                                 filters=config.param_state_size, padding='SAME',
                                 name='latent_convolution',
                                 activation=tf.nn.leaky_relu)
        return x







