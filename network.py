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
    def B(self):


        prim_num = 0
        BCB = []
        for ij in self.grid_dict['inv_j']:

            B = np.zeros([6, 12])

            B[0, 0] = ij[1];  B[0, 3] = ij[5]; B[0, 6] = ij[9];  B[0, 9] = ij[13];   B[1, 1] = ij[2]; B[1, 4] = ij[6]
            B[1, 7] = ij[10]; B[1, 10] = ij[14];   B[2, 2] = ij[3]; B[2, 5] = ij[7]; B[2, 8] = ij[11]; B[2, 11] = ij[15]
            B[3, 0] = ij[2]; B[3, 1] = ij[1]; B[3, 3] = ij[6]; B[3, 4] = ij[5]; B[3, 6] = ij[10];  B[3, 7] = ij[9]
            B[3, 9] = ij[14];  B[3, 10] = ij[13]; B[4, 2] = ij[3]; B[4, 5] = ij[2]; B[4, 8] = ij[7]; B[4, 11] = ij[6]
            B[4, 2] = ij[11]; B[4, 5] = ij[10];  B[4, 8] = ij[15]; B[4, 11] = ij[14];   B[5, 0] = ij[3];  B[5, 2] = ij[1]
            B[5, 3] = ij[7]; B[5, 5] = ij[5];  B[5, 6] = ij[11];  B[5, 8] = ij[9]; B[5, 9] = ij[14];  B[5, 11] = ij[13]
            C = self.C(100, 0.25)
            CB = np.matmul(C, B)
            BCB.append( np.matmul(np.transpose(B), CB))
            prim_num += 1

        return tf.constant(BCB)

    def C(self, E, v):
        C = np.zeros([6, 6])
        a = E/((1 + v)*(1 - 2*v))
        b = 1 - v
        c = 0.5 - v

        C[0, 0] = b
        C[1, 1] = b
        C[2, 2] = b

        C[3, 3] = c
        C[4, 4] = c
        C[5, 5] = c

        C *= a
        return np.asmatrix(C)





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
        return out

    def full_fem_differentiate(self, x):
        #data_size = x.get_shape().as_list()[1]
        #flatten_prim_idx = tf.constant(self.grid_dict['prim_points'], dtype=tf.int64)
        #n_filters = x.get_shape().as_list()[4]
        #inverse_jacobians = tf.constant(np.reshape(self.grid_dict['inv_j'], newshape=(-1, 4, 4)), dtype=tf.float32)



        #flat_output =  tf.reshape(x, shape=(tf.shape(x)[0], -1, tf.shape(x)[4]))
        #mask = tf.tile(tf.eye(3), multiples=(4, 1))


        #prim_aligned_features = tf.gather(flat_output, flatten_prim_idx, axis=1)
        #flat_prim_aligned_features = tf.reshape(prim_aligned_features, shape=(-1, tf.shape(prim_aligned_features)[1], 12))
        #flat_prim_aligned_features = tf.stack([flat_prim_aligned_features , flat_prim_aligned_features , flat_prim_aligned_features ], axis=3)
        #flat_prim_aligned_features = tf.einsum('ij, ktij ->ktij', mask, flat_prim_aligned_features)
        #flat_prim_aligned_features = tf.reduce_sum(flat_prim_aligned_features, axis=3)

       # differentiate = tf.einsum('jqk, ijqt -> ijkt', inverse_jacobians, prim_aligned_features)

       #primitive_features = tf.reshape(differentiate, shape=(-1, tf.shape(inverse_jacobians)[0], 4 * n_filters))
       # b = tf.reshape(primitive_features, shape=(tf.shape(primitive_features)[1], -1))

       # accumulate_features_on_points = tf.sparse_tensor_dense_matmul(self.multi_hot, b)

       # out = tf.reshape(accumulate_features_on_points, shape=(-1, data_size, data_size, data_size, 4 * n_filters))
        return self.B()

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

            out = tf.concat((x_s, out), axis=4)
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

        self.y = tf.placeholder(dtype=tf.float32, shape=(None, param_state_size, param_state_size, param_state_size, sdf_state_size + param_state_size),
                                    name='label_encodings')
        sequence_length = tf.placeholder(dtype=tf.int32, name='sequence_length')

        self.full_encoding = tf.placeholder(dtype=tf.float32, shape=(1, param_state_size, param_state_size, param_state_size, sdf_state_size + param_state_size),
                                                name='start_encoding')

        self.sdf_encodings = tf.placeholder(dtype=tf.float32, shape=(None, param_state_size, param_state_size, param_state_size, sdf_state_size),
                                                 name='sdf_encodings')


        with tf.variable_scope('Integrater_Network'):
            self.list_of_encodings = self.full_encoding

            def body(encoding, idx, list_of_encodings, sdf_encodings):
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

                encoding = encoding + tf.pad(T, tf.constant([[0, 0], [0, 0], [0, 0], [0, 0], [0, sdf_state_size]]), 'CONSTANT')

                updated_encoding = tf.slice(encoding, [0, 0, 0, 0, 0], [-1, -1, -1, -1, param_state_size])
                next_sdf_state = tf.slice(sdf_encodings, [idx + 1, 0, 0, 0, 0], [1, -1, -1, -1, -1])
                updated_encoding = tf.concat((updated_encoding, next_sdf_state), axis=4)

                list_of_encodings = tf.concat((list_of_encodings, updated_encoding), axis=0)

                idx = idx + 1

                return updated_encoding, idx, list_of_encodings, sdf_encodings

            def condition(encoding, idx, list_of_encodings, sdf_encodings):
                return tf.less(idx, sequence_length-1)

            idx = tf.constant(0, dtype=tf.int32)
            self.full_encoding, idx, self.list_of_encodings, self.sdf_encodings = tf.while_loop(condition, body, [
            self.full_encoding, idx, self.list_of_encodings, self.sdf_encodings], shape_invariants=[tf.TensorShape([1, param_state_size, param_state_size, param_state_size, sdf_state_size + param_state_size]),
                                                                                                         idx.get_shape(), tf.TensorShape([None, param_state_size, param_state_size, param_state_size, sdf_state_size + param_state_size]),
                                                                                                         tf.TensorShape([None,  param_state_size, param_state_size, param_state_size, sdf_state_size])])

            self.loss_int = tf.reduce_mean(tf.square(self.list_of_encodings - self.y))
            self.merged_int = tf.summary.scalar('Integrator Loss', self.loss_int)
            self.train_int = tf.train.AdamOptimizer(learning_rate=config.lr_max).minimize(self.loss_int)



class TestNetWork(layers):
    def __init__(self, voxel_side):
        super(TestNetWork, self).__init__()
        self.x = tf.placeholder(dtype=tf.float32, shape=(None,  voxel_side,  voxel_side,  voxel_side, 3), name='velocity')
        self.out = self.full_fem_differentiate(self.x)


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
                self.d_labels = tf.identity(self.central_difference(self.diff_labels), name='diff')
            else:
                dy = self.fem_differentiate(self.labels)
                self.d_labels = tf.identity(self.fem_differentiate(self.y), name='diff')

        with tf.variable_scope('Loss_Estimation'):
            self.l2_loss_v = tf.reduce_mean(tf.slice(tf.abs(self.y - self.labels), [0,1,1,1,0], [-1,-1,-1,-1,-1]))
            self.l2_loss_dv = tf.reduce_mean(tf.slice(tf.abs(dy - self.d_labels), [0, 1, 1, 1, 0], [-1, -1, -1, -1, -1]))

            self.loss = 0.2*self.l2_loss_dv + self.l2_loss_v

        with tf.variable_scope('Train'):
            self.step = tf.placeholder(dtype=tf.int32, name='step')
            self.lr = util.cosine_annealing(lr_max=config.lr_max, lr_min=config.lr_min, max_step= config.period, step=self.step)
            self.train = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5).minimize(self.loss)

        tf.summary.scalar('Encoder Loss', self.loss)
        tf.summary.scalar('Encoder Field Loss', self.l2_loss_v)
        tf.summary.scalar('Encoder Gradient Loss', self.l2_loss_dv)
        tf.summary.histogram('Latent State', self.full_encoding)
        self.merged = tf.identity(tf.summary.merge_all(), name='merged')

    def decoder_network(self, x, sdf, sb_blocks=1, n_filters=128):

        if not config.conv:
            output_dim = pow(self.param_state_size, 3)*n_filters
            x = tf.layers.dense(x, output_dim, activation=tf.nn.leaky_relu, name='decoder_dense')
            x = tf.reshape(x, shape=(-1, self.param_state_size, self.param_state_size, self.param_state_size, n_filters), name='decoder_reshape')

        x = tf.layers.conv3d(x, strides=(1, 1, 1),
                             kernel_size=(3, 3, 3),
                             filters=n_filters, padding='SAME',
                             name='latent_output_convolution'
                             , activation=tf.nn.leaky_relu)

        x = self.BB(x, int(self.q), FEM=config.use_fem, sb_blocks=sb_blocks, n_filters=n_filters)

        sdf = tf.layers.conv3d(sdf, strides=(1, 1, 1),
                             kernel_size=(3, 3, 3),
                             filters=n_filters, padding='SAME',
                             name='output_convolution_sdf',
                               activation=tf.nn.relu)


        x = self.differentiate_features(x, n_filters=n_filters, name='1')
        x = tf.identity(tf.concat((x, sdf), axis=4), name='merge_sdf')
        x = self.differentiate_features(x, n_filters=n_filters, name='2')
        x = tf.layers.conv3d(x,   strides=(1, 1, 1),
                                  kernel_size=(3, 3, 3),
                                  filters=3, padding='SAME',
                                  name='output_convolution')
        return x

    def encoder_network(self, x, sb_blocks=1, n_filters=128):

        x = self.BB(x, int(self.q), FEM=config.use_fem, upsample=False, sb_blocks=sb_blocks, n_filters=n_filters)

        if not config.conv:
            x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(x, self.param_state_size, activation=tf.nn.leaky_relu)
        else:
            x = tf.layers.conv3d(x, strides=(1, 1, 1),
                                 kernel_size=(3, 3, 3),
                                 filters=config.field_state, padding='SAME',
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
                                 filters=config.sdf_state, padding='SAME',
                                 name='latent_convolution',
                                 activation=tf.nn.leaky_relu)
        return x





class PINN:
    def __init__(self, layers, neurons):

        self.x = tf.placeholder(dtype=tf.float64, shape=(None, 1), name='x')
        self.z = tf.placeholder(dtype=tf.float64, shape=(None, 1), name='z')
        self.t = tf.placeholder(dtype=tf.float64, shape=(None, 1), name='time')

        mask = tf.placeholder(dtype=tf.bool, shape=(None, 1), name='mask')
        flag = tf.placeholder(dtype=tf.float64, shape=(None, 1), name='flag')
        flag = tf.tile(flag, [1, 2])
        inverted_scalar_mask = tf.math.logical_not(mask)
        vector_mask = tf.reshape(tf.tile(mask, [1, 2]), (-1, 2))
        mask = tf.reshape(tf.tile(mask, [1, 4]), (-1, 2, 2))

        inverted_mask = tf.math.logical_not(mask)
        inverted_vector_mask = tf.math.logical_not(vector_mask)

        self.ice_pressure = tf.placeholder(dtype=tf.float64, shape=(None, 1), name='surface_pressure')
        bulk_mod = tf.placeholder(dtype=tf.float64, shape=(None, 1), name='bulk_modulus')
        shear_mod = tf.placeholder(dtype=tf.float64, shape=(None, 1), name='shear_modulus')

        with tf.GradientTape(persistent=True) as t:
            t.watch(self.x)
            t.watch(self.z)

            r = tf.sqrt(tf.square(self.x)  + tf.square(self.z))
            inputs = tf.concat((r, self.x, self.z, self.t), axis=1)

            activation = tf.nn.leaky_relu
            activation2 = tf.nn.relu

            P = tf.layers.dense(inputs, neurons, activation=tf.nn.tanh, name='PINN_Layer_1')
            P = tf.layers.dense(P, neurons, activation=activation, name='PINN_Layer_2')
            P = tf.layers.dense(P, neurons, activation=activation, name='PINN_Layer_3')

            P = tf.layers.dense(P, neurons, activation=activation, name='PINN_Layer_4')
            P = tf.layers.dense(P, neurons, activation=activation, name='PINN_Layer_5')
            P = tf.layers.dense(P, neurons, activation=activation, name='PINN_Layer_6')

            P = tf.layers.dense(P, neurons, activation=activation, name='PINN_Layer_7')
            P = tf.layers.dense(P, neurons, activation=activation, name='PINN_Layer_8')

            P = tf.layers.dense(P, neurons, activation=activation, name='PINN_Layer_9')
            P = tf.layers.dense(P, neurons, activation=activation, name='PINN_Layer_10')

            self.u = tf.layers.dense(P, 2, activation=None, name='output_layer')

            self.u = (r-0.6) * self.u
            ux = tf.gather(self.u, [-1, 0], axis=1)
            uz = tf.gather(self.u, [-1, 1], axis=1)

            ux_x = t.gradient(ux, self.x)
            ux_z = t.gradient(ux, self.z)

            self.grad_ux = tf.concat((ux_x, ux_z), axis=1)

            uz_x = t.gradient(uz, self.x)
            uz_z = t.gradient(uz, self.z)

            self.grad_uz = tf.concat((uz_x, uz_z), axis=1)
        del t

        self.divergence_s = ux_x + uz_z
        self.grad_s = tf.identity(tf.stack((self.grad_ux, self.grad_uz), axis=2), name='grad_s')
        self.grad_st = tf.transpose(self.grad_s, name='grad_st', perm=[0, 2, 1])
        self.strain = self.grad_s + self.grad_st

        ident = tf.tile(tf.expand_dims(tf.eye(2, dtype=tf.float64), 0), multiples=(tf.shape(self.x)[0], 1, 1))
        shear_matrix = tf.reshape(tf.tile(shear_mod, multiples=[1, 4]), shape=(-1, 2, 2))
        self.stress = tf.tile(tf.expand_dims((bulk_mod + tf.cast(2.0 / 3.0, dtype=tf.float64) * shear_mod) * self.divergence_s, axis=2), multiples=[1, 2, 2])*ident + shear_matrix * self.strain



        self.xo = tf.reshape(tf.boolean_mask(self.x, inverted_scalar_mask, axis=0), shape=(-1, 1))
        self.zo = tf.reshape(tf.boolean_mask(self.z, inverted_scalar_mask, axis=0), shape=(-1, 1))
        self.uo = tf.reshape(tf.boolean_mask(self.u, inverted_vector_mask, axis=0), shape=(-1, 2))

        self.ice_pressure = tf.reshape(tf.boolean_mask(self.ice_pressure, inverted_scalar_mask, axis=0), shape=(-1, 1))
        self.lower_bound_u = tf.multiply(flag, self.u)

        self.stress_domain = tf.reshape(tf.boolean_mask(self.stress, mask, axis=0), (-1, 2, 2))
        self.stress_boundary = tf.reshape(tf.boolean_mask(self.stress, inverted_mask, axis=0), (-1, 2, 2))

        self.strain_domain = tf.reshape(tf.boolean_mask(self.strain, mask, axis=0), (-1, 2, 2))
        self.strain_boundary = tf.reshape(tf.boolean_mask(self.strain, inverted_mask, axis=0), (-1, 2, 2))

        self.n = tf.concat((self.x, self.z), axis=1) / tf.sqrt(
            (tf.pow(self.x, 2) + tf.pow(self.z, 2)))

        self.no = tf.concat((self.xo, self.zo), axis=1) / tf.sqrt(
            (tf.pow(self.xo, 2) + tf.pow(self.zo, 2)))

        self.energy = tf.einsum('kij,kij->k', self.stress, self.strain)

        self.boundary_energy =  self.ice_pressure*self.no - tf.einsum('kij, kj->ki', self.stress_boundary, self.no)
        self.pr_sample_boundary_energy = tf.reduce_sum(self.boundary_energy, axis=1)

        self.loss = tf.reduce_mean(tf.square(self.energy)) + 1000*tf.reduce_mean(tf.square(self.boundary_energy))
        self.train = tf.train.AdamOptimizer(tf.clip_by_value(self.loss/5000, 0.0, 0.0001)).minimize(self.loss)








