import tensorflow as tf
import numpy as np
import fetch_data as fd
class layers:

    def SB(self, x, j, n_filters=128, n_blocks=1, upsample=True):
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

    def BB(self, x, bb_blocks=1, n_filters=128, upsample=True):
        for q in range(bb_blocks):
            x = self.SB(x, q, n_filters=n_filters, n_blocks=bb_blocks, upsample=upsample)
            if upsample:
                x = self.upsample(x)
            else:
                x = tf.nn.max_pool3d(input=x, ksize=(1, 1, 1, 1, 1), padding='VALID', strides=(1, 2, 2, 2, 1))
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

class IntegratorNetwork:

    def __init__(self, param_state_size=8, sequence_length = 30):

        self.y = tf.placeholder(dtype=tf.float32, shape=(sequence_length, 2*param_state_size), name='label_encodings')

        self.full_encoding = tf.placeholder(dtype=tf.float32, shape=(1, 1, 2*param_state_size), name='start_encoding')
        self.parm_encodings = tf.placeholder(dtype=tf.float32, shape=(sequence_length, param_state_size), name= 'parameter_encodings')

        self.list_of_encodings = self.full_encoding

        def body(encoding, idx, list_of_encodings, parm_encodings):
            f1 = tf.nn.dropout(tf.layers.dense(encoding, 1024, activation=tf.nn.elu), 0.9)
            f2 = tf.nn.dropout(tf.layers.dense(f1, 512, activation=tf.nn.elu), 0.9)
            T = tf.nn.dropout(tf.layers.dense(f2, param_state_size, activation=tf.nn.elu), 0.9)

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
        self.train = tf.train.AdamOptimizer(0.0001).minimize(self.loss)


class NetWork(layers):

    def __init__(self, voxel_side, param_state_size=8):

        self.param_state_size = param_state_size
        self.vx_log2 = np.log2(voxel_side)
        self.q = self.vx_log2 - np.log2(param_state_size)

        self.x = tf.placeholder(dtype=tf.float32, shape=(None,  voxel_side,  voxel_side,  voxel_side, 3), name='velocity')
        self.sdf = tf.placeholder(dtype=tf.float32, shape=(None,  voxel_side,  voxel_side,  voxel_side, 1), name='sdf')

        self.c = self.encoder_network(self.x)

        with tf.variable_scope('boundary_conditions'):
            self.encoded_sdf = self.encoder_network(self.sdf)

        self.full_encoding = tf.concat((self.c, self.encoded_sdf), axis=1)


        self.y = self.decoder_network(self.full_encoding)

        self.dy = self.central_difference(self.y)
        self.dx = self.central_difference(self.x)

        self.l2_loss_v = tf.reduce_mean(tf.square(self.y - self.x))
        self.l2_loss_dv = tf.reduce_mean(tf.square(self.dy - self.dx))

        self.loss = self.l2_loss_v + self.l2_loss_dv
        self.train = tf.train.AdamOptimizer(0.0001).minimize(self.loss)



    def decoder_network(self, x, sb_blocks=1):

        output_dim = pow(self.param_state_size, 3)*128
        x = tf.layers.dense(x, output_dim, activation=tf.nn.leaky_relu)
        x = tf.reshape(x, shape=(-1, self.param_state_size, self.param_state_size, self.param_state_size, 128))

        x = self.BB(x, int(self.q))
        x = tf.layers.conv3d(x,   strides=(1, 1, 1),
                                  kernel_size=(3, 3, 3),
                                  filters=3, padding='SAME',
                                  activation=tf.nn.leaky_relu,
                                  name='output_convolution')
        return x

    def encoder_network(self, x,  sb_blocks=1):

        x = self.BB(x, int(self.q), upsample=False)
        x = tf.contrib.layers.flatten(x)
        c = tf.layers.dense(x, self.param_state_size, activation=tf.nn.leaky_relu)
        return c


   
