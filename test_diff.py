import tensorflow as tf
import numpy as np
import fetch_data as fd
import network
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

def sample_rheology(theta, r):
    E = 10
    bulk_stress = 10
    v = 0.25
    shear_stress = E / (2*(1 + v))
    bulk_stress = E / (3*(1 - 2*v))

    return bulk_stress, shear_stress, v

def sample_surface_pressure(x, z):
    pressure = 0
    if z > 0:
        pressure = 1
    return pressure

def xyz(theta, r):
    x = r * np.cos(theta)
    z = r * np.sin(theta)
    return x, z

def sample_domain_proxy(x, z, loss, R=1, samples=1):
    list = np.transpose([np.squeeze(x), np.squeeze(z), np.squeeze(loss)])
    batch_size = np.shape(x)[0]

    if samples > batch_size:
        samples =  batch_size

    list = list[np.random.normal(list[:,2], 20).argsort()]

    list = list[-samples:, :]

    list_r_th = list
    list_r_th[:, 0] = np.sqrt(np.power(list[:, 0], 2) + np.power(list[:, 1], 2))
    list_r_th[:, 1] = np.arctan2(list[:, 0], list[:, 1])
    list_r_th[:, 0] = np.clip(np.random.normal(list_r_th[:, 0], 0.1), 0.6 * R, 0.99*R)
    list_r_th[:, 1] = np.random.normal(list_r_th[:, 1], 1)

    x = list_r_th[:, 0] * np.cos(list_r_th[:, 1])
    z = list_r_th[:, 0] * np.sin(list_r_th[:, 1])

    sample_points = []
    bulk_stress, shear, v = sample_rheology(0, 0)
    for i in range(samples):
        sample_points.append([x[i], z[i], bulk_stress, shear, 0, 1, 0 ])

    return sample_points


def sample_domain(R):
    theta = 2*np.pi*np.random.rand()
    r = np.clip(R*(0.6 + 0.4*np.random.rand()), 0.6*R, R)
    x, z = xyz(theta, r)
    bulk, shear, v = sample_rheology(theta, r)

    return x, z, bulk, shear, 0, 1, 0

def boundary_displacement(x, z):
    if z > 0.9:
        return
def sample_boundary(R):
    theta = 2*np.pi*np.random.rand()
    flag = 0
    pressure=0
    if np.random.rand() < 0.5:
        x, z = xyz(theta, R)
        pressure = sample_surface_pressure(x, z)
    else:
        x, z = xyz(theta, 0.6*R)
        flag=1

    bulk, shear, v = sample_rheology(theta, R)

    return x, z, bulk, shear, pressure, 0, flag

def boundary_batch(batch_size, R):
    sample_points = []
    for i in range(batch_size):
        sample_points.append(sample_boundary(R))
    return sample_points

def domain_batch(batch_size, R):
    sample_points = []
    for i in range(batch_size):
        sample_points.append(sample_domain(R))
    return sample_points


tf.reset_default_graph()
net = network.PINN(5, 128)
init_s = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_s)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axis([-1, 1, -1, 1])
    db = domain_batch(1, 1)
    bb = boundary_batch(3, 1)
    fb = np.asmatrix(np.squeeze([np.concatenate((db, bb), axis=0)]))
    last_energy = 0
    d_energy = 0

    for i in range(1000000):
        time = np.random.rand() * np.ones([np.shape(fb[:, 1])[0], np.shape(fb[:, 1])[1]])
        feed_dict = {'x:0': fb[:, 0], 'z:0': fb[:, 1], 'bulk_modulus:0': fb[:, 2], 'shear_modulus:0': fb[:, 3], 'surface_pressure:0': fb[:, 4], 'mask:0': fb[:, 5],'flag:0': fb[:, 6], 'time:0': time}

        surface_energy, energy, divergence, _, loss, x, z, xo, zo, u = sess.run([net.pr_sample_boundary_energy, net.energy, net.divergence_s, net.train, net.loss, net.x, net.z,  net.xo, net.zo, net.u], feed_dict=feed_dict)

        db_random = sample_domain_proxy(x, z, divergence, R=1, samples=3)

        print(loss)

        v_length = np.sqrt(np.array(np.sum(np.square(u), axis=1)))
        ax.quiver(np.array(x).flatten(), np.array(z).flatten(),
                      np.array(u[:, 0]).flatten(), np.array(u[:, 1]).flatten())

        ax.scatter(np.array(x).flatten(), np.array(z).flatten(), vmin=0., vmax=0.1, s = 100*np.array(energy).flatten(), c = np.array(energy).flatten())
        ax.scatter(np.array(xo).flatten(), np.array(zo).flatten(), vmin=0., vmax=0.1,
                       s=100 * np.array(surface_energy).flatten(), c=np.array(surface_energy).flatten())
        plt.pause(0.00000001)
        ax.clear()

        if np.shape(fb)[0] < 1500:
            db_new = domain_batch(3, 1)
            bb_new = boundary_batch(1, 1)
            fb_new = np.asmatrix(np.squeeze([np.concatenate((db_new, bb_new, db_random), axis=0)]))
            fb = np.asmatrix(np.squeeze([np.concatenate((fb_new, fb), axis=0)]))

        else:
            db = domain_batch(1, 1)
            bb = boundary_batch(3, 1)
            db_random = sample_domain_proxy(x, z, divergence, R=1, samples=3)
            fb_sw = np.asmatrix(np.squeeze([np.concatenate((db, bb, db_random), axis=0)]))

            for i in range(np.shape(fb_sw)[0]):
                index = int(np.floor(np.shape(fb)[0]*np.random.rand()))
                fb[index,:] = fb_sw[i, :]










