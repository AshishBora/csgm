"""Estimators for compressed sensing"""
# pylint: disable = C0301, C0103, C0111

import tensorflow as tf
import numpy as np
import utils
import scipy.fftpack as fftpack
import celebA_model_def


def dct2(image_channel):
    return fftpack.dct(fftpack.dct(image_channel.T, norm='ortho').T, norm='ortho')


def idct2(image_channel):
    return fftpack.idct(fftpack.idct(image_channel.T, norm='ortho').T, norm='ortho')


def vec(channels):
    image = np.zeros((64, 64, 3))
    for i, channel in enumerate(channels):
        image[:, :, i] = channel
    return image.reshape([-1])


def devec(vector):
    image = np.reshape(vector, [64, 64, 3])
    channels = [image[:, :, i] for i in range(3)]
    return channels


def wavelet_basis(path='../wavelet_basis.npy'):
    W_ = np.load(path)
    # W_ initially has shape (4096,64,64), i.e. 4096 64x64 images
    # reshape this into 4096x4096, where each row is an image
    # take transpose to make columns images
    W_ = W_.reshape((4096, 4096))
    W = np.zeros((12288, 12288))
    W[0::3, 0::3] = W_
    W[1::3, 1::3] = W_
    W[2::3, 2::3] = W_
    return W


def lasso_dct_estimator(hparams):  #pylint: disable = W0613
    """LASSO with DCT"""
    def estimator(A_val, y_batch_val, hparams):
        # One can prove that taking 2D DCT of each row of A,
        # then solving usual LASSO, and finally taking 2D ICT gives the correct answer.
        for i in range(A_val.shape[1]):
            A_val[:, i] = vec([dct2(channel) for channel in devec(A_val[:, i])])

        x_hat_batch = []
        for j in range(hparams.batch_size):
            y_val = y_batch_val[j]
            z_hat = utils.solve_lasso(A_val, y_val, hparams)
            x_hat = vec([idct2(channel) for channel in devec(z_hat)]).T
            x_hat = np.maximum(np.minimum(x_hat, 1), -1)
            x_hat_batch.append(x_hat)
        return x_hat_batch
    return estimator


def lasso_wavelet_estimator(hparams):  #pylint: disable = W0613
    """LASSO with Wavelet"""
    def estimator(A_val, y_batch_val, hparams):
        x_hat_batch = []
        W = wavelet_basis()
        WA = np.dot(W, A_val)
        for j in range(hparams.batch_size):
            y_val = y_batch_val[j]
            z_hat = utils.solve_lasso(WA, y_val, hparams)
            x_hat = np.dot(z_hat, W)
            x_hat_max = np.abs(x_hat).max()
            x_hat = x_hat / (1.0 * x_hat_max)
            x_hat_batch.append(x_hat)
        x_hat_batch = np.asarray(x_hat_batch)
        return x_hat_batch
    return estimator


def lasso_wavelet_ycbcr_estimator(hparams):  #pylint: disable = W0613
    """LASSO with Wavelet in YCbCr"""

    def estimator(A_val, y_batch_val, hparams):
        x_hat_batch = []

        W = wavelet_basis()
        # U, V = utils.RGB_matrix()
        # V = (V/127.5) - 1.0
        # U = U/127.5
        def convert(W):
            # convert W from YCbCr to RGB
            W_ = W.copy()
            V = np.zeros((12288, 1))
            # R
            V[0::3] = ((255.0/219.0)*(-16.0)) + ((255.0*0.701/112.0)*(-128.0))
            W_[:, 0::3] = (255.0/219.0)*W[:, 0::3] + (0.0)*W[:, 1::3] + (255.0*0.701/112.0)*W[:, 2::3]
            # G
            V[1::3] = ((255.0/219.0)*(-16.0)) - ((0.886*0.114*255.0/(112.0*0.587)) *(-128.0)) - ((255.0*0.701*0.299/(112.0*0.587))*(-128.0))
            W_[:, 1::3] = (255.0/219.0)*W[:, 0::3] - (0.886*0.114*255.0/(112.0*0.587))*W[:, 1::3] - (255.0*0.701*0.299/(112.0*0.587))*W[:, 2::3]
            # B
            V[2::3] = ((255.0/219.0)*(-16.0)) + ((0.886*255.0/(112.0))*(-128.0))
            W_[:, 2::3] = (255.0/219.0)*W[:, 0::3]  + (0.886*255.0/(112.0))*W[:, 1::3] + 0.0*W[:, 2::3]
            return W_, V

        # WU = np.dot(W, U.T)
        WU, V = convert(W)
        WU = WU/127.5
        V = (V/127.5) - 1.0
        WA = np.dot(WU, A_val)
        y_batch_val_temp = y_batch_val - np.dot(V.T, A_val)
        for j in range(hparams.batch_size):
            y_val = y_batch_val_temp[j]
            z_hat = utils.solve_lasso(WA, y_val, hparams)
            x_hat = np.dot(z_hat, WU) + V.ravel()
            print x_hat.shape
            x_hat_max = np.abs(x_hat).max()
            print x_hat_max
            x_hat = x_hat / (1.0 * x_hat_max)
            x_hat_batch.append(x_hat)
        x_hat_batch = np.asarray(x_hat_batch)
        return x_hat_batch

    return estimator


def dcgan_estimator(hparams):
    # pylint: disable = C0326

    # Get a session
    sess = tf.Session()

    # Set up palceholders
    A = tf.placeholder(tf.float32, shape=(hparams.n_input, hparams.num_measurements), name='A')
    y_batch = tf.placeholder(tf.float32, shape=(hparams.batch_size, hparams.num_measurements), name='y_batch')

    # Create the generator
    z_batch = tf.Variable(tf.random_normal([hparams.batch_size, 100]))
    x_hat_batch, restore_dict_gen, restore_path_gen = celebA_model_def.dcgan_gen(z_batch, sess, hparams)

    # Create the discriminator
    prob, restore_dict_discrim, restore_path_discrim = celebA_model_def.dcgan_discrim(x_hat_batch, sess, hparams)

    # measure the estimate
    measurement_is_sparse = (hparams.measurement_type in ['inpaint', 'superres'])
    y_hat_batch = tf.matmul(x_hat_batch, A, b_is_sparse=measurement_is_sparse, name='y2_batch')

    # define all losses
    m_loss1_batch =  tf.reduce_mean(tf.abs(y_batch - y_hat_batch), 1)
    m_loss2_batch =  tf.reduce_mean((y_batch - y_hat_batch)**2, 1)
    zp_loss_batch =  tf.reduce_sum(z_batch**2, 1)
    d_loss1_batch = -tf.log(prob)
    d_loss2_batch =  tf.log(1-prob)
    # deviation_loss = tf.reduce_mean((x_g_batch - x_hat_batch)**2, 1)

    # define total loss
    total_loss_batch = hparams.mloss1_weight * m_loss1_batch \
                     + hparams.mloss2_weight * m_loss2_batch \
                     + hparams.zprior_weight * zp_loss_batch \
                     + hparams.dloss1_weight * d_loss1_batch \
                     + hparams.dloss2_weight * d_loss2_batch
                     # + hparams.deviation_weight * deviation_loss
    total_loss = tf.reduce_mean(total_loss_batch)

    # Compute means for logging
    m_loss1 = tf.reduce_mean(m_loss1_batch)
    m_loss2 = tf.reduce_mean(m_loss2_batch)
    zp_loss = tf.reduce_mean(zp_loss_batch)
    d_loss1 = tf.reduce_mean(d_loss1_batch)
    d_loss2 = tf.reduce_mean(d_loss2_batch)

    # Set up gradient descent
    global_step = tf.Variable(0, trainable=False)
    learning_rate = utils.get_learning_rate(global_step, hparams)
    opt = utils.get_optimizer(learning_rate, hparams)
    update_op = opt.minimize(total_loss, var_list=[z_batch], global_step=global_step, name='update_op')

    # Intialize and restore model parameters
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    restorer_gen = tf.train.Saver(var_list=restore_dict_gen)
    restorer_discrim = tf.train.Saver(var_list=restore_dict_discrim)
    restorer_gen.restore(sess, restore_path_gen)
    restorer_discrim.restore(sess, restore_path_discrim)

    def estimator(A_val, y_batch_val, hparams):
        """Function that returns the estimated image"""
        best_keeper = utils.BestKeeper(hparams)
        feed_dict = {A: A_val, y_batch: y_batch_val}
        for i in range(hparams.num_random_restarts):
            sess.run([z_batch.initializer])
            for j in range(hparams.max_update_iter):
                _, lr_val, total_loss_val, \
                m_loss1_val, \
                m_loss2_val, \
                zp_loss_val, \
                d_loss1_val, \
                d_loss2_val = sess.run([update_op, learning_rate, total_loss,
                                        m_loss1,
                                        m_loss2,
                                        zp_loss,
                                        d_loss1,
                                        d_loss2], feed_dict=feed_dict)
                logging_format = 'rr {} iter {} lr {} total_loss {} m_loss1 {} m_loss2 {} zp_loss {} d_loss1 {} d_loss2 {}'
                print logging_format.format(i, j, lr_val, total_loss_val,
                                            m_loss1_val,
                                            m_loss2_val,
                                            zp_loss_val,
                                            d_loss1_val,
                                            d_loss2_val)
            x_hat_batch_val, total_loss_batch_val = sess.run([x_hat_batch, total_loss_batch], feed_dict=feed_dict)
            best_keeper.report(x_hat_batch_val, total_loss_batch_val)
        return best_keeper.get_best()

    return estimator
