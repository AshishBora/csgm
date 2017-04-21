"""Model definitions for MNIST"""
# pylint: disable = C0301, C0103, R0914, C0111

import tensorflow as tf

def end_to_end(hparams):
    layer_sizes = [50, 200]

    y_batch = tf.placeholder(tf.float32, shape=(hparams.batch_size, hparams.num_measurements), name='y_batch')

    hidden = y_batch
    prev_hidden_size = hparams.num_measurements
    for i, hidden_size in enumerate(layer_sizes):
        layer_name = 'hidden{0}'.format(i)
        with tf.variable_scope(layer_name):
            weights = tf.get_variable('weights', shape=[prev_hidden_size, hidden_size])
            biases = tf.get_variable('biases', initializer=tf.zeros([hidden_size]))
            hidden = tf.nn.relu(tf.matmul(hidden, weights) + biases, name=layer_name)
        prev_hidden_size = hidden_size

    with tf.variable_scope('sigmoid_logits'):
        weights = tf.get_variable('weights', shape=[prev_hidden_size, 784])
        biases = tf.get_variable('biases', initializer=tf.zeros([784]))
        logits = tf.add(tf.matmul(hidden, weights), biases, name='logits')

    x_hat_batch = tf.nn.sigmoid(logits)

    restore_vars = ['hidden0/weights',
                    'hidden0/biases',
                    'hidden1/weights',
                    'hidden1/biases',
                    'sigmoid_logits/weights',
                    'sigmoid_logits/biases']
    restore_dict = {var.op.name: var for var in tf.global_variables() if var.op.name in restore_vars}

    return y_batch, x_hat_batch, restore_dict
