"""Script to train VAE on MNIST
This file is from : https://jmetzen.github.io/notebooks/vae.ipynb
"""
# pylint: skip-file


import numpy as np
import tensorflow as tf
from tensorflow import Variable
import matplotlib.pyplot as plt
import pickle as pkl
from tensorflow.examples.tutorials.mnist import input_data
import time

np.random.seed(0)
tf.set_random_seed(0)


def xavier_init(fan_in, fan_out, constant=1):
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)


class VariationalAutoencoder(object):
    """ Variation Autoencoder (VAE) with an sklearn-like interface implemented using TensorFlow.

    This implementation uses probabilistic encoders and decoders using Gaussian
    distributions and  realized by multi-layer perceptrons. The VAE can be learned
    end-to-end.

    See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
    """
    def __init__(self, network_architecture, transfer_fct=tf.nn.softplus,
                 learning_rate=0.001, batch_size=100):
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # tf Graph input
        self.x = tf.placeholder(tf.float32, [None, network_architecture["n_input"]],'x')

        # Create autoencoder network
        self._create_network()
        # Define loss function based variational upper-bound and
        # corresponding optimizer
        self._create_loss_optimizer()

        # Initializing the tensor flow variables
        init = tf.initialize_all_variables()

        # Launch the session
        self.sess = tf.InteractiveSession()
        self.sess.run(init)

    def _create_network(self):
		# Initialize autoencode network weights and biases
		self.network_weights = self._initialize_weights(**self.network_architecture)

		# Use recognition network to determine mean and
		# (log) variance of Gaussian distribution in latent
		# space
		self.z_mean, self.z_log_sigma_sq =             self._recognition_network(self.network_weights["weights_recog"],
									  self.network_weights["biases_recog"])

		# Draw one sample z from Gaussian distribution
		n_z = self.network_architecture["n_z"]
		eps = tf.random_normal((self.batch_size, n_z), 0, 1,
							   dtype=tf.float32)
		# z = mu + sigma*epsilon
		self.z = tf.add(self.z_mean,
						tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))
		# Use generator to determine mean of
		# Bernoulli distribution of reconstructed input
		self.x_reconstr_mean =self._generator_network(self.network_weights["weights_gener"],self.network_weights["biases_gener"])

    def _initialize_weights(self, n_hidden_recog_1, n_hidden_recog_2,
                            n_hidden_gener_1,  n_hidden_gener_2,
                            n_input, n_z):
        all_weights = dict()
        all_weights['weights_recog'] = {
            'h1': tf.Variable(xavier_init(n_input, n_hidden_recog_1),name='recognition_weights_1'),
            'h2': tf.Variable(xavier_init(n_hidden_recog_1, n_hidden_recog_2),name='recognition_weights_2'),
            'out_mean': tf.Variable(xavier_init(n_hidden_recog_2, n_z),name='recognition_weights_mu'),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_recog_2, n_z),'recognition_weights_sigma')}
        all_weights['biases_recog'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32),name='recognition_bias_1'),
            'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32),name='recognition_bias_2'),
            'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32),name='recognition_bias_mu'),
            'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32),name='recognition_bias_sigma')}
        all_weights['weights_gener'] = {
            'h1': tf.Variable(xavier_init(n_z, n_hidden_gener_1),'generator_weights_1'),
            'h2': tf.Variable(xavier_init(n_hidden_gener_1, n_hidden_gener_2),'generator_weights_2'),
            'out_mean': tf.Variable(xavier_init(n_hidden_gener_2, n_input),'generator_weights_mu')}
        all_weights['biases_gener'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_gener_1], dtype=tf.float32),'generator_bias_1'),
            'b2': tf.Variable(tf.zeros([n_hidden_gener_2], dtype=tf.float32),'generator_bias_2'),
            'out_mean': tf.Variable(tf.zeros([n_input], dtype=tf.float32),'generator_bias_mu')}
        return all_weights

    def _recognition_network(self, weights, biases):
        # Generate probabilistic encoder (recognition network), which
        # maps inputs onto a normal distribution in latent space.
        # The transformation is parametrized and can be learned.
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.x, weights['h1']),
                                           biases['b1']),name='recognition_activation_1')
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']),
                                           biases['b2']),name='recognition_activation_2')
        z_mean = tf.add(tf.matmul(layer_2, weights['out_mean']),
                        biases['out_mean'],'recognition_output_mu')
        z_log_sigma_sq =             tf.add(tf.matmul(layer_2, weights['out_log_sigma']),
                   biases['out_log_sigma'],'recognition_output_sigma')
        return (z_mean, z_log_sigma_sq)

    def _generator_network(self, weights, biases):
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto a Bernoulli distribution in data space.
        # The transformation is parametrized and can be learned.
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.z, weights['h1']),
                                           biases['b1']),name='generator_activation_1')
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']),
                                           biases['b2']),name='generator_activation_2')
        x_reconstr_mean =             tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['out_mean']),
                                 biases['out_mean']),name='generator_output')
        return x_reconstr_mean

    def _create_loss_optimizer(self):
        # The loss is composed of two terms:
        # 1.) The reconstruction loss (the negative log probability
        #     of the input under the reconstructed Bernoulli distribution
        #     induced by the decoder in the data space).
        #     This can be interpreted as the number of "nats" required
        #     for reconstructing the input when the activation in latent
		#     is given.
        # Adding 1e-10 to avoid evaluatio of log(0.0)

        reconstr_loss =-tf.reduce_sum(self.x * tf.log(1e-10 + self.x_reconstr_mean)+ (1-self.x) * tf.log(1e-10 + 1 - self.x_reconstr_mean),1,name='reconstruction_loss')
        # add reconstr_loss for gaussian output as well
        #reconstr_loss=-0.5* tf.reduce_sum(tf.squared_difference(self.x,self.x_reconstr_mean)/tf.exp(self.z_log_sigma_sq))
        # 2.) The latent loss, which is defined as the Kullback Leibler divergence
        ##    between the distribution in latent space induced by the encoder on
        #     the data and some prior. This acts as a kind of regularizer.
        #     This can be interpreted as the number of "nats" required
        #     for transmitting the the latent space distribution given
        #     the prior.
        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq
                                           - tf.square(self.z_mean)
                                           - tf.exp(self.z_log_sigma_sq), 1,name='latent_loss')
        self.cost = tf.reduce_mean(reconstr_loss + latent_loss,name='total_loss')   # average over batch
        # Use ADAM optimizer
        self.optimizer =tf.train.AdamOptimizer(learning_rate=self.learning_rate,name='AdamOptimizer').minimize(self.cost)

    def partial_fit(self, X):
        """Train model based on mini-batch of input data.

        Return cost of mini-batch.
        """
        opt, cost = self.sess.run((self.optimizer, self.cost),
                                  feed_dict={self.x: X})
        return cost

    def transform(self, X):
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.z_mean, feed_dict={self.x: X})

	def reconstruct(self, X):
		""" Use VAE to reconstruct given data. """
		return self.sess.run(self.x_reconstr_mean,
							 feed_dict={self.x: X})


def train(network_architecture, learning_rate=0.001,batch_size=100, training_epochs=10, display_step=1):
	vae=VariationalAutoencoder(network_architecture)
	# Training cycle
	for epoch in range(training_epochs):
		avg_cost = 0.
		total_batch = int(n_samples / batch_size)
		# Loop over all batches
		for i in range(total_batch):
			batch_xs, _ = mnist.train.next_batch(batch_size)

			# Fit training using batch data
			cost = vae.partial_fit(batch_xs)
			# Compute average loss
			avg_cost += cost / n_samples * batch_size

		# Display logs per epoch step
		if epoch % display_step == 0:
			print "Epoch:", '%04d' % (epoch+1),                   "cost=", "{:.9f}".format(avg_cost)
	return vae


if __name__=='__main__':

	mnist = input_data.read_data_sets('../data/MNIST', one_hot=True)
	n_samples = mnist.train.num_examples

	network_architecture =     dict(n_hidden_recog_1=500, # 1st layer encoder neurons
			 n_hidden_recog_2=500, # 2nd layer encoder neurons
			 n_hidden_gener_1=500, # 1st layer decoder neurons
			 n_hidden_gener_2=500, # 2nd layer decoder neurons
			 n_input=784, # MNIST data input (img shape: 28*28)
			 n_z=20)  # dimensionality of latent space

	vae = train(network_architecture,training_epochs=75)
	time_name=time.strftime("%d-%m-%Y_%H:%M:%S")
	model_name='../models/'+time_name+'.ckpt'
	f1=open('../models/'+time_name+'_architecture.pkl','w')
	pkl.dump(network_architecture,f1)
	f1.close()

	saver=tf.train.Saver()
	saver.save(vae.sess,model_name)
