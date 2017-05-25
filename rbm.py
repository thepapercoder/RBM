import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import os, utils


class RBM(object):

    def __init__(self, data_shape, n_rating=5, directory_name='rbm', model_name='', learning_rate=0.001, batch_size=128, n_epoch=50, k_step=1, hidden_dim=100):
        """
        RBM class - implementation of Restricted Boltzmann Machine for Collaborative Filtering
        Paper can be found at: http://www.machinelearning.org/proceedings/icml2007/papers/407.pdf

        :param data_shape: The shape of the input data ([number_of_user, number_of_movie])
        :param directory_name: Save folder
        :param model_name: Name of the RBM model to save
        :param learning_rate: Learning rate (How fast the model will learn)
        :param batch_size: Batch size - number of example will be fetch to model at one interation
        :param n_epoch: Number of time run through over dataset
        :param k_step: Number of gibb step
        :param hidden_dim: Dimension of hidden state
        """
        self.LEARNING_RATE = learning_rate
        self.BATCH_SIZE = batch_size
        self.N_EPOCH = n_epoch
        self.K_STEP = k_step
        self.HIDDEN_DIM = hidden_dim
        self.N_RATING = n_rating
        self.VISIBLE_DIM = data_shape[1]

        # Model path
        self.models_dir = 'models/'  # dir to save/restore models
        self.data_dir = 'data/'  # directory to store algorithm data
        self.summary_dir = 'logs/'  # directory to store tensorflow summaries
        self.directory_name = directory_name + "/" if directory_name[-1] != '/' else directory_name
        self.models_dir += self.directory_name
        self.data_dir += self.directory_name
        self.summary_dir += self.directory_name

        for d in [self.models_dir, self.data_dir, self.summary_dir]:
            if not os.path.isdir(d):
                os.mkdir(d)

        self.model_name = model_name
        if self.model_name == '':
            self.model_name = 'rbm-{}-{}-{}-{}-{}'.format(self.VISIBLE_DIM, self.HIDDEN_DIM,
                                                             self.N_EPOCH, self.BATCH_SIZE, self.LEARNING_RATE)

    def train(self, trX, valX, restore_previous_model=False):
        """
        Train function, take the input train and validation set, fit it to the model and output the RMSE score

        :param trX: Training set
        :param valX: Validation set
        :param restore_previous_model: Do we restore the pre-trained model or not?
        """
        ops.reset_default_graph()

        self._create_graph()
        merged = tf.summary.merge_all()
        init_op = tf.initialize_all_variables()

        with tf.Session() as self.sess:
            self.sess.run(init_op)

            if restore_previous_model:
                self.saver.restore(self.sess, self.models_dir + self.model_name)
                self.model_name += '-restored{}'.format(self.N_EPOCH)

            writer = tf.summary.FileWriter(self.summary_dir, self.sess.graph_def)

            for epoch in range(self.N_EPOCH):
                np.random.shuffle(trX)
                batches = [_ for _ in utils.gen_batches(trX, self.BATCH_SIZE)]

                total_error = 0.0
                for batch in batches:
                    result = self.sess.run([self.run_update, self.cost], feed_dict={self.X: batch})
                    total_error += result[1]
                print("Cost at epoch %s of training: %s" % (epoch, total_error/len(batches)))

                # if epoch % 5 == 0:
                #     result = self.sess.run([merged, self.cost], feed_dict={self.X: valX})
                #     summary_str = result[0]
                #     err = result[1]
                #
                #     writer.add_summary(summary_str, 1)
                #     print("Cost at epoch %s on validation set: %s" % (epoch, err))
            self.saver.save(self.sess, self.models_dir + self.model_name)

    def _create_variable(self):
        """
        Create variable for the graph, there are four variable we need to create,
         the first is X - the input data with the shape of [batch_size, number_of_movie]
         next is W, h_bias, v_bias is the weight and the bias.
        """
        self.X = tf.placeholder(tf.float32, [None, self.VISIBLE_DIM])

        abs_val = -4 * np.sqrt(6. / (self.HIDDEN_DIM + self.VISIBLE_DIM))
        self.W = tf.get_variable("W", [self.VISIBLE_DIM, self.HIDDEN_DIM], tf.float32,
                            tf.random_uniform_initializer(minval=-abs_val, maxval=abs_val))
        self.h_bias = tf.get_variable("h_bias", [self.HIDDEN_DIM], tf.float32,
                                 tf.constant_initializer(0.0))
        self.v_bias = tf.get_variable("v_bias", [self.VISIBLE_DIM], tf.float32,
                                 tf.constant_initializer(0.0))

    def _create_graph(self):
        """
        Create the graph model for the RBM, calculate and update all the weights and biases
        """
        self._create_variable()
        positive, negative, h_prob_0, h_prob_1, v_prob, v_sample = self.gibb_step(self.K_STEP, self.X)
        w_update = self.W.assign_add(self.LEARNING_RATE * (positive - negative))
        h_bias_update = self.h_bias.assign_add(self.LEARNING_RATE * tf.reduce_mean(h_prob_0 - h_prob_1, 0))
        v_bias_update = self.v_bias.assign_add(self.LEARNING_RATE * tf.reduce_mean(self.X - v_prob, 0))

        self.run_update = [w_update, h_bias_update, v_bias_update]

        self.cost = tf.reduce_mean(tf.abs(tf.subtract(self.X, v_sample)))
        _ = tf.summary.scalar("cost", self.cost)
        self.saver = tf.train.Saver()

    @staticmethod
    def sample_prob(probs):
        """
        Takes a tensor of probabilities (as from a sigmoidal activation) and samples from all the distributions
        :param probs: the probabilities tensor
        :return: sampled probabilities tensor
        """
        return tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs))))

    def sample_h_given_v(self, v_0):
        """
        Sample hidden layer when given visible layer
        :param v_0: visible layer sample
        """
        h_prob = tf.nn.sigmoid(tf.matmul(v_0, self.W) + self.h_bias)
        h_sample = self.sample_prob(h_prob)
        return h_prob, h_sample

    def sample_v_given_h(self, h_sample):
        """
        Sample visible layer when given hidden layer
        :param h_sample: hidden layer sample
        """
        # Remove all the movie have not been rating
        v_prob_tmp = tf.matmul(h_sample, tf.transpose(self.W)) + self.v_bias
        v_mask = tf.sign(self.X)
        v_prob_tmp = tf.reshape((v_prob_tmp * v_mask), [tf.shape(v_prob_tmp)[0], -1, self.N_RATING])
        v_prob = tf.nn.softmax(v_prob_tmp)
        v_prob = tf.reshape(v_prob, [tf.shape(v_prob_tmp)[0], -1])
        v_sample = self.sample_prob(v_prob)
        return v_prob, v_sample

    def gibb_step(self, k, v_0):
        """
        Perform gibb sampling for Contrastive Diversion or k-CD for k step
        :param k: number of gibb step
        :param v_0: input visible sample
        """
        v_prob = None
        v_sample = None
        h_prob_0 = None
        h_prob_1 = None
        positive = None
        for step in range(k):
            # Positive phase
            h_prob, h_sample = self.sample_h_given_v(v_0)
            if step == 0:
                h_prob_0 = h_prob
                positive = tf.matmul(tf.transpose(v_0), h_sample)
            # Negative phase
            v_prob, v_sample = self.sample_v_given_h(h_sample)
            h_prob_1, h_sample_1 = self.sample_h_given_v(v_prob)
            v_0 = v_prob
        negative = tf.matmul(tf.transpose(v_prob), h_prob_1)
        return positive, negative, h_prob_0, h_prob_1, v_prob, v_sample
