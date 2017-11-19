import os

import numpy as np
import tensorflow as tf
from datetime import datetime

import images
from constants import NUM_CHANNELS, NUM_LABELS, IMG_PATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT

tf.app.flags.DEFINE_string('save_path', './runs', "Directory where to write event logs and checkpoint")
tf.app.flags.DEFINE_string('train_data_dir', './data/training', "Directory containing training images/ groundtruth/")
tf.app.flags.DEFINE_string('eval_data_dir', None, "Directory containing eval images")
tf.app.flags.DEFINE_boolean('restore_model', False, "Restore the model from previous checkpoint")
tf.app.flags.DEFINE_integer('num_epoch', 5, "Number of pass on the dataset during training")
tf.app.flags.DEFINE_integer('batch_size', 64, "Batch size of training instances")
tf.app.flags.DEFINE_float('lr', 0.01, "Initial learning rate")
tf.app.flags.DEFINE_float('momentum', 0.01, "Momentum")
tf.app.flags.DEFINE_integer('seed', 2017, "Random seed for reproducibility")

FLAGS = tf.app.flags.FLAGS


class Options(object):
    """Options used by our model."""

    def __init__(self):
        self.save_path = FLAGS.save_path
        self.train_data_dir = FLAGS.train_data_dir
        self.eval_data_dir = FLAGS.eval_data_dir
        self.restore_model = FLAGS.restore_model
        self.num_epoch = FLAGS.num_epoch
        self.batch_size = FLAGS.batch_size
        self.seed = FLAGS.seed
        self.lr = FLAGS.lr
        self.momentum = FLAGS.momentum


class ConvolutionalModel:
    """Two layers patch convolution model (baseline)"""

    def __init__(self, options, session):
        self._options = options
        self._session = session
        self.patches, self.labels = self.prepare_data()
        np.random.seed(options.seed)
        self.summary_ops = []

        self.build_graph()
        summary_path = os.path.join(options.save_path, datetime.now().isoformat('T', 'seconds'))
        self.summary_writer = tf.summary.FileWriter(summary_path, session.graph)

    def forward(self, patches):
        """Build the graph for the forward pass."""
        opts = self._options

        # recenter in [-.5, .5]
        data = patches - 0.5

        with tf.variable_scope('conv1'):
            conv1_weights = tf.Variable(
                tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                                    stddev=0.1,
                                    seed=opts.seed), name='weight')

            conv = tf.nn.conv2d(data,
                                conv1_weights,
                                strides=[1, 1, 1, 1],
                                padding='SAME',
                                name='conv')

            conv1_biases = tf.Variable(tf.zeros([32]), name='bias')

            relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases), name='relu')

            pool = tf.nn.max_pool(relu,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='SAME',
                                  name='pooling')

        with tf.variable_scope('conv2'):
            conv2_weights = tf.Variable(
                tf.truncated_normal([5, 5, 32, 64],
                                    stddev=0.1,
                                    seed=opts.seed), name='weight')
            conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]), name='bias')

            conv2 = tf.nn.conv2d(pool,
                                 conv2_weights,
                                 strides=[1, 1, 1, 1],
                                 padding='SAME',
                                 name='conv')

            relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases), name='relu')

            pool2 = tf.nn.max_pool(relu2,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME',
                                   name='pooling')

        # reshape the pooled layers
        reshape = tf.reshape(pool2, [pool2.shape[0], -1])

        with tf.variable_scope('fc1'):
            fc1_weights = tf.Variable(  # fully connected, depth 512.
                tf.truncated_normal([int(IMG_PATCH_SIZE / 4 * IMG_PATCH_SIZE / 4 * 64), 512],
                                    stddev=0.1,
                                    seed=opts.seed), name='weight')
            fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]), name='bias')

            hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases, name='relu')

        with tf.variable_scope('fc2'):
            fc2_weights = tf.Variable(
                tf.truncated_normal([512, NUM_LABELS],
                                    stddev=0.1,
                                    seed=opts.seed), name='weight')
            fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]), name='bias')
            out = tf.matmul(hidden, fc2_weights) + fc2_biases
            # out = tf.sigmoid(out, name='sigmoid')  TODO check

        return out

    def cross_entropy_loss(self, labels, pred_logits):
        # TODO add regularizer
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred_logits, labels=labels)
        loss = tf.reduce_mean(cross_entropy)
        return loss

    def optimize(self, loss):
        """Build the graph to optimize the loss function."""
        opts = self._options

        # Optimizer nodes.
        # Exponential learning rate decay.
        lr = tf.train.exponential_decay(
            learning_rate=opts.lr,
            global_step=opts.batch_size * self._global_step,
            decay_steps=opts.num_train_patches,
            decay_rate=0.95,
            staircase=True,
            name='learning_rate')

        self._lr = lr
        self.summary_ops.append(tf.summary.scalar('learning_rate', lr))

        # Use simple momentum for the optimization.
        optimizer = tf.train.GradientDescentOptimizer(lr)
        train = optimizer.minimize(loss, global_step=self._global_step)
        return train

    def image_summary(self):
        self._images_to_display = tf.placeholder(tf.uint8, name="image_display")
        self._image_summary = tf.summary.image('samples', self._images_to_display)

    def build_graph(self):
        """Build the graph for the full model."""
        opts = self._options

        # Global step: scalar, i.e., shape [].
        global_step = tf.Variable(0, name="global_step")
        self._global_step = global_step

        # data placeholders
        patches_node = tf.placeholder(tf.float32,
                                      shape=(opts.batch_size, IMG_PATCH_SIZE, IMG_PATCH_SIZE, NUM_CHANNELS))
        labels_node = tf.placeholder(tf.int64,
                                     shape=(opts.batch_size,))

        predict_logits = self.forward(patches_node)
        loss = self.cross_entropy_loss(labels_node, predict_logits)
        self.summary_ops.append(tf.summary.scalar("loss", loss))
        self._train = self.optimize(loss)

        self._predictions = tf.argmax(predict_logits, dimension=1)
        self._loss = loss
        self._patches_node = patches_node
        self._labels_node = labels_node
        self._predict_logits = predict_logits

        self.image_summary()

        # Properly initialize all variables.
        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()

    def prepare_data(self):
        """load images, create patches and labels"""
        opts = self._options

        train_data_dir = os.path.join(opts.train_data_dir, 'images/')
        train_labels_dir = os.path.join(opts.train_data_dir, 'groundtruth/')

        # Extract it into np arrays.
        train_images = images.load(train_data_dir)
        train_data = images.extract_patches(IMG_PATCH_SIZE, *train_images)

        train_groundtruth = images.load(train_labels_dir)
        train_groundtruth_patches = images.extract_patches(IMG_PATCH_SIZE, *train_groundtruth)
        train_labels = images.labels_for_patches(train_groundtruth_patches)

        opts.num_train_patches = train_labels.shape[0]

        print("Data directory:", opts.train_data_dir)
        print("Number of patches:", opts.num_train_patches)
        print("Ratio of road patches: {:.3f}".format(train_labels.mean()))

        return train_data, train_labels

    def train(self):
        """Train the model for one epoch."""
        opts = self._options

        indices = np.arange(0, opts.num_train_patches)
        np.random.shuffle(indices)

        # TODO save
        # TODO make prediction by image:
        # - feed uint8 images instead
        # - split patches only in model
        # - display some image with prediction

        summary_op = tf.summary.merge(self.summary_ops)

        n_errors = 0
        total = 0

        for batch_i, offset in enumerate(range(0, opts.num_train_patches - opts.batch_size, opts.batch_size)):
            batch_indices = indices[offset:offset + opts.batch_size]
            feed_dict = {
                self._patches_node: self.patches[batch_indices, :, :, :],
                self._labels_node: self.labels[batch_indices]
            }

            summary_str, _, l, lr, predictions, predictions, step = self._session.run(
                [summary_op, self._train, self._loss, self._lr, self._predict_logits, self._predictions, self._global_step],
                feed_dict=feed_dict)

            n_errors += np.abs(self.labels[batch_indices] - predictions).sum()
            total += opts.batch_size
            print('Average error rate: {}'.format(n_errors / total), end='\r')

            self.summary_writer.add_summary(summary_str, global_step=step)

        print()

        N_EVAL_IMAGE = 1
        n_patches_eval = int(N_EVAL_IMAGE * (IMAGE_WIDTH / IMG_PATCH_SIZE) * (IMAGE_HEIGHT / IMG_PATCH_SIZE))
        n_batches_eval = int(n_patches_eval / opts.batch_size)
        predictions = np.ndarray(n_patches_eval)

        assert IMAGE_WIDTH % IMG_PATCH_SIZE == 0
        assert n_patches_eval % opts.batch_size == 0, \
            'batch_size {} should divide n_patches_eval {}'.format(opts.batch_size, n_patches_eval)

        for batch in range(n_batches_eval):
            offset = batch * opts.batch_size

            feed_dict = {
                self._patches_node: self.patches[offset:offset + opts.batch_size, :, :, :],
            }
            predictions[offset:offset + opts.batch_size] = self._session.run(self._predictions, feed_dict)

        first_image_patches = self.patches[:n_patches_eval]
        first_image = images.image_from_patches(first_image_patches, IMAGE_WIDTH, IMAGE_HEIGHT)
        mask = images.image_from_predictions(predictions, IMG_PATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT)
        overlay = images.overlay(first_image, mask)
        overlay = np.array(overlay)
        overlay = overlay[np.newaxis, :, :, :]
        image_sum, step = self._session.run([self._image_summary, self._global_step], feed_dict={self._images_to_display: overlay})
        self.summary_writer.add_summary(image_sum, global_step=step)

        self.summary_writer.flush()


def main(_):
    opts = Options()

    with tf.Graph().as_default(), tf.Session() as session:
        with tf.device("/cpu:0"):
            model = ConvolutionalModel(opts, session)

        for i in range(opts.num_epoch):
            print("Train epoch: {}".format(i))
            model.train()  # Process one epoch


if __name__ == '__main__':
    tf.app.run()
