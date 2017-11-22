import os, glob
from datetime import datetime

import numpy as np
import tensorflow as tf

import images
from constants import NUM_CHANNELS, NUM_LABELS, IMG_PATCH_SIZE, PATCHES_PER_IMAGE

tf.app.flags.DEFINE_string('save_path', os.path.abspath("./runs"),
                           "Directory where to write event logs and checkpoint")
tf.app.flags.DEFINE_string('train_data_dir', os.path.abspath("./data/training"),
                           "Directory containing training images/ groundtruth/")
tf.app.flags.DEFINE_string('eval_data_dir', None, "Directory containing eval images")
tf.app.flags.DEFINE_boolean('restore_model', False, "Restore the model from previous checkpoint")
tf.app.flags.DEFINE_integer('num_epoch', 5, "Number of pass on the dataset during training")
tf.app.flags.DEFINE_integer('batch_size', 25, "Batch size of training instances")
tf.app.flags.DEFINE_float('lr', 0.01, "Initial learning rate")
tf.app.flags.DEFINE_float('momentum', 0.01, "Momentum")
tf.app.flags.DEFINE_float('lambda_reg', 5e-4, "Weight regularizer")
tf.app.flags.DEFINE_integer('seed', 2017, "Random seed for reproducibility")
tf.app.flags.DEFINE_integer('eval_every', 500, "Number of steps between evaluations")
tf.app.flags.DEFINE_integer('num_eval_images', 10, "Number of images to predict for an evaluation")

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
        self.eval_every = FLAGS.eval_every
        self.lambda_reg = FLAGS.lambda_reg
        self.num_eval_images = FLAGS.num_eval_images
        self.num_train_patches = None

        self.num_patches_eval = self.num_eval_images * PATCHES_PER_IMAGE

        assert self.num_patches_eval % self.batch_size == 0, \
            'batch_size {} should divide num_patches_eval {}'.format(self.batch_size, self.num_patches_eval)
        self.num_batches_eval = int(self.num_patches_eval / self.batch_size)


class ConvolutionalModel:
    """Two layers patch convolution model (baseline)"""

    def __init__(self, options, session):
        self._options: Options = options
        self._session = session
        self.patches, self.labels = self.prepare_data()
        np.random.seed(options.seed)

        self.summary_ops = []
        self.build_graph()

        summary_path = os.path.join(options.save_path, datetime.now().strftime("%Y-%m-%dT%Hh%Mm%Ss"))
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

        self._regularized_weights = [fc1_weights, fc1_biases, fc2_weights, fc2_biases]

        return out

    def cross_entropy_loss(self, labels, pred_logits):
        opts = self._options

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred_logits, labels=labels)
        loss = tf.reduce_mean(cross_entropy)

        # Add the regularization term to the loss.
        regularizers = tf.add_n([tf.nn.l2_loss(w) for w in self._regularized_weights])
        loss += opts.lambda_reg * regularizers
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
        opts = self._options
        self._images_to_display = tf.placeholder(tf.uint8, name="image_display")
        self._image_summary = tf.summary.image('samples', self._images_to_display, max_outputs=opts.num_eval_images)

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

        self._predictions = tf.argmax(predict_logits, axis=1)
        self._loss = loss
        self._patches_node = patches_node
        self._labels_node = labels_node
        self._predict_logits = predict_logits

        self.image_summary()
        self.summary_op = tf.summary.merge(self.summary_ops)

        self._missclassification_rate = tf.placeholder(tf.float64, name='misclassification_rate')
        self.misclassification_summary = tf.summary.scalar('misclassification_rate', self._missclassification_rate)

        # Properly initialize all variables.
        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()

    def prepare_data(self):
        """load images, create patches and labels"""
        opts = self._options

        train_data_dir = os.path.abspath(os.path.join(opts.train_data_dir, 'images/'))
        train_labels_dir = os.path.abspath(os.path.join(opts.train_data_dir, 'groundtruth/'))
        print("Train data {}".format(train_data_dir))
        print("Train labels {}".format(train_labels_dir))

        # Extract it into np arrays.
        train_images = images.load(train_data_dir)
        train_data = images.extract_patches(train_images, IMG_PATCH_SIZE)

        train_groundtruth = images.load(train_labels_dir)
        train_groundtruth_patches = images.extract_patches(train_groundtruth, IMG_PATCH_SIZE)
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

        num_errors = 0
        total = 0

        for batch_i, offset in enumerate(range(0, opts.num_train_patches - opts.batch_size, opts.batch_size)):
            batch_indices = indices[offset:offset + opts.batch_size]
            feed_dict = {
                self._patches_node: self.patches[batch_indices, :, :, :],
                self._labels_node: self.labels[batch_indices]
            }

            summary_str, _, l, lr, predictions, predictions, step = self._session.run(
                [self.summary_op, self._train, self._loss, self._lr, self._predict_logits, self._predictions,
                 self._global_step],
                feed_dict=feed_dict)

            self.summary_writer.add_summary(summary_str, global_step=step)

            num_errors += np.abs(self.labels[batch_indices] - predictions).sum()
            total += opts.batch_size

            # from time to time do full prediction on some images
            if step > 0 and step % opts.eval_every == 0:
                eval_predictions = np.ndarray(shape=(opts.num_patches_eval,))

                for batch in range(opts.num_batches_eval):
                    offset = batch * opts.batch_size

                    feed_dict = {
                        self._patches_node: self.patches[offset:offset + opts.batch_size, :, :, :],
                    }
                    eval_predictions[offset:offset + opts.batch_size] = self._session.run(self._predictions, feed_dict)

                # reconstruct original images
                original_images_patches = self.patches[:opts.num_patches_eval]
                new_shape = (opts.num_eval_images, PATCHES_PER_IMAGE, IMG_PATCH_SIZE, IMG_PATCH_SIZE, NUM_CHANNELS)
                original_images_patches = np.resize(original_images_patches, new_shape)
                images_display = images.images_from_patches(original_images_patches)

                # construct masks
                mask_patches = images.predictions_to_patches(eval_predictions, IMG_PATCH_SIZE)
                new_size = (opts.num_eval_images, PATCHES_PER_IMAGE, IMG_PATCH_SIZE, IMG_PATCH_SIZE, 1)
                mask_patches = np.resize(mask_patches, new_size)
                masks = images.images_from_patches(mask_patches)

                # blend for overlays
                overlays = images.overlays(images_display, masks)

                # display in summary
                image_sum, step = self._session.run([self._image_summary, self._global_step],
                                                    feed_dict={self._images_to_display: overlays})
                self.summary_writer.add_summary(image_sum, global_step=step)

        # save the missclassification rate over the epoch
        misclassification, step = self._session.run([self.misclassification_summary, self._global_step],
                                                    feed_dict={self._missclassification_rate: num_errors / total})
        self.summary_writer.add_summary(misclassification, global_step=step)
        self.summary_writer.flush()



    def save(self, epoch=0):
        opts = self._options
        model_data_dir = os.path.abspath(os.path.join(opts.train_data_dir, 'models/model-epoch-{:03d}.chkpt'.format(epoch)))
        save_path = self.saver.save(self._session, model_data_dir )
        # create checkpoint
        print("Model saved in file: {}".format(save_path))

    def restore(self):
        opts = self._options
        # Restore variables from disk
        model_data_dir = os.path.abspath(os.path.join(opts.train_data_dir, 'models/model-epoch-*.chkpt.meta'))
        # get latest saved model
        latest_model_filename = sorted(glob.glob(model_data_dir))[-1][:-5]
        self.saver.restore(self._session, latest_model_filename)
        print("Model restored from from file: {}".format(latest_model_filename))


def main(_):
    opts = Options()

    with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
        with tf.device("/gpu:0"):
            model = ConvolutionalModel(opts, session)

        if opts.restore_model:
            model.restore()
        else:
            for i in range(opts.num_epoch):
                print("Train epoch: {}".format(i))
                model.train()  # Process one epoch
                model.save(i)   # Save model to disk


if __name__ == '__main__':
    tf.app.run()
