import code
import glob
import os
from datetime import datetime

import numpy as np
import tensorflow as tf

import images
from constants import NUM_CHANNELS, IMG_PATCH_SIZE, FOREGROUND_THRESHOLD
from nn_utils import conv_conv_pool, upsample_concat

tf.app.flags.DEFINE_string('save_path', os.path.abspath("./runs"),
                           "Directory where to write checkpoints, overlays and submissions")
tf.app.flags.DEFINE_string('log_dir', os.path.abspath("./log_dir"),
                           "Directory where to write logfiles")
tf.app.flags.DEFINE_string('train_data_dir', os.path.abspath("./data/training"),
                           "Directory containing training images/ groundtruth/")
tf.app.flags.DEFINE_string('eval_data_dir', None, "Directory containing eval images")
tf.app.flags.DEFINE_boolean('restore_model', False, "Restore the model from previous checkpoint")
tf.app.flags.DEFINE_string('restore_date', None, "Restore the model from specific date")
tf.app.flags.DEFINE_boolean('interactive', False, "Spawn interactive Tensorflow session")
tf.app.flags.DEFINE_integer('num_epoch', 5, "Number of pass on the dataset during training")
tf.app.flags.DEFINE_integer('batch_size', 25, "Batch size of training instances")
tf.app.flags.DEFINE_float('lr', 0.01, "Initial learning rate")
tf.app.flags.DEFINE_float('momentum', 0.01, "Momentum")
tf.app.flags.DEFINE_float('lambda_reg', 5e-4, "Weight regularizer")
tf.app.flags.DEFINE_integer('seed', 2017, "Random seed for reproducibility")
tf.app.flags.DEFINE_integer('eval_every', 500, "Number of steps between evaluations")
tf.app.flags.DEFINE_integer('num_eval_images', 4, "Number of images to predict for an evaluation")
tf.app.flags.DEFINE_integer('patch_size', 128, "Size of the prediction image")
tf.app.flags.DEFINE_integer('gpu', -1, "GPU to run the model on")
tf.app.flags.DEFINE_integer('stride', 16, "Sliding delta for patches")
tf.app.flags.DEFINE_boolean('image_augmentation', False, "Augment training set of images with transformations")
tf.app.flags.DEFINE_float('dropout', 0.8, "Probability to keep an input")

FLAGS = tf.app.flags.FLAGS


class Options(object):
    """Options used by our model."""

    def __init__(self):
        self.save_path = FLAGS.save_path
        self.log_dir = FLAGS.log_dir
        self.train_data_dir = FLAGS.train_data_dir
        self.eval_data_dir = FLAGS.eval_data_dir
        self.restore_model = FLAGS.restore_model
        self.restore_date = FLAGS.restore_date
        self.num_epoch = FLAGS.num_epoch
        self.batch_size = FLAGS.batch_size
        self.seed = FLAGS.seed
        self.lr = FLAGS.lr
        self.momentum = FLAGS.momentum
        self.eval_every = FLAGS.eval_every
        self.lambda_reg = FLAGS.lambda_reg
        self.num_eval_images = FLAGS.num_eval_images
        self.interactive = FLAGS.interactive
        self.patch_size = FLAGS.patch_size
        self.stride = FLAGS.stride
        self.gpu = FLAGS.gpu
        self.image_augmentation = FLAGS.image_augmentation
        self.dropout = FLAGS.dropout


class ConvolutionalModel:
    """Two layers patch convolution model (baseline)"""

    def __init__(self, options, session):
        self._options = options
        self._session = session

        np.random.seed(options.seed)

        self.summary_ops = []
        self.build_graph()

        self.experiment_name = datetime.now().strftime("%Y-%m-%dT%Hh%Mm%Ss")
        experiment_path = os.path.abspath(os.path.join(options.save_path, self.experiment_name))
        summary_path = os.path.join(options.log_dir, self.experiment_name)
        self.summary_writer = tf.summary.FileWriter(summary_path, session.graph)

    def forward(self, patches):
        """Build the graph for the forward pass."""
        opts = self._options

        # recenter in [-.5, .5]
        data = patches - 0.5

    def make_unet(self, X):
        """Build a U-Net architecture
        Args:
            X (4-D Tensor): (N, H, W, C)
            training (1-D Tensor): Boolean Tensor is required for batchnormalization layers
        Returns:
            output (4-D Tensor): (N, H, W, 2)

        Notes:
            U-Net: Convolutional Networks for Biomedical Image Segmentation
            https://arxiv.org/abs/1505.04597
        Source:
            https://github.com/kkweon/UNet-in-Tensorflow/blob/master/train.py
        """
        net = X - 0.5  # TODO check
        net = tf.layers.conv2d(net, 3, (1, 1), name="color_space_adjust")

        dropout_keep = tf.placeholder_with_default(1.0, shape=())
        training = tf.placeholder_with_default(False, shape=())

        conv1, pool1 = conv_conv_pool(net, [8, 8], training, name="1", dropout_keep=dropout_keep)
        conv2, pool2 = conv_conv_pool(pool1, [16, 16], training, name="2", dropout_keep=dropout_keep)
        conv3, pool3 = conv_conv_pool(pool2, [32, 32], training, name="3", dropout_keep=dropout_keep)
        conv4, pool4 = conv_conv_pool(pool3, [64, 64], training, name="4", dropout_keep=dropout_keep)
        conv5 = conv_conv_pool(pool4, [128, 128], training, name="5", pool=False, dropout_keep=dropout_keep)

        up6 = upsample_concat(conv5, conv4, name="6")
        conv6 = conv_conv_pool(up6, [64, 64], training, name="6", pool=False, dropout_keep=dropout_keep)

        up7 = upsample_concat(conv6, conv3, name="7")
        conv7 = conv_conv_pool(up7, [32, 32], training, name="7", pool=False, dropout_keep=dropout_keep)

        up8 = upsample_concat(conv7, conv2, name="8")
        conv8 = conv_conv_pool(up8, [16, 16], training, name="8", pool=False, dropout_keep=dropout_keep)

        up9 = upsample_concat(conv8, conv1, name="9")
        conv9 = conv_conv_pool(up9, [8, 8], training, name="9", pool=False)

        self._dropout_keep = dropout_keep
        self._training = training

        return tf.layers.conv2d(conv9, 2, (1, 1), name='out', padding='same')

    def cross_entropy_loss(self, labels, pred_logits):
        batch_size, patch_height, patch_width = labels.shape

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=tf.reshape(pred_logits, (batch_size * patch_height * patch_width, 2)),
            labels=tf.reshape(labels, (batch_size * patch_height * patch_width,)))
        loss = tf.reduce_mean(cross_entropy)

        return loss

    def optimize(self, loss):
        """Build the graph to optimize the loss function."""
        opts = self._options

        # Use simple momentum for the optimization.
        optimizer = tf.train.AdamOptimizer(learning_rate=opts.lr)
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
                                      shape=(opts.batch_size, opts.patch_size, opts.patch_size, NUM_CHANNELS))
        labels_node = tf.placeholder(tf.int64,
                                     shape=(opts.batch_size, opts.patch_size, opts.patch_size))

        predict_logits = self.make_unet(patches_node)
        predictions = tf.nn.softmax(predict_logits, dim=3)
        predictions = predictions[:, :, :, 1]
        loss = self.cross_entropy_loss(labels_node, predict_logits)

        self.add_metrics_summary(labels_node, predictions)

        self._train = self.optimize(loss)
        self.summary_ops.append(tf.summary.scalar("loss", loss))

        self._loss = loss
        self._predictions = predictions
        self._patches_node = patches_node
        self._labels_node = labels_node
        self._predict_logits = predict_logits

        self.image_summary()
        self.summary_op = tf.summary.merge(self.summary_ops)

        self._missclassification_rate = tf.placeholder(tf.float64, name='misclassification_rate')
        self.misclassification_summary = tf.summary.scalar('misclassification_rate', self._missclassification_rate)

        # Properly initialize all variables.
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

        self.saver = tf.train.Saver()

    def add_metrics_summary(self, labels, predictions):
        """add accuracy, precision, recall, f1_score to tensorboard"""
        flat_labels = tf.layers.flatten(labels)
        flat_predictions = tf.layers.flatten(predictions)
        accuracy = tf.metrics.accuracy(labels=flat_labels, predictions=flat_predictions)[1]
        recall = tf.metrics.recall(labels=flat_labels, predictions=flat_predictions)[1]
        precision = tf.metrics.precision(labels=flat_labels, predictions=flat_predictions)[1]
        f1_score = 2 / (1 / recall + 1 / precision)

        self.summary_ops.append(tf.summary.scalar("accuracy", accuracy))
        self.summary_ops.append(tf.summary.scalar("recall", recall))
        self.summary_ops.append(tf.summary.scalar("precision", precision))
        self.summary_ops.append(tf.summary.scalar("f1_score", f1_score))

    def train(self, imgs, labels):
        """Train the model for one epoch

        params:
            imgs: [num_images, img_height, img_width, num_channel]
            labels: [num_images, num_patches_side, num_patches_side]
        """
        opts = self._options

        patches = images.extract_patches(imgs, opts.patch_size, stride=opts.stride, augmented=opts.image_augmentation)
        labels_patches = images.extract_patches(labels, opts.patch_size, stride=opts.stride, augmented=opts.image_augmentation)
        labels_patches = (labels_patches >= 0.5) * 1.

        num_train_patches = patches.shape[0]

        indices = np.arange(0, num_train_patches)
        np.random.shuffle(indices)

        num_errors = 0
        total = 0

        for batch_i, offset in enumerate(range(0, num_train_patches - opts.batch_size, opts.batch_size)):
            batch_indices = indices[offset:offset + opts.batch_size]
            feed_dict = {
                self._patches_node: patches[batch_indices, :, :, :],
                self._labels_node: labels_patches[batch_indices],
                self._dropout_keep: opts.dropout,
                self._training: False,  # TODO
            }

            summary_str, _, l, predictions, predictions, step = self._session.run(
                [self.summary_op, self._train, self._loss, self._predict_logits, self._predictions,
                 self._global_step],
                feed_dict=feed_dict)

            print("Batch {} Step {}".format(batch_i, step), end="\r")
            self.summary_writer.add_summary(summary_str, global_step=step)

            num_errors += np.abs(labels_patches[batch_indices] - predictions).sum()
            total += opts.batch_size

            # from time to time do full prediction on some images
            if step > 0 and step % opts.eval_every == 0:
                print()
                images_to_predict = imgs[:opts.num_eval_images, :, :, :]
                masks = self.predict(images_to_predict)
                overlays = images.overlays(images_to_predict, masks)

                # display in summary
                image_sum, step = self._session.run([self._image_summary, self._global_step],
                                                    feed_dict={self._images_to_display: overlays})
                self.summary_writer.add_summary(image_sum, global_step=step)
        print()

        # save the missclassification rate over the epoch
        misclassification, step = self._session.run([self.misclassification_summary, self._global_step],
                                                    feed_dict={self._missclassification_rate: num_errors / total})
        self.summary_writer.add_summary(misclassification, global_step=step)
        self.summary_writer.flush()

    def predict(self, imgs):
        """Run inference on `imgs` and return predicted masks

        imgs: [num_images, image_height, image_width, num_channel]
        returns: masks [num_images, images_height, image_width] with road = 1, other = 0
        """
        opts = self._options

        num_images = imgs.shape[0]
        print("Running prediction on {} images... ".format(num_images), end="")

        patches = images.extract_patches(imgs, opts.patch_size, opts.stride)
        num_patches = patches.shape[0]
        num_channel = imgs.shape[3]

        # patches padding to have full batches
        if num_patches % opts.batch_size != 0:
            num_extra_patches = opts.batch_size - (num_patches % opts.batch_size)
            extra_patches = np.zeros((num_extra_patches, opts.patch_size, opts.patch_size, num_channel))
            patches = np.concatenate([patches, extra_patches], axis=0)

        num_batches = int(patches.shape[0] / opts.batch_size)
        eval_predictions = np.ndarray(shape=(patches.shape[0], opts.patch_size, opts.patch_size))

        for batch in range(num_batches):
            offset = batch * opts.batch_size

            feed_dict = {
                self._patches_node: patches[offset:offset + opts.batch_size, :, :, :],
            }
            eval_predictions[offset:offset + opts.batch_size, :, :] = self._session.run(self._predictions, feed_dict)

        # remove padding
        eval_predictions = eval_predictions[0:num_patches]
        patches_per_image = int(num_patches / num_images)

        # construct masks
        new_shape = (num_images, patches_per_image, opts.patch_size, opts.patch_size, 1)
        masks = images.images_from_patches(eval_predictions.reshape(new_shape), stride=opts.stride)
        print("Done")
        return masks

    def save(self, epoch=0):
        opts = self._options
        model_data_dir = os.path.abspath(
            os.path.join(opts.save_path, self.experiment_name, 'model-epoch-{:03d}.chkpt'.format(epoch)))
        saved_path = self.saver.save(self._session, model_data_dir)
        # create checkpoint
        print("Model saved in file: {}".format(saved_path))

    def restore(self, date=None, epoch=None):
        """ Restores model from saved checkpoint

        date: which model should be restored (None is most recent)
        epoch: at which epoch model should be restored (None is most recent)
        """
        opts = self._options

        # get experiment name to restore from
        if date is None:
            dates = [date for date in glob.glob(os.path.join(opts.save_path, "*")) if os.path.isdir(date)]
            model_data_dir = sorted(dates)[-1]
        else:
            model_data_dir = os.path.abspath(os.path.join(opts.save_path, date))

        # get epoch construct final path
        if epoch is None:
            model_data_dir = os.path.abspath(os.path.join(model_data_dir, 'model-epoch-*.chkpt.meta'))
            model_data_dir = sorted(glob.glob(model_data_dir))[-1][:-5]
        else:
            model_data_dir = os.path.abspath(os.path.join(opts.save_path, 'model-epoch-{:03d}.chkpt'.format(epoch)))

        self.saver.restore(self._session, model_data_dir)
        print("Model restored from from file: {}".format(model_data_dir))


def main(_):
    opts = Options()
    if opts.gpu == -1:
        config = tf.ConfigProto()
    else:
        config = tf.ConfigProto(device_count={'GPU': opts.gpu}, allow_soft_placement=True)

    with tf.Graph().as_default(), tf.Session(config=config) as session:
        device = '/device:CPU:0' if opts.gpu == -1 else '/device:GPU:{}'.format(opts.gpu)
        print("Running on device {}".format(device))
        with tf.device(device):
            model = ConvolutionalModel(opts, session)

        if opts.restore_model:
            print("Restore date: {}".format(opts.restore_date))
            model.restore(date=opts.restore_date)

        if opts.num_epoch > 0:
            train_images, train_groundtruth = images.load_train_data(opts.train_data_dir, opts.patch_size)

            for i in range(opts.num_epoch):
                print("==== Train epoch: {} ====".format(i))
                tf.local_variables_initializer().run()  # Reset scores
                model.train(train_images, train_groundtruth)  # Process one epoch
                model.save(i)  # Save model to disk

        if opts.eval_data_dir:
            print("Running inference on eval data {}".format(opts.eval_data_dir))
            eval_images = images.load(opts.eval_data_dir)
            masks = model.predict(eval_images)
            masks = images.quantize_mask(masks, patch_size=IMG_PATCH_SIZE, threshold=FOREGROUND_THRESHOLD)
            overlays = images.overlays(eval_images, masks)
            save_dir = os.path.abspath(os.path.join(opts.save_path, model.experiment_name))
            images.save_all(overlays, save_dir)
            images.save_submission_csv(masks, save_dir, IMG_PATCH_SIZE)

        if opts.interactive:
            code.interact(local=locals())


if __name__ == '__main__':
    tf.app.run()
