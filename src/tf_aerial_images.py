import os
import sys

import numpy as np
import tensorflow as tf

from constants import PIXEL_DEPTH, NUM_CHANNELS, NUM_LABELS, FOREGROUND_THRESHOLD, IMG_PATCH_SIZE
from images import load_images, extract_patches, overlay
from mask_to_submission import masks_to_submission

tf.app.flags.DEFINE_string('train_dir', '/tmp/train_satellite', "Directory where to write event logs and checkpoint")
tf.app.flags.DEFINE_string('train_data_dir', './data/training', "Directory containing training images/ groundtruth/")
tf.app.flags.DEFINE_string('eval_data_dir', None, "Directory containing eval images")
tf.app.flags.DEFINE_boolean('restore_model', False, "Restore the model from previous checkpoint")
tf.app.flags.DEFINE_integer('num_epoch', 10, "Number of pass on the dataset during training")
tf.app.flags.DEFINE_integer('batch_size', 16, "Batch size of training instances")
tf.app.flags.DEFINE_integer('seed', 2017, "Random seed for reproducibility")

FLAGS = tf.app.flags.FLAGS


def labels_for_patches(patches):
    foreground = patches.mean(axis=(1, 2)) > FOREGROUND_THRESHOLD
    labels = np.vstack([~foreground, foreground]).T * 1
    return labels


def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and 1-hot labels."""
    return 100.0 - (
        100.0 *
        np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) /
        predictions.shape[0])


def main(argv):
    data_dir = FLAGS.train_data_dir
    train_data_dir = os.path.join(data_dir, 'images/')
    train_labels_dir = os.path.join(data_dir, 'groundtruth/')

    # Extract it into np arrays.
    train_images = load_images(train_data_dir)
    train_data = extract_patches(IMG_PATCH_SIZE, *train_images)

    train_groundtruth = load_images(train_labels_dir)
    train_groundtruth_patches = extract_patches(IMG_PATCH_SIZE, *train_groundtruth)
    train_labels = labels_for_patches(train_groundtruth_patches)

    idx0 = (train_labels[:, 0] == 1).nonzero()[0]
    idx1 = (train_labels[:, 1] == 1).nonzero()[0]
    print('Number of data points per class: c0={} c1={}'.format(idx0.shape[0], idx1.shape[0]))

    print('Balancing training data...')
    min_c = min(idx0.shape[0], idx1.shape[0])
    new_indices = np.concatenate([idx0[0:min_c], idx1[0:min_c]])
    new_indices.sort(kind='mergesort')
    train_data = train_data[new_indices, :, :, :]
    train_labels = train_labels[new_indices]
    print("train data shape: {}".format(train_data.shape))
    print('Number of data points per class: c0={} c1={}'.format(min_c, min_c))

    train_size = train_labels.shape[0]

    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    train_data_node = tf.placeholder(
        tf.float32,
        shape=(FLAGS.batch_size, IMG_PATCH_SIZE, IMG_PATCH_SIZE, NUM_CHANNELS))
    train_labels_node = tf.placeholder(tf.float32,
                                       shape=(FLAGS.batch_size, NUM_LABELS))
    train_all_data_node = tf.constant(train_data)

    # The variables below hold all the trainable weights. They are passed an
    # initial value which will be assigned when when we call:
    # {tf.initialize_all_variables().run()}
    conv1_weights = tf.Variable(
        tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                            stddev=0.1,
                            seed=FLAGS.seed))
    conv1_biases = tf.Variable(tf.zeros([32]))
    conv2_weights = tf.Variable(
        tf.truncated_normal([5, 5, 32, 64],
                            stddev=0.1,
                            seed=FLAGS.seed))
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]))
    fc1_weights = tf.Variable(  # fully connected, depth 512.
        tf.truncated_normal([int(IMG_PATCH_SIZE / 4 * IMG_PATCH_SIZE / 4 * 64), 512],
                            stddev=0.1,
                            seed=FLAGS.seed))
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]))
    fc2_weights = tf.Variable(
        tf.truncated_normal([512, NUM_LABELS],
                            stddev=0.1,
                            seed=FLAGS.seed))
    fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))

    # Make an image summary for 4d tensor image with index idx
    def get_image_summary(img, idx=0):
        v = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
        img_w = img.get_shape().as_list()[1]
        img_h = img.get_shape().as_list()[2]
        min_value = tf.reduce_min(v)
        v = v - min_value
        max_value = tf.reduce_max(v)
        v = v / (max_value * PIXEL_DEPTH)
        v = tf.reshape(v, (img_w, img_h, 1))
        v = tf.transpose(v, (2, 0, 1))
        v = tf.reshape(v, (-1, img_w, img_h, 1))
        return v

    # Make an image summary for 3d tensor image with index idx
    def get_image_summary_3d(img):
        v = tf.slice(img, (0, 0, 0), (1, -1, -1))
        img_w = img.get_shape().as_list()[1]
        img_h = img.get_shape().as_list()[2]
        v = tf.reshape(v, (img_w, img_h, 1))
        v = tf.transpose(v, (2, 0, 1))
        v = tf.reshape(v, (-1, img_w, img_h, 1))
        return v

    def predict(images):
        """Predict a batch of images

        Shape:
            images: [N, height, width, channel]
        Return:
            mask of road prediction [N, height, width, 1]
        """
        n = images.shape[0]
        patches = extract_patches(IMG_PATCH_SIZE, *images)
        n_patch_axis = int(images.shape[1] / IMG_PATCH_SIZE)
        data_node = tf.constant(patches)
        output = tf.nn.softmax(model(data_node))
        output_prediction = s.run(output)
        patches_road = (output_prediction[:, 0] >= 0.5) * 1

        # expand the patches to get a mask image
        masks = np.repeat(
            np.repeat(
                patches_road.reshape(n, n_patch_axis, n_patch_axis),
                IMG_PATCH_SIZE, axis=2),
            IMG_PATCH_SIZE, axis=1)

        return masks

    def predict_all(input_directory, output_directory):
        """Predict the mask of every images in input_directory
        Output an overlay for each image and a submission.csv file"""
        print("Predictions of {} images".format(input_directory))
        if not os.path.isdir(output_directory):
            os.mkdir(output_directory)

        images = load_images(input_directory)
        masks = predict(images)

        print("Save predictions in {}".format(output_directory))
        for idx, (image, mask) in enumerate(zip(images, masks)):
            filename = os.path.join(output_directory, 'overlay_{:03}.png'.format(idx))
            overlay_img = overlay(image, mask)
            overlay_img.save(filename)

        csv_file_name = os.path.join(output_directory, "submission.csv")
        print("Writing predictions in CSV file {}".format(csv_file_name))
        masks_to_submission(csv_file_name, *masks)

    # We will replicate the model structure for the training subgraph, as well
    # as the evaluation subgraphs, while sharing the trainable parameters.
    def model(data, train=False):
        """The Model definition."""

        # recenter in [-.5, .5]
        data = data - 0.5

        conv = tf.nn.conv2d(data,
                            conv1_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')

        relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))

        pool = tf.nn.max_pool(relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

        conv2 = tf.nn.conv2d(pool,
                             conv2_weights,
                             strides=[1, 1, 1, 1],
                             padding='SAME')

        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

        pool2 = tf.nn.max_pool(relu2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

        pool_shape = pool2.get_shape().as_list()
        reshape = tf.reshape(
            pool2,
            [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])

        hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)

        # hidden = tf.nn.dropout(hidden, 0.5, seed=FLAGS.seed)
        out = tf.matmul(hidden, fc2_weights) + fc2_biases

        if train:
            summary_id = '_0'
            s_data = get_image_summary(data)
            tf.summary.image('summary_data' + summary_id, s_data)
            s_conv = get_image_summary(conv)
            tf.summary.image('summary_conv' + summary_id, s_conv)
            s_pool = get_image_summary(pool)
            tf.summary.image('summary_pool' + summary_id, s_pool)
            s_conv2 = get_image_summary(conv2)
            tf.summary.image('summary_conv2' + summary_id, s_conv2)
            s_pool2 = get_image_summary(pool2)
            tf.summary.image('summary_pool2' + summary_id, s_pool2)

        return out

    # Training computation: logits + cross-entropy loss.
    logits = model(train_data_node, True)  # FLAGS.batch_size*NUM_LABELS
    # print 'logits = ' + str(logits.get_shape()) + ' train_labels_node = ' + str(train_labels_node.get_shape())
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=train_labels_node))
    tf.summary.scalar('loss', loss)

    all_params_node = [conv1_weights, conv1_biases, conv2_weights, conv2_biases, fc1_weights, fc1_biases, fc2_weights,
                       fc2_biases]
    all_params_names = ['conv1_weights', 'conv1_biases', 'conv2_weights', 'conv2_biases', 'fc1_weights', 'fc1_biases',
                        'fc2_weights', 'fc2_biases']
    all_grads_node = tf.gradients(loss, all_params_node)
    all_grad_norms_node = []
    for i in range(0, len(all_grads_node)):
        norm_grad_i = tf.global_norm([all_grads_node[i]])
        all_grad_norms_node.append(norm_grad_i)
        tf.summary.scalar(all_params_names[i], norm_grad_i)

    # L2 regularization for the fully connected parameters.
    regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                    tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
    # Add the regularization term to the loss.
    loss += 5e-4 * regularizers

    # Optimizer: set up a variable that's incremented once per batch and
    # controls the learning rate decay.
    batch = tf.Variable(0)
    # Decay once per epoch, using an exponential schedule starting at 0.01.
    learning_rate = tf.train.exponential_decay(
        0.01,  # Base learning rate.
        FLAGS.batch_size * batch,  # Current index into the dataset.
        train_size,  # Decay step.
        0.95,  # Decay rate.
        staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)

    # Use simple momentum for the optimization.
    optimizer = tf.train.MomentumOptimizer(learning_rate,
                                           0.0).minimize(loss,
                                                         global_step=batch)

    # Predictions for the minibatch, validation set and test set.
    train_prediction = tf.nn.softmax(logits)
    # We'll compute them only once in a while by calling their {eval()} method.
    train_all_prediction = tf.nn.softmax(model(train_all_data_node))

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Create a local session to run this computation.
    with tf.Session() as s:

        if FLAGS.restore_model:
            # Restore variables from disk.
            saver.restore(s, FLAGS.train_dir + "/model.ckpt")
            print("Model restored.")

        else:
            # Run all the initializers to prepare the trainable parameters.
            tf.global_variables_initializer().run()

            # Build the summary operation based on the TF collection of Summaries.
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(FLAGS.train_dir,
                                                   graph=s.graph)
            print('Initialized!')
            # Loop through training steps.
            print('Total number of iterations = ' + str(int(FLAGS.num_epoch * train_size / FLAGS.batch_size)))

            training_indices = range(train_size)

            for iepoch in range(FLAGS.num_epoch):

                # Permute training indices
                perm_indices = np.random.permutation(training_indices)

                for step in range(int(train_size / FLAGS.batch_size)):

                    offset = (step * FLAGS.batch_size) % (train_size - FLAGS.batch_size)
                    batch_indices = perm_indices[offset:(offset + FLAGS.batch_size)]

                    # Compute the offset of the current minibatch in the data.
                    # Note that we could use better randomization across epochs.
                    batch_data = train_data[batch_indices, :, :, :]
                    batch_labels = train_labels[batch_indices]
                    # This dictionary maps the batch data (as a np array) to the
                    # node in the graph is should be fed to.
                    feed_dict = {train_data_node: batch_data,
                                 train_labels_node: batch_labels}

                    if step % 1000 == 0:

                        summary_str, _, l, lr, predictions = s.run(
                            [summary_op, optimizer, loss, learning_rate, train_prediction],
                            feed_dict=feed_dict)
                        # summary_str = s.run(summary_op, feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, step)
                        summary_writer.flush()

                        print('Epoch %.2f' % (float(step) * FLAGS.batch_size / train_size))
                        print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                        print('Minibatch error: %.1f%%' % error_rate(predictions,
                                                                     batch_labels))

                        sys.stdout.flush()
                    else:
                        # Run the graph and fetch some of the nodes.
                        _, l, lr, predictions = s.run(
                            [optimizer, loss, learning_rate, train_prediction],
                            feed_dict=feed_dict)

                # Save the variables to disk.
                save_path = saver.save(s, FLAGS.train_dir + "/model.ckpt")
                print("Model saved in file: %s" % save_path)

        predict_all(os.path.join(FLAGS.train_data_dir, 'images'), "predictions_training/")

        if FLAGS.eval_data_dir:
            predict_all(FLAGS.eval_data_dir, "predictions_eval/")


if __name__ == '__main__':
    tf.app.run()
