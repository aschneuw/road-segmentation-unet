import tensorflow as tf

import images
from constants import IMG_PATCH_SIZE


class Summary:
    """Handle tensorflow summaries."""

    def __init__(self, options, session, summary_path):
        self._options = options
        self._session = session
        self._summary_writer = tf.summary.FileWriter(summary_path, session.graph)
        self.summary_ops = []

    def flush(self):
        self._summary_writer.flush()

    def add(self, summary_str, global_step=None):
        self._summary_writer.add_summary(summary_str, global_step)

    def get_summary_op(self, scalars):
        for key, value in scalars.items():
            self.summary_ops.append(tf.summary.scalar(key, value))

        return tf.summary.merge(self.summary_ops)

    def initialize_eval_summary(self):
        opts = self._options
        self._images_to_display = tf.placeholder(tf.uint8, name="image_display")
        self._image_summary = [
            tf.summary.image('eval_images', self._images_to_display, max_outputs=opts.num_eval_images)]

        self._masks_to_display = tf.placeholder(tf.uint8, name="mask_display")
        self._image_summary.append(
            tf.summary.image('eval_masks', self._masks_to_display, max_outputs=opts.num_eval_images))

        # eval data placeholders
        self._eval_predictions = tf.placeholder(tf.int64, name="eval_predictions")
        self._eval_labels = tf.placeholder(tf.int64, name="eval_labels")

        predictions = self._eval_predictions
        labels = self._eval_labels

        accuracy, recall, precision, f1_score = self.get_prediction_metrics(labels, predictions)

        self._image_summary.append(tf.summary.scalar("eval accuracy", accuracy))
        self._image_summary.append(tf.summary.scalar("eval recall", recall))
        self._image_summary.append(tf.summary.scalar("eval precision", precision))
        self._image_summary.append(tf.summary.scalar("eval f1_score", f1_score))
        self._image_summary = tf.summary.merge(self._image_summary)

    def initialize_overlap_summary(self):
        opts = self._options
        self._overlap_images = tf.placeholder(tf.uint8, name="overlap_images")
        self._overlap_summary = tf.summary.image('groundtruth_vs_prediction', self._overlap_images,
                                                 max_outputs=opts.num_eval_images)

    def initialize_train_summary(self):
        self._train_predictions = tf.placeholder(tf.int64, name="train_predictions")
        self._train_labels = tf.placeholder(tf.int64, name="train_labels")

        predictions = self._train_predictions
        labels = self._train_labels

        accuracy, recall, precision, f1_score = self.get_prediction_metrics(labels, predictions)

        self._train_summary = [tf.summary.scalar("train accuracy", accuracy)]
        self._train_summary.append(tf.summary.scalar("train recall", recall))
        self._train_summary.append(tf.summary.scalar("train precision", precision))
        self._train_summary.append(tf.summary.scalar("train f1_score", f1_score))
        self._train_summary = tf.summary.merge(self._train_summary)

    def initialize_missclassification_summary(self):
        self._missclassification_rate = tf.placeholder(tf.float64, name='misclassification_rate')
        self._misclassification_summary = tf.summary.scalar('misclassification_rate', self._missclassification_rate)

    def add_to_overlap_summary(self, true_labels, predicted_labels, global_step):
        overlapped = images.overlap_pred_true(predicted_labels, true_labels)

        feed_dict_eval = {
            self._overlap_images: overlapped
        }
        overlap_summary, step = self._session.run([self._overlap_summary, global_step],
                                                  feed_dict=feed_dict_eval)
        self._summary_writer.add_summary(overlap_summary, global_step=step)

    def add_to_eval_patch_summary(self, labels):
        opts = self._options
        eval_labels = labels[:opts.num_eval_images, :, :]
        eval_labels = images.img_float_to_uint8(eval_labels)
        eval_labels = tf.expand_dims(eval_labels, -1)

        image_summary = tf.summary.image("eval_groundtruth", eval_labels, max_outputs=eval_labels.shape[0])

        image_summary_ = self._session.run(image_summary)
        self._summary_writer.add_summary(image_summary_, 0)

    def add_to_pixel_missclassification_summary(self, num_errors, total, global_step):
        misclassification, step = self._session.run([self._misclassification_summary, global_step],
                                                    feed_dict={self._missclassification_rate: num_errors / total})
        self._summary_writer.add_summary(misclassification, global_step=step)

    def add_to_eval_summary(self, masks, overlays, labels, global_step):
        opts = self._options

        eval_pred = self.img_to_label_patches(masks)
        eval_true = self.img_to_label_patches(labels[:opts.num_eval_images, :, :])

        feed_dict_eval = {
            self._masks_to_display: images.img_float_to_uint8(masks),
            self._images_to_display: overlays,
            self._eval_predictions: eval_pred,
            self._eval_labels: eval_true
        }

        image_sum, step = self._session.run([self._image_summary, global_step],
                                            feed_dict=feed_dict_eval)
        self._summary_writer.add_summary(image_sum, global_step=step)

    def add_to_training_summary(self, predictions, labels, global_step):
        train_predictions = self.img_to_label_patches(predictions)
        train_labels = self.img_to_label_patches(labels)

        feed_dict_train = {
            self._train_predictions: train_predictions,
            self._train_labels: train_labels
        }

        train_sum, step = self._session.run([self._train_summary, global_step],
                                            feed_dict=feed_dict_train)
        self._summary_writer.add_summary(train_sum, global_step=step)

    def img_to_label_patches(self, img, patch_size=IMG_PATCH_SIZE):
        img = images.extract_patches(img, patch_size)
        img = images.labels_for_patches(img)
        img.resize((img.shape[0], patch_size, patch_size))

        return img

    def get_prediction_metrics(self, labels, predictions):
        accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions)[1]
        recall = tf.metrics.recall(labels=labels, predictions=predictions)[1]
        precision = tf.metrics.precision(labels=labels, predictions=predictions)[1]
        f1_score = 2 / (1 / recall + 1 / precision)

        return accuracy, recall, precision, f1_score
