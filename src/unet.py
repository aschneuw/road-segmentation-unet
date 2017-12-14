import tensorflow as tf


def forward(self, X, num_layers, root_size):
    dropout_keep = tf.placeholder_with_default(1.0, shape=(), name="dropout_keep")
    self._dropout_keep = dropout_keep

    net = X - 0.5
    net = tf.layers.conv2d(net, 3, (1, 1), name="color_space_adjust")

    num_filters = root_size
    conv = []

    for layer_i in range(num_layers):
        if dropout_keep is not None:
            net = tf.nn.dropout(net, dropout_keep)

        with tf.variable_scope("conv_{}".format(layer_i)):
            net = tf.layers.conv2d(net, num_filters, (3, 3), padding='valid', name="conv1")
            net = tf.nn.relu(net, name="relu1")
            net = tf.layers.conv2d(net, num_filters, (3, 3), padding='valid', name="conv2")
            net = tf.nn.relu(net, name="relu2")

        conv.append(net)
        net = tf.layers.max_pooling2d(net, (2, 2), strides=(2, 2), name="pool")

        num_filters *= 2

    num_filters = int(num_filters / 2)
    net = conv.pop()

    for layer_i in range(num_layers - 1):
        num_filters = int(num_filters / 2)

        if dropout_keep is not None:
            net = tf.nn.dropout(net, dropout_keep)

        net = tf.layers.conv2d_transpose(net, num_filters, strides=(2, 2), kernel_size=(2, 2),
                                         name="up_conv_{}".format(layer_i))

        traverse = conv.pop()
        with tf.variable_scope("crop_{}".format(num_layers - layer_i)):
            traverse = tf.image.resize_image_with_crop_or_pad(traverse, int(net.shape[1]), int(net.shape[2]))
        net = tf.concat([traverse, net], axis=3, name="concat")

        with tf.variable_scope("conv_{}".format(num_layers + layer_i)):
            net = tf.layers.conv2d(net, num_filters, (3, 3), padding='valid', name="conv1")
            net = tf.nn.relu(net, name="relu1")
            net = tf.layers.conv2d(net, num_filters, (3, 3), padding='valid', name="conv2")
            net = tf.nn.relu(net, name="relu2")

    assert len(conv) == 0

    net = tf.layers.conv2d(net, 2, (1, 1), padding='same', name="weight_output")

    return net


def input_size_needed(output_size, num_layers):
    for i in range(num_layers - 1):
        assert output_size % 2 == 0, 'expand layer {} has size {} not divisible by 2' \
            .format(num_layers - i, output_size)
        output_size = (output_size + 4) / 2

    for i in range(num_layers - 1):
        output_size = (output_size + 4) * 2

    return int(output_size + 4)
