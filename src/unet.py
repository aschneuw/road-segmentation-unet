import tensorflow as tf

"""
U-Net implementation according to https://arxiv.org/pdf/1505.04597.pdf

Generic implementation of U-Net with parametrised number of layers and filters.
Added dilated (or atrous) convolution for wider receptive field.
Used for binary image segmentation.
"""


def forward(X, num_layers, root_size, dilated_layers, dropout_keep=None):
    """Build the U-Net static computation graph

    :param X: input of network, use `input_size_needed` to find which image size it expect
    :param num_layers: number of layers down
    :param root_size: number of filters of the first layer, will be doubled every down move
    :param dilated_layers: add dilated convolution to transverse crop module
    :param dropout_keep: tf variable keep_proba to enable dropout or not
    :return: the network
    """
    net = X - 0.5
    net = tf.layers.conv2d(net, 3, (1, 1), name="color_space_adjust")

    num_filters = root_size
    conv = []

    for layer_i in range(num_layers):
        if dropout_keep is not None:
            net = tf.nn.dropout(net, dropout_keep)

        if dilated_layers:
            with tf.variable_scope("conv_dilut_{}".format(layer_i)):
                dilated = tf.layers.conv2d(net, num_filters, (3, 3), dilation_rate=(2, 2), padding='valid',
                                           name="atrous_conv1")
                dilated = tf.nn.relu(dilated, name="relu1")
                dilated = tf.layers.conv2d(dilated, num_filters, (3, 3), dilation_rate=(2, 2), padding='valid',
                                           name="atrous_conv2")
                dilated = tf.nn.relu(dilated, name="relu2")

        with tf.variable_scope("conv_{}".format(layer_i)):
            net = tf.layers.conv2d(net, num_filters, (3, 3), padding='valid', name="conv1")
            net = tf.nn.relu(net, name="relu1")
            net = tf.layers.conv2d(net, num_filters, (3, 3), padding='valid', name="conv2")
            net = tf.nn.relu(net, name="relu2")

        if dilated_layers:
            conv.append((net, dilated))
        else:
            conv.append(net)

        net = tf.layers.max_pooling2d(net, (2, 2), strides=(2, 2), name="pool")

        num_filters *= 2

    num_filters = int(num_filters / 2)
    net = conv.pop()
    if dilated_layers:
        net = net[0]

    for layer_i in range(num_layers - 1):
        num_filters = int(num_filters / 2)

        if dropout_keep is not None:
            net = tf.nn.dropout(net, dropout_keep)

        net = tf.layers.conv2d_transpose(net, num_filters, strides=(2, 2), kernel_size=(2, 2),
                                         name="up_conv_{}".format(layer_i))

        if dilated_layers:
            traverse, traverse_dilat = conv.pop()
            with tf.variable_scope("crop_{}".format(layer_i)):
                traverse_crop = tf.image.resize_image_with_crop_or_pad(traverse, int(net.shape[1]), int(net.shape[2]))

            with tf.variable_scope("crop_dilat_{}".format(layer_i)):
                traverse_dilat_crop = tf.image.resize_image_with_crop_or_pad(traverse_dilat, int(net.shape[1]),
                                                                             int(net.shape[2]))

            net = tf.concat([traverse_crop, traverse_dilat_crop, net], axis=3, name="concat")
        else:
            traverse = conv.pop()
            with tf.variable_scope("crop_{}".format(layer_i)):
                traverse_crop = tf.image.resize_image_with_crop_or_pad(traverse, int(net.shape[1]), int(net.shape[2]))

            net = tf.concat([traverse_crop, net], axis=3, name="concat")

        with tf.variable_scope("conv_{}".format(num_layers + layer_i)):
            net = tf.layers.conv2d(net, num_filters, (3, 3), padding='valid', name="conv1")
            net = tf.nn.relu(net, name="relu1")
            net = tf.layers.conv2d(net, num_filters, (3, 3), padding='valid', name="conv2")
            net = tf.nn.relu(net, name="relu2")

    assert len(conv) == 0

    net = tf.layers.conv2d(net, 2, (1, 1), padding='same', name="weight_output")

    return net


def input_size_needed(output_size, num_layers):
    """Utility function to compute image size for a given U-Net output

    The U-Net crops some border of the image during prediction, it can be tedious to check if an input size is valid
    output_size: width/height of the output image
    num_layer: number of layers
    """
    for i in range(num_layers - 1):
        assert output_size % 2 == 0, 'expand layer {} has size {} not divisible by 2' \
            .format(num_layers - i, output_size)
        output_size = (output_size + 4) / 2

    for i in range(num_layers - 1):
        output_size = (output_size + 4) * 2

    return int(output_size + 4)
