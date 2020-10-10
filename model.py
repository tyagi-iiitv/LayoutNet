import tensorflow as tf
slim = tf.contrib.slim


def batch_norm_params(is_training):
    return {
        "decay": 0.9,
        "epsilon": 1e-5,
        "scale": True,
        "updates_collections": None,
        "is_training": is_training
    }


def lrelu(inputs, leak=0.2, scope="lrelu"):
    with tf.variable_scope(scope):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * inputs + f2 * abs(inputs)


def embed_arg_scope(is_training=True, outputs_collections=None):
    with slim.arg_scope([slim.fully_connected],
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params(is_training),
                        outputs_collections=outputs_collections)as arg_scp:
        return arg_scp


def embeddingSemvec(inputs, is_training, reuse=None, scope=None):
    with tf.variable_scope(scope or "embeddingSemvec", values=[inputs], reuse=reuse) as scp:
        end_pts_collection = scp.name + "end_pts"
        with slim.arg_scope(embed_arg_scope(is_training, end_pts_collection)):
            category = tf.concat([inputs[:, 0:6], inputs[:, 0:6], inputs[:, 0:6], inputs[:, 0:6], inputs[:, 0:6], inputs[:, 0:6], inputs[:, 0:6], inputs[:, 0:6], inputs[:, 0:6], inputs[:, 0:6]],1)
            textratio = tf.concat([inputs[:, 6:13], inputs[:, 6:13], inputs[:, 6:13], inputs[:, 6:13], inputs[:, 6:13], inputs[:, 6:13], inputs[:, 6:13], inputs[:, 6:13], inputs[:, 6:13], inputs[:, 6:13]],1)
            imgratio = tf.concat([inputs[:, 13:23], inputs[:, 13:23], inputs[:, 13:23], inputs[:, 13:23], inputs[:, 13:23], inputs[:, 13:23], inputs[:, 13:23], inputs[:, 13:23], inputs[:, 13:23], inputs[:, 13:23]], 1)
            net1 = slim.fully_connected(category, 48,
                                        normalizer_fn=None,
                                        normalizer_params=None,
                                        scope="category1")
            net2 = slim.fully_connected(textratio, 48,
                                        normalizer_fn=None,
                                        normalizer_params=None,
                                        scope="textRatio1")
            net3 = slim.fully_connected(imgratio, 48,
                                        normalizer_fn=None,
                                        normalizer_params=None,
                                        scope="imgratio1")
            net = tf.concat([net1, net2, net3], 1)
            net = slim.fully_connected(net, 32,
                                       normalizer_fn=None,
                                       normalizer_params=None,
                                       scope="semvec")
            end_pts = slim.utils.convert_collection_to_dict(end_pts_collection)
    return net, end_pts


def embedImg_arg_scope(is_training=True, outputs_collections=None):
    with slim.arg_scope([slim.fully_connected],
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params(is_training),
                        outputs_collections=outputs_collections)as arg_scp:
        return arg_scp


def embeddingImg(inputs, is_training, reuse=None, scope=None):
    with tf.variable_scope(scope or "embeddingImg", values=[inputs], reuse=reuse) as scp:
        end_pts_collection = scp.name + "end_pts"
        with slim.arg_scope(embedImg_arg_scope(is_training, end_pts_collection)):
            inputs = tf.reduce_mean(inputs, [1, 2])
            net = slim.fully_connected(inputs, 512,
                                       normalizer_fn=None,
                                       normalizer_params=None,
                                       scope="fc1")
            net = slim.fully_connected(net, 256,
                                       normalizer_fn=None,
                                       normalizer_params=None,
                                       scope="fc2")
            net = slim.fully_connected(net, 128,
                                       normalizer_fn=None,
                                       normalizer_params=None,
                                       scope="fc3")
            end_pts = slim.utils.convert_collection_to_dict(end_pts_collection)
    return net, end_pts


def embedTex_arg_scope(is_training=True, outputs_collections=None):
    with slim.arg_scope([slim.fully_connected],
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params(is_training),
                        outputs_collections=outputs_collections)as arg_scp:
        return arg_scp


def embeddingTex(inputs, is_training, reuse=None, scope=None):
    with tf.variable_scope(scope or "embeddingTex", values=[inputs], reuse=reuse) as scp:
        end_pts_collection = scp.name + "end_pts"
        with slim.arg_scope(embedTex_arg_scope(is_training, end_pts_collection)):
            net = slim.fully_connected(inputs, 256,
                                       normalizer_fn=None,
                                       normalizer_params=None,
                                       scope="fc1")
            net = slim.fully_connected(net, 256,
                                       normalizer_fn=None,
                                       normalizer_params=None,
                                       scope="fc2")
            net = slim.fully_connected(net, 128,
                                       normalizer_fn=None,
                                       normalizer_params=None,
                                       scope="fc3")
            end_pts = slim.utils.convert_collection_to_dict(end_pts_collection)
    return net, end_pts


def embedFusion_arg_scope(is_training=True, outputs_collections=None):
    with slim.arg_scope([slim.fully_connected],
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params(is_training),
                        outputs_collections=outputs_collections)as arg_scp:
        return arg_scp


def embeddingFusion(input1, input2, input3, is_training, reuse=None, scope=None):
    with tf.variable_scope(scope or "embeddingFusion", values=[input1, input2], reuse=reuse) as scp:
        end_pts_collection = scp.name + "end_pts"
        with slim.arg_scope(embedFusion_arg_scope(is_training, end_pts_collection)):
            net = tf.concat([input1, input2, input3], 1)
            net = slim.fully_connected(net, 256,
                                       normalizer_fn=None,
                                       normalizer_params=None,
                                       scope="fc1")
            net = slim.fully_connected(net, 128,
                                       normalizer_fn=None,
                                       normalizer_params=None,
                                       scope="fc2")
            end_pts = slim.utils.convert_collection_to_dict(end_pts_collection)
    return net, end_pts


def gen_arg_scope(is_training=True, outputs_collections=None):
    with slim.arg_scope([slim.conv2d_transpose, slim.fully_connected],
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params(is_training),
                        outputs_collections=outputs_collections):
        with slim.arg_scope([slim.conv2d_transpose],
                            kernel_size=[5, 5], stride=2,
                            padding="SAME") as arg_scp:
            return arg_scp


def generator(z, is_training, y=None, reuse=None, scope=None):
    if y is None:
        inputs = z
    else:
        inputs = tf.concat((z, y), 1)

    with tf.variable_scope(scope or "generator", values=[z], reuse=reuse) as scp:
        end_pts_collection = scp.name + "end_pts"
        with slim.arg_scope(gen_arg_scope(is_training, end_pts_collection)):
            net = slim.fully_connected(inputs, 4 * 4 * 512,
                                       normalizer_fn=None,
                                       normalizer_params=None,
                                       scope="projection")
            net = tf.reshape(net, [-1, 4, 4, 512])
            net = slim.batch_norm(net, scope="batch_norm",
                                  **batch_norm_params(is_training))
            net = slim.conv2d_transpose(net, 256, scope="conv_tp0")
            net = slim.conv2d_transpose(net, 128, scope="conv_tp1")
            net = slim.conv2d_transpose(net, 64, scope="conv_tp2")
            net = slim.conv2d_transpose(net, 3,
                                        activation_fn=tf.nn.tanh,
                                        normalizer_fn=None,
                                        normalizer_params=None,
                                        scope="conv_tp3")
            end_pts = slim.utils.convert_collection_to_dict(end_pts_collection)

    return net, end_pts


def disc_arg_scope(is_training=True, outputs_collections=None):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                        activation_fn=lrelu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params(is_training),
                        outputs_collections=outputs_collections):
        with slim.arg_scope([slim.conv2d],
                            kernel_size=[5, 5], stride=2,
                            padding="SAME") as arg_scp:
            return arg_scp


def discriminator(inputs, is_training, y=None, z=None, reuse=None, scope=None):
    if y is None:
        inputss = inputs
    else:
        inputss = tf.concat([inputs, y], 3)

    with tf.variable_scope(scope or "discriminator", values=[inputs], reuse=reuse) as scp:
        end_pts_collection = scp.name + "end_pts"
        with slim.arg_scope(disc_arg_scope(is_training, end_pts_collection)):
            net = slim.conv2d(inputss, 64,
                              normalizer_fn=None,
                              normalizer_params=None,
                              scope="conv0")
            net = slim.conv2d(net, 128, scope="conv1")
            net = slim.conv2d(net, 256, scope="conv2")
            net = slim.conv2d(net, 512, scope="conv3")
            net = slim.conv2d(net, 128,
                              activation_fn=None,
                              kernel_size=[4, 4], stride=1, padding="VALID",
                              normalizer_fn=None,
                              normalizer_params=None,
                              scope="conv4")
            if z is None:
                net = tf.squeeze(net, [1, 2], name="squeeze")
            else:
                net = tf.squeeze(net, [1, 2], name="squeeze")
                net = tf.concat((net, z), 1)
                net = slim.fully_connected(net, 1,
                                           normalizer_fn=None,
                                           normalizer_params=None,
                                           scope="layout")
            end_pts = slim.utils.convert_collection_to_dict(end_pts_collection)

    return net, end_pts


def enc_arg_scope(is_training=True, outputs_collections=None):
    with slim.arg_scope([slim.conv2d],
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                        activation_fn=lrelu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params(is_training),
                        kernel_size=[5, 5], stride=2, padding="SAME",
                        outputs_collections=outputs_collections) as arg_scp:
        return arg_scp


def encoder(inputs, is_training, y=None, reuse=None, scope=None):
    inputss = inputs

    with tf.variable_scope(scope or "encoder", values=[inputs], reuse=reuse) as scp:
        end_pts_collection = scp.name + "end_pts"
        with slim.arg_scope(enc_arg_scope(is_training, end_pts_collection)):
            net = slim.conv2d(inputss, 64,
                              normalizer_fn=None,
                              normalizer_params=None,
                              scope="conv0")
            net = slim.conv2d(net, 128, scope="conv1")
            net = slim.conv2d(net, 256, scope="conv2")
            net = slim.conv2d(net, 512, scope="conv3")

            if y is None:
                net = net
            else:
                net = tf.concat([net, y], 3)

            net1 = slim.conv2d(net, 128,
                               activation_fn=None,
                               kernel_size=[4, 4], stride=1, padding="VALID",
                               normalizer_fn=None,
                               normalizer_params=None,
                               scope="conv4_1")
            net2 = slim.conv2d(net, 128,
                               activation_fn=None,
                               kernel_size=[4, 4], stride=1, padding="VALID",
                               normalizer_fn=None,
                               normalizer_params=None,
                               scope="conv4_2")
            net1 = tf.squeeze(net1, [1, 2], name="squeeze")
            net2 = tf.squeeze(net2, [1, 2], name="squeeze")
            end_pts = slim.utils.convert_collection_to_dict(end_pts_collection)

    return net1, net2, end_pts
