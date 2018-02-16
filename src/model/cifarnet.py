def modify_parser(subparsers):
    import argparse
    from interval import interval

    subparser = subparsers.add_parser('cifarnet')

    subparser.add_argument('--loss',choices=['mse','xentropy'],default='xentropy')
    subparser.add_argument('--dropout',type=interval(float),default=0.5)
    subparser.add_argument('--l2',type=interval(float),default=1e-6)
    subparser.add_argument('--l1',type=interval(float),default=0.0)


def inference(x_,data,opts,is_training):

    import tensorflow as tf
    slim=tf.contrib.slim

    num_classes=data.dimY
    dropout_keep_prob=opts['dropout']

    init = tf.contrib.layers.xavier_initializer(
            uniform=True,
            seed=0,
            dtype=tf.float32
            )
    
    init_trunc_normal = tf.truncated_normal_initializer(
            stddev=1/192.0,
            seed=0,
            dtype=tf.float32
            )

    with tf.variable_scope('model', [x_, num_classes]):
        net = slim.conv2d(x_, 64, [5, 5], scope='conv1', weights_initializer=init)
        net = slim.max_pool2d(net, [2, 2], 2, scope='pool1')
        net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm1')
        net = slim.conv2d(net, 64, [5, 5], scope='conv2', weights_initializer=init)
        net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm2')
        net = slim.max_pool2d(net, [2, 2], 2, scope='pool2')
        net = slim.flatten(net)
        net = slim.fully_connected(net, 384, scope='fc3', weights_initializer=init)
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout3')
        net = slim.fully_connected(net, 192, scope='fc4', weights_initializer=init)
        logits = slim.fully_connected(net, num_classes,
                                      biases_initializer=tf.zeros_initializer(),
                                      weights_initializer=init_trunc_normal,
                                      weights_regularizer=None,
                                      activation_fn=None,
                                      scope='logits')
   
    return logits

import model._losses
loss = model._losses.classification

