def modify_parser(subparsers):
    import argparse
    from interval import interval

    subparser_model = subparsers.add_parser('lenet')


def inference(x_,data,opts):

  import tensorflow as tf
  slim=tf.contrib.slim
  end_points = {}

  num_classes=data.dimY
  is_training=False
  dropout_keep_prob=0.5

  with tf.variable_scope('model', [x_, num_classes]):
    net = slim.conv2d(x_, 32, [5, 5], scope='conv1')
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool1')
    net = slim.conv2d(net, 64, [5, 5], scope='conv2')
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool2')
    net = slim.flatten(net)
    end_points['Flatten'] = net

    net = slim.fully_connected(net, 1024, scope='fc3')
    net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                       scope='dropout3')
    logits = slim.fully_connected(net, num_classes, activation_fn=None,
                                  scope='fc4')

  end_points['Logits'] = logits

  return logits
