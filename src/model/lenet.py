def modify_parser(subparsers):
    import argparse
    from interval import interval

    subparser = subparsers.add_parser('lenet')

    subparser.add_argument('--loss',choices=['mse','xentropy'],default='xentropy')
    subparser.add_argument('--l2',type=interval(float),default=1e-6)
    subparser.add_argument('--l1',type=interval(float),default=0.0)


def inference(x_,data,opts):

    import tensorflow as tf
    slim=tf.contrib.slim
    end_points = {}
   
    num_classes=data.dimY
    is_training=False
    dropout_keep_prob=1.0 #0.5

    init = tf.contrib.layers.xavier_initializer(
            uniform=True,
            seed=0,
            dtype=tf.float32
            )
   
    with tf.variable_scope('model', [x_, num_classes]):
        net = slim.conv2d(x_, 32, [5, 5], scope='conv1', weights_initializer=init)
        net = slim.max_pool2d(net, [2, 2], 2, scope='pool1')
        net = slim.conv2d(net, 64, [5, 5], scope='conv2', weights_initializer=init)
        net = slim.max_pool2d(net, [2, 2], 2, scope='pool2')
        net = slim.flatten(net)
        end_points['Flatten'] = net
     
        net = slim.fully_connected(net, 1024, scope='fc3', weights_initializer=init)
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout3')
        logits = slim.fully_connected(net, num_classes, activation_fn=None, weights_initializer=init,
                                    scope='fc4')

    #with tf.name_scope('model'):
#
        #def conv_and_maxpool(x,n):
            #w = tf.Variable(tf.random_normal([5, 5, 1, 32]))
            #b = tf.Variable(tf.random_normal([32]))
            #m = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='SAME')
            #y = tf.nn.bias_add(m1, b)
            #k = 2
            #return tf.nn.max_pool(y, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
#
        #y1 = conv_and_maxpool(x,32)
        #y2 = conv_and_maxpool(x,64)
#
        ## fully connected layer
        #wd1 = tf.Variable(tf.random_normal([7*7*64, 1024]))
        #bd1 = tf.Variable(tf.random_normal([1024]))
        #fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
        #fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        #fc1 = tf.nn.relu(fc1)
   
    end_points['Logits'] = logits
   
    return logits

def loss(args,y_,y):
    import tensorflow as tf

    with tf.name_scope('losses'):
        xentropy = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y),
                    #tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y),
                    name='xentropy')
        tf.add_to_collection(tf.GraphKeys.LOSSES,xentropy)

        mse = tf.losses.mean_squared_error(y_,y)

        argmax_y =tf.argmax(y ,axis=1)
        argmax_y_=tf.argmax(y_,axis=1)
        results = tf.cast(tf.equal(argmax_y,argmax_y_),tf.float32)
        #results = tf.Print(results,[argmax_y,argmax_y_,results])
        accuracy = tf.reduce_mean(results,name='accuracy')
        tf.add_to_collection(tf.GraphKeys.LOSSES,accuracy)

        regularization=0
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            regularization+=tf.nn.l2_loss(var)*args['l2']

        print('loss=',args['loss'])
        loss = eval(args['loss'])+regularization

    return loss
