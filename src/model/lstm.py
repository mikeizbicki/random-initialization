def modify_parser(subparsers):
    import argparse
    from interval import interval

    subparser_model = subparsers.add_parser('lstm',help='lstm for stream data')

    parser_weights = subparser_model.add_argument_group('weight initialization')

    parser_weights.add_argument('--loss',choices=['mse','xentropy','xentropy_tanh','huber','absdiff'],default='xentropy')

    parser_weights.add_argument('--dropout',type=interval(float),default=0.8)
    parser_weights.add_argument('--l2',type=interval(float),default=1e-6)
    parser_weights.add_argument('--l1',type=interval(float),default=0.0)

def inference(x_,data,opts,is_training):
    import tensorflow as tf
    import tflearn
    import numpy as np

    print('x_=',x_.get_shape())

    dropout_rate=1.0
    #dropout_rate=tf.cond(
            #is_training,
            #lambda:opts['dropout'],
            #lambda:1.0
            #)

    net = tflearn.embedding(x_, input_dim=10000, output_dim=128)
    net = tflearn.lstm(net, 128, dropout=dropout_rate)
    net = tflearn.fully_connected(net, data.dimY, activation='linear')
    return net

def loss(args,y_,y):
    import tensorflow as tf

    with tf.name_scope('losses'):
        xentropy = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y),
                    #tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y),
                    name='xentropy')
        tf.add_to_collection(tf.GraphKeys.LOSSES,xentropy)

        xentropy_tanh = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=tf.tanh(y)),
                    #tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y),
                    name='xentropy')
        tf.add_to_collection(tf.GraphKeys.LOSSES,xentropy_tanh)

        mse = tf.losses.mean_squared_error(y_,y)
        huber = tf.losses.huber_loss(y_,y)
        absdiff = tf.losses.absolute_difference(y_,y)

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

