from __future__ import print_function

def modify_parser(subparsers):
    import argparse
    from interval import interval

    subparser_model = subparsers.add_parser('mean')

    parser = subparser_model.add_argument_group('weight initialization')
    parser.add_argument('--loss',choices=['mse','huber','absdiff'],default='mse')
    parser.add_argument('--warm_start',choices=['zero','mu_hat','mu_med'],default='mu_hat')
    parser.add_argument('--l2',type=interval(float),default=0.0)
    #parser.add_argument('--l1',type=interval(float),default=0.0)

def inference(x_,data,opts,is_training=True):
    global mu_true
    mu_true=data.mu
    import tensorflow as tf
    import numpy as np

    if opts['warm_start']=='zero':
        w0=np.zeros([1,data.dimY])
    elif opts['warm_start']=='mu_med':
        w0=data.mu_med
    elif opts['warm_start']=='mu_hat':
        w0=data.mu_hat
    
    w = tf.Variable(np.float32(w0),name='w')
    return w

def loss(args,y_,y):
    import tensorflow as tf

    with tf.name_scope('losses'):
        diff_y = tf.abs(y_-y)
        diff_mu = tf.abs(mu-y)

        mse_y = tf.reduce_sum(diff_y**2)
        mse_mu = tf.reduce_sum(diff_mu**2)
        #tf.add_to_collection(tf.GraphKeys.LOSSES,mse_y)
        #tf.add_to_collection(tf.GraphKeys.LOSSES,mse_mu)

        huber_y = tf.reduce_sum(tf.where(diff_y<1,diff_y**2,diff_y))
        huber_mu = tf.reduce_sum(tf.where(diff_mu<1,diff_mu**2,diff_mu))
        #tf.add_to_collection(tf.GraphKeys.LOSSES,huber_y)
        #tf.add_to_collection(tf.GraphKeys.LOSSES,huber_mu)

        absdiff_y = tf.reduce_sum(diff_y)
        absdiff_mu = tf.reduce_sum(diff_mu)
        #tf.add_to_collection(tf.GraphKeys.LOSSES,absdiff_y)
        #tf.add_to_collection(tf.GraphKeys.LOSSES,absdiff_mu)

        regularization=0
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            regularization+=tf.nn.l2_loss(var)*args['l2']

        loss = eval(args['loss']+'_y')+regularization
        tf.add_to_collection(tf.GraphKeys.LOSSES,loss)
        tf.add_to_collection(tf.GraphKeys.LOSSES,mse_mu)

    loss_per_dp=loss #FIXME: this won't work for batch sizes larger than 1

    return loss,loss_per_dp

