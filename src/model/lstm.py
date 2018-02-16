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

import model._losses
loss = model._losses.classification

