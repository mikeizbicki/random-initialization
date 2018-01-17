def modify_parser(subparsers):
    import argparse
    from interval import interval

    subparser_model = subparsers.add_parser('custom')
    parser_network = subparser_model.add_argument_group('network options')
    parser_network.add_argument('--layers',type=interval(int),nargs='*',default=[interval(int)(100)])
    parser_network.add_argument('--activation',choices=['none','relu','relu6','crelu','elu','softplus','softsign','sigmoid','tanh','logistic'],default='relu')

    parser_weights = subparser_model.add_argument_group('weight initialization')
    parser_weights.add_argument('--scale',type=interval(float),default=1.0)
    parser_weights.add_argument('--mean',type=interval(float),default=0.0)
    parser_weights.add_argument('--randomness',choices=['normal','uniform','laplace'],default='normal')
    parser_weights.add_argument('--abs',type=bool,default=False)
    parser_weights.add_argument('--normalize',type=str,default='False')


def inference(x_,data,opts,is_training=True):
    import tensorflow as tf
    import numpy as np

    def randomness(size,seed):
        from stable_random import stable_random
        r=stable_random(size,opts['seed_np']+seed,dist=opts['randomness']).astype(np.float32)
        if opts['normalize']=='True':
            r/=np.amax(np.abs(r))
        if opts['abs']:
            r=np.abs(r)
        return opts['mean']+opts['scale']*r

    if opts['activation']=='none':
        activation=tf.identity
    else:
        activation=eval('tf.nn.'+opts['activation'])

    with tf.name_scope('model'):
        bias=1.0
        layer=0
        y = x_
        n0 = data.dimX
        for n in opts['layers']:
            print('    layer'+str(layer)+' nodes: '+str(n))
            with tf.name_scope('layer'+str(layer)):
                w = tf.Variable(randomness(n0+[n],layer),name='w')
                b = tf.Variable(randomness([1,n],layer+1),name='b')
                y = activation(tf.matmul(y,w)+b)
            n0 = [n]
            if opts['activation']=='crelu':
                n0*=2
            layer+=1

        with tf.name_scope('layer_final'):
            w = tf.Variable(randomness(n0+[data.dimY],layer+1),name='w')
            b = tf.Variable(bias,name='b')
            y = tf.matmul(y,w)+b

    return y
