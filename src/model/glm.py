def modify_parser(subparsers):
    import argparse
    from interval import interval

    subparser_model = subparsers.add_parser('glm')

    parser_weights = subparser_model.add_argument_group('weight initialization')
    parser_weights.add_argument('--scale',type=interval(float),default=1.0)
    parser_weights.add_argument('--mean',type=interval(float),default=0.0)
    parser_weights.add_argument('--randomness',choices=['normal','uniform','laplace'],default='normal')
    parser_weights.add_argument('--abs',type=bool,default=False)
    parser_weights.add_argument('--normalize',action='store_true')
    parser_weights.add_argument('--zerow',action='store_true')

    parser_weights.add_argument('--loss',choices=['mse','xentropy','xentropy_robust','huber','absdiff'],default='xentropy')
    #parser_weights.add_argument('--eval',choices=['mse','xentropy','accuracy'],default='accuracy')

    parser_weights.add_argument('--l2',type=interval(float),default=1e-6)
    parser_weights.add_argument('--l1',type=interval(float),default=0.0)

def inference(x_,data,opts,is_training):
    import tensorflow as tf
    import numpy as np

    def randomness(size,seed):
        from stable_random import stable_random
        r=stable_random(size,opts['seed_np']+seed,dist=opts['randomness']).astype(np.float32)
        if opts['normalize']:
            r/=np.amax(np.abs(r))
        if opts['abs']:
            r=np.abs(r)
        return opts['mean']+opts['scale']*r

    with tf.name_scope('model'):
        if opts['zerow']:
            w0=np.float32(np.zeros(data.dimX+[data.dimY]))
        else:
            w0=randomness(data.dimX+[data.dimY],0)
        w = tf.Variable(w0,name='w')
        b = tf.Variable(1.0,name='b')
        waxes=range(0,len(data.dimX))
        print('x_=',x_)
        print('w=',w)
        print('b=',b)
        y = tf.tensordot(x_,w,axes=(map(lambda x: x+1,waxes),waxes))+b

    return y

import model._losses
loss = model._losses.classification

