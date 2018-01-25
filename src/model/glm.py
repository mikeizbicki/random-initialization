def modify_parser(subparsers):
    import argparse
    from interval import interval

    subparser_model = subparsers.add_parser('glm')

    parser_weights = subparser_model.add_argument_group('weight initialization')
    parser_weights.add_argument('--scale',type=interval(float),default=1.0)
    parser_weights.add_argument('--mean',type=interval(float),default=0.0)
    parser_weights.add_argument('--randomness',choices=['normal','uniform','laplace'],default='normal')
    parser_weights.add_argument('--abs',type=bool,default=False)
    parser_weights.add_argument('--normalize',type=str,default='False')

    parser_weights.add_argument('--loss',choices=['mse','xentropy','huber','absdiff'],default='xentropy')
    parser_weights.add_argument('--l2',type=interval(float),default=1e-6)
    parser_weights.add_argument('--l1',type=interval(float),default=0.0)

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

    with tf.name_scope('model'):
        w = tf.Variable(randomness(data.dimX+[data.dimY],0),name='w')
        b = tf.Variable(1.0,name='b')
        waxes=range(0,len(data.dimX))
        y = tf.tensordot(x_,w,axes=(map(lambda x: x+1,waxes),waxes))+b

    return y

def loss(args,y_,y):
    import tensorflow as tf

    with tf.name_scope('losses'):
        xentropy = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y),
                    #tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y),
                    name='xentropy')
        tf.add_to_collection(tf.GraphKeys.LOSSES,xentropy)

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
