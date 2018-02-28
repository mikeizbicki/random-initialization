def modify_parser(subparsers):
    import argparse
    from interval import interval

    subparser = subparsers.add_parser('simple')

    subparser_preprocess = subparser.add_argument_group('data preprocessing options')
    subparser_preprocess.add_argument('--unit_norm',action='store_true')
    subparser_preprocess.add_argument('--triangle',type=interval(int),default=0)
    subparser_preprocess.add_argument('--label_corruption',type=interval(int),default=0)
    subparser_preprocess.add_argument('--label_unshifted_percentile',type=interval(float),default=100.0)
    subparser_preprocess.add_argument('--gaussian_X',type=interval(int),default=0)
    subparser_preprocess.add_argument('--zero_Y',type=interval(int),default=0)
    subparser_preprocess.add_argument('--max_Y',type=interval(int),default=0)
    subparser_preprocess.add_argument('--pad_dim',type=interval(int),default=0)
    subparser_preprocess.add_argument('--pad_dim_numbad',type=interval(int),default=0)

    subparser_preprocess.add_argument('--parallel',type=int,default=10)
    subparser_preprocess.add_argument('--prefetch',type=int,default=1000)

def preprocess_data(data,partitionargs):
    import tensorflow as tf
    import numpy as np

    with tf.name_scope('inputs'):
        data.init(partitionargs['data'])

        def unit_norm(x,y,id):
            if partitionargs['preprocess']['unit_norm']:
                return (x/tf.norm(x),y,id)
            else:
                return (x,y,id)

        gaussian_X_norms=tf.random_normal([partitionargs['preprocess']['gaussian_X']]+data.dimX)
        def gaussian_X(x,y,id):
            x2=tf.cond(
                    partitionargs['preprocess']['gaussian_X']>id,
                    lambda:gaussian_X_norms[id,...]*tf.norm(x),
                    lambda:x
                    )
            return (x2,y,id)

        def zero_Y(x,y,id):
            y2=tf.cond(
                    partitionargs['preprocess']['zero_Y']>id,
                    lambda:0*y,
                    lambda:y
                    )
            return (x,y2,id)

        def max_Y(x,y,id):
            try:
                y2=tf.cond(
                        partitionargs['preprocess']['max_Y']>id,
                        lambda:data.train_Y_max,
                        lambda:y
                        )
            except AttributeError:
                y2=y
            return (x,y2,id)

        def label_corruption(x,y,id):
            if partitionargs['preprocess']['label_corruption']>0:
                y2=tf.cond(
                        id>=partitionargs['preprocess']['label_corruption'],
                        lambda: y,
                        lambda: tf.concat([y[1:],y[:1]],axis=0)
                        )
            else:
                y2=y
            return (x,y2,id)

        def label_unshifted_percentile(x,y,id):
            if partitionargs['preprocess']['label_unshifted_percentile']<100.0:
                y2=tf.cond(
                        id%100<=int(partitionargs['preprocess']['label_unshifted_percentile']),
                        lambda: y,
                        lambda: tf.concat([y[1:],y[:1]],axis=0)
                        )
            else:
                y2=y
            return (x,y2,id)

        def pad_dim(x,y,id):
            if partitionargs['preprocess']['pad_dim']>0:
                x2=tf.pad(
                        x,
                        [[0,partitionargs['preprocess']['pad_dim']],[0,partitionargs['preprocess']['pad_dim']],[0,0]],
                        constant_values=0.5,
                        )
                return (x2,y,id)
            else:
                return (x,y,id)

        def triangle(x,y,id):
            if partitionargs['preprocess']['triangle']>0:
                dimXnew=[data.dimX[0]+partitionargs['preprocess']['pad_dim'],data.dimX[1]+partitionargs['preprocess']['pad_dim']]
                triangle=np.float32(1.414*np.tril(np.ones(dimXnew, dtype=int), -1))
                triangle=triangle.reshape(dimXnew+[data.dimX[2]])
                x2=tf.cond(
                        id<partitionargs['preprocess']['triangle'],
                        lambda:triangle,
                        lambda:x
                        )
                return (x2,y,id)
            else:
                return (x,y,id)

        with tf.device('/cpu:0'):
            p=partitionargs['preprocess']['parallel']
            data.train = data.train.map(num_parallel_calls=p,map_func=unit_norm)
            data.train = data.train.map(num_parallel_calls=p,map_func=label_corruption)
            data.train = data.train.map(num_parallel_calls=p,map_func=label_unshifted_percentile)
            data.train = data.train.map(num_parallel_calls=p,map_func=gaussian_X)
            data.train = data.train.map(num_parallel_calls=p,map_func=zero_Y)
            data.train = data.train.map(num_parallel_calls=p,map_func=max_Y)
            data.train = data.train.map(num_parallel_calls=p,map_func=pad_dim)
            data.train = data.train.map(num_parallel_calls=p,map_func=triangle)
            #data.train = data.train.shuffle(partitionargs['preprocess']['batch_size']*20,seed=0)
            #data.train = data.train.batch(partitionargs['preprocess']['batch_size'])
            data.train = data.train.prefetch(partitionargs['preprocess']['prefetch'])

            data.valid = data.valid.map(num_parallel_calls=p,map_func=unit_norm)
            data.valid = data.valid.map(num_parallel_calls=p,map_func=pad_dim)
            #data.valid = data.valid.batch(partitionargs['preprocess']['batch_size_test'])
            data.valid = data.valid.prefetch(partitionargs['preprocess']['prefetch'])

            data.test = data.test.map(num_parallel_calls=p,map_func=unit_norm)
            data.test = data.test.map(num_parallel_calls=p,map_func=pad_dim)
            #data.test = data.test.batch(partitionargs['preprocess']['batch_size_test'])
            data.test = data.test.prefetch(partitionargs['preprocess']['prefetch'])

        if partitionargs['preprocess']['pad_dim']>0:
            data.dimX=[
                    data.dimX[0]+partitionargs['preprocess']['pad_dim'],
                    data.dimX[1]+partitionargs['preprocess']['pad_dim'],
                    data.dimX[2]
                    ]

        return data.train,data.valid,data.test
