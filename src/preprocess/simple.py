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

        def add_var(x,y,id):
            return (x,y,id,0)

        def unit_norm(x,y,id,mod):
            if partitionargs['preprocess']['unit_norm']:
                return (x/tf.norm(x),y,id,mod)
            else:
                return (x,y,id,mod)

        gaussian_X_norms=tf.random_normal([partitionargs['preprocess']['gaussian_X']]+data.dimX)
        def gaussian_X(x,y,id,mod):
            (mod2,x2)=tf.cond(
                    partitionargs['preprocess']['gaussian_X']>id,
                    lambda:(1,gaussian_X_norms[id,...]*tf.norm(x)),
                    lambda:(mod,x)
                    )
            return (x2,y,id,mod2)

        def zero_Y(x,y,id,mod):
            (mod2,y2)=tf.cond(
                    partitionargs['preprocess']['zero_Y']>id,
                    lambda:(1,0*y),
                    lambda:(mod,y)
                    )
            return (x,y2,id,mod2)

        def max_Y(x,y,id,mod):
            try:
                (mod2,y2)=tf.cond(
                        partitionargs['preprocess']['max_Y']>id,
                        lambda:(1,data.train_Y_max),
                        lambda:(mod,y)
                        )
            except AttributeError:
                y2=y
                mod2=mod
            return (x,y2,id,mod2)

        def label_corruption(x,y,id,mod):
            if partitionargs['preprocess']['label_corruption']>0:
                (mod2,y2)=tf.cond(
                        id>=partitionargs['preprocess']['label_corruption'],
                        lambda: (mod,y),
                        lambda: (  1,tf.concat([y[1:],y[:1]],axis=0))
                        )
            else:
                y2=y
                mod2=mod
            return (x,y2,id,mod2)

        def label_unshifted_percentile(x,y,id,mod):
            if partitionargs['preprocess']['label_unshifted_percentile']<100.0:
                (mod2,y2)=tf.cond(
                        id%100<=int(partitionargs['preprocess']['label_unshifted_percentile']),
                        lambda: (mod,y),
                        lambda: (  1,tf.concat([y[1:],y[:1]],axis=0))
                        )
            else:
                y2=y
                mod2=mod
            return (x,y2,id,mod2)

        def pad_dim(x,y,id,mod):
            if partitionargs['preprocess']['pad_dim']>0:
                x2=tf.pad(
                        x,
                        [[0,partitionargs['preprocess']['pad_dim']],[0,partitionargs['preprocess']['pad_dim']],[0,0]],
                        constant_values=0.5,
                        )
                return (x2,y,id,mod)
            else:
                return (x,y,id,mod)

        def triangle(x,y,id,mod):
            if partitionargs['preprocess']['triangle']>0:
                dimXnew=[data.dimX[0]+partitionargs['preprocess']['pad_dim'],data.dimX[1]+partitionargs['preprocess']['pad_dim']]
                triangle=np.float32(1.414*np.tril(np.ones(dimXnew, dtype=int), -1))
                triangle=triangle.reshape(dimXnew+[data.dimX[2]])
                (mod2,x2)=tf.cond(
                        id<partitionargs['preprocess']['triangle'],
                        lambda:(1,triangle),
                        lambda:(mod,x)
                        )
                return (x2,y,id,mod2)
            else:
                return (x,y,id,mod)

        with tf.device('/cpu:0'):
            p=partitionargs['preprocess']['parallel']
            data.train = data.train.map(num_parallel_calls=p,map_func=add_var)
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

            data.valid = data.valid.map(num_parallel_calls=p,map_func=add_var)
            data.valid = data.valid.map(num_parallel_calls=p,map_func=unit_norm)
            data.valid = data.valid.map(num_parallel_calls=p,map_func=pad_dim)
            #data.valid = data.valid.batch(partitionargs['preprocess']['batch_size_test'])
            data.valid = data.valid.prefetch(partitionargs['preprocess']['prefetch'])

            data.test = data.test.map(num_parallel_calls=p,map_func=add_var)
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
