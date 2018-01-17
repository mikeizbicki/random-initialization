def modify_parser(subparsers):
    import argparse
    from interval import interval

    parser = subparsers.add_parser('mnist', help='a handwritten digit dataset')
    parser.add_argument('--data_dir',type=str,default='data/mnist')
    parser.add_argument('--numdp',type=interval(int),default=60000)
    parser.add_argument('--numdp_test',type=interval(int),default=10000)

    parser.add_argument('--seed',type=interval(int),default=0)
    parser.add_argument('--label_corruption',type=interval(float),default=0)
    parser.add_argument('--noise',type=interval(float),default=0)
    parser.add_argument('--loud',type=interval(int),default=0)
    parser.add_argument('--ones',type=interval(int),default=0)

def init(args):

    global train
    global test 
    global dimX
    global dimY

    import tensorflow as tf
    from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
    import numpy as np
    import random

    datasets = read_data_sets(args['data_dir'],False)

    class Dataset:
        def __init__(self):
            pass

    dimX = [28,28,1]
    dimY = 10
    numdp = args['numdp']

    # training data

    X = np.concatenate([datasets.train._images.reshape([55000]+dimX),
                        datasets.validation._images.reshape([5000]+dimX)])
    X = X[0:args['numdp'],...]
    Y = np.eye(10)[np.concatenate([datasets.train._labels,
                                   datasets.validation._labels])]
    Y = Y[0:args['numdp']]

    random.seed(args['seed'])
    np.random.seed(args['seed'])
    noise_binary=np.random.binomial(1,args['noise'],size=X.shape)
    noise_exponential=np.random.exponential(size=X.shape)
    noise=noise_binary*noise_exponential
    X=np.maximum(noise,X)

    random.seed(args['seed'])
    np.random.seed(args['seed'])
    num_corrupted=int(args['label_corruption']*numdp)
    Y_shift=np.random.randint(1,9,size=[num_corrupted])
    Y_shifted=(np.argmax(Y[0:num_corrupted],axis=1)+Y_shift)%10
    Y_corrupted=np.eye(10)[Y_shifted]
    Y[0:num_corrupted] = Y_corrupted

    X[0:args['loud']] *= args['numdp']
    Y[0:args['loud']] = Y[0]

    X[0:args['ones']] = 1+0*X[0:args['ones']]
    train=tf.data.Dataset.from_tensor_slices((np.float32(X),np.float32(Y)))

    # testing data

    X = datasets.test._images.reshape([10000]+dimX)
    X = X[0:args['numdp_test'],...]
    Y = np.eye(10)[datasets.test._labels]
    Y = Y[0:args['numdp_test']]
    test=tf.data.Dataset.from_tensor_slices((np.float32(X),np.float32(Y)))

