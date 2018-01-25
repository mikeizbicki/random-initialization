def modify_parser(subparsers):
    import argparse
    from interval import interval

    parser = subparsers.add_parser('mnist', help='a handwritten digit dataset')
    parser.add_argument('--data_dir',type=str,default='data/mnist')
    parser.add_argument('--numdp',type=interval(int),default=60000)
    parser.add_argument('--numdp_balanced',action='store_true')
    parser.add_argument('--numdp_test',type=interval(int),default=10000)
    parser.add_argument('--seed',type=interval(int),default=0)

def init(args):

    global train
    global train_numdp
    global train_X
    global train_Y
    global test
    global test_numdp
    global test_X
    global test_Y
    global dimX 
    global dimY 

    import tensorflow as tf
    from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
    import numpy as np
    import random

    random.seed(args['seed'])
    np.random.seed(args['seed'])
    datasets = read_data_sets(args['data_dir'],False)

    class Dataset:
        def __init__(self):
            pass

    dimX = [28,28,1]
    dimY = 10
    numdp = args['numdp']

    # training data

    train_X = np.concatenate([datasets.train._images.reshape([55000]+dimX),
                        datasets.validation._images.reshape([5000]+dimX)])
    train_Y = np.eye(10)[np.concatenate([datasets.train._labels,
                                   datasets.validation._labels])]

    random.seed(args['seed'])
    np.random.seed(args['seed'])
    np.random.shuffle(train_X)

    random.seed(args['seed'])
    np.random.seed(args['seed'])
    np.random.shuffle(train_Y)

    if args['numdp_balanced']:
        Yargmax = np.argmax(train_Y,axis=1)
        numdp_per_class=args['numdp']/10
        allIndices=[]
        for i in range(0,10):
            arr,=np.where(Yargmax==i)
            allIndices += np.ndarray.tolist(arr[:numdp_per_class])
        train_X = train_X[allIndices]
        train_Y = train_Y[allIndices]

        random.seed(args['seed'])
        np.random.seed(args['seed'])
        np.random.shuffle(train_X)

        random.seed(args['seed'])
        np.random.seed(args['seed'])
        np.random.shuffle(train_Y)

    else:
        train_X = train_X[0:args['numdp'],...]
        train_Y = train_Y[0:args['numdp']]

    Id = np.array(range(0,args['numdp']))
    train=tf.data.Dataset.from_tensor_slices((np.float32(train_X),np.float32(train_Y),Id))

    # testing data

    test_X = datasets.test._images.reshape([10000]+dimX)
    test_X = test_X[0:args['numdp_test'],...]
    test_Y = np.eye(10)[datasets.test._labels]
    test_Y = test_Y[0:args['numdp_test']]
    Id = np.array(range(0,args['numdp_test']))
    test=tf.data.Dataset.from_tensor_slices((np.float32(test_X),np.float32(test_Y),Id))

