def modify_parser(subparsers):
    import argparse
    from interval import interval

    parser = subparsers.add_parser('cifar10', help='a small image classification dataset')
    parser.add_argument('--data_dir',type=str,default='data/cifar10')
    parser.add_argument('--numdp',type=interval(int),default=50000)
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
    from tflearn.datasets import cifar10
    import numpy as np
    import random

    random.seed(args['seed'])
    np.random.seed(args['seed'])
    (train_X, train_Y), (test_X, test_Y) = cifar10.load_data(dirname=args['data_dir'])
    train_Y = np.eye(10)[train_Y]
    test_Y = np.eye(10)[test_Y]

    print('train_X=',train_X.shape)
    print('train_Y=',train_Y.shape)

    dimX = [32,32,3]
    dimY = 10
    numdp = args['numdp']
    train_numdp = numdp

    # training data

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

    test_X = test_X[0:args['numdp_test'],...]
    test_Y = test_Y[0:args['numdp_test']]
    #test_X = train_X
    #test_Y = train_Y
    Id = np.array(range(0,args['numdp_test']))
    test=tf.data.Dataset.from_tensor_slices((np.float32(test_X),np.float32(test_Y),Id))


