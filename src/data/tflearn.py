def modify_parser(subparsers):
    import argparse
    from interval import interval

    parser = subparsers.add_parser('tflearn-image', help='toy deep learning image datasets from tflearn library')
    parser.add_argument('--name',choices=['cifar10','cifar100','mnist','oxflower17','svhn','titanic'])
    parser.add_argument('--data_dir',type=str,default='data/tflearn')
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
    import tflearn
    import numpy as np
    import random

    random.seed(args['seed'])
    np.random.seed(args['seed'])

    
    if args['name']=='cifar10':
        (train_X, train_Y), (test_X, test_Y) = tflearn.datasets.cifar10.load_data(args['data_dir'])
    elif args['name']=='mnist':
        train_X, train_Y, test_X, test_Y = tflearn.datasets.mnist.load_data(args['data_dir'])
    else:
        raise ValueError(args['name'] + ' not yet implemented')

    #load_data=tflearn.datasets.cifar10.load_data
    #(train_X, train_Y), (test_X, test_Y) = load_data(dirname=args['data_dir'])

    print('train_X=',train_X.shape)
    print('train_Y=',train_Y.shape)

    dimX=list(train_X.shape)[1:]
    dimY=np.amax(train_Y,axis=0)+1
    train_Y = np.eye(dimY)[train_Y]
    test_Y = np.eye(dimY)[test_Y]
    numdp = args['numdp']
    train_numdp = numdp

    print('dimX=',dimX)
    print('dimY=',dimY)

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



