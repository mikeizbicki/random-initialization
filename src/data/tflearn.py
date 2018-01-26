def modify_parser(subparsers):
    import argparse
    from interval import interval

    parser = subparsers.add_parser('tflearn', help='toy deep learning image datasets from tflearn library')
    parser.add_argument('--name',choices=['cifar10','cifar100','imdb','mnist','oxflower17','svhn','titanic'])
    parser.add_argument('--data_dir',type=str,default='data/tflearn')
    parser.add_argument('--numdp',type=interval(int),default=1e10)
    parser.add_argument('--numdp_balanced',action='store_true')
    parser.add_argument('--numdp_test',type=interval(int),default=1e10)
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
    elif args['name']=='imdb':
        (train_X, train_Y), _, (test_X, test_Y) = tflearn.datasets.imdb.load_data(args['data_dir']+'/imdb.pkl')
        train_X = tflearn.data_utils.pad_sequences(train_X, maxlen=100, value=0.)
        train_Y = tflearn.data_utils.to_categorical(train_Y,nb_classes=2)
        test_Y = tflearn.data_utils.to_categorical(test_Y,nb_classes=2)
        test_X = tflearn.data_utils.pad_sequences(test_X, maxlen=100, value=0.)
        train_Y = np.argmax(train_Y,axis=1)
        test_Y = np.argmax(test_Y,axis=1)
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
    train_numdp = min(args['numdp'],train_X.shape[0])
    test_numdp = min(args['numdp_test'],test_X.shape[0])

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
        numdp_per_class=train_numdp/dimY
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
        train_X = train_X[0:train_numdp,...]
        train_Y = train_Y[0:train_numdp]

    Id = np.array(range(0,train_numdp))
    train=tf.data.Dataset.from_tensor_slices((np.float32(train_X),np.float32(train_Y),Id))

    # testing data

    test_X = test_X[0:test_numdp,...]
    test_Y = test_Y[0:test_numdp]
    Id = np.array(range(0,test_numdp))
    test=tf.data.Dataset.from_tensor_slices((np.float32(test_X),np.float32(test_Y),Id))



