def modify_parser(subparsers):
    import argparse
    from interval import interval

    parser = subparsers.add_parser('synthetic', help='generate a synthetic dataset')
    parser.add_argument('--type',choices=['regression','classification'],default='classification')
    parser.add_argument('--variance',type=interval(float),default=1)
    parser.add_argument('--numdp',type=interval(int),default=100)
    parser.add_argument('--numdp_test',type=interval(int),default=100)
    parser.add_argument('--numdp_valid',type=interval(int),default=100)
    parser.add_argument('--dimX',type=interval(int),default=[100],nargs='*')
    parser.add_argument('--dimX_nuisance',type=interval(int),nargs='+',default=[0])
    parser.add_argument('--dimY',type=interval(int),default=2)

    parser.add_argument('--seed',type=int,default=0)

def init(args):
    import tensorflow as tf
    import numpy as np
    import random

    global train
    global train_numdp
    global train_X
    global train_Y
    global valid
    global valid_numdp
    global valid_X
    global valid_Y
    global test
    global test_numdp
    global test_X
    global test_Y
    global dimX 
    global dimY 

    dimX_orig=args['dimX']
    dimX_nuisance=args['dimX_nuisance']
    #dimX=dimX_nuisance 
    dimX=[a+b for (a,b) in zip(dimX_orig,dimX_nuisance)]
    dimY=args['dimY']

    random.seed(args['seed'])
    np.random.seed(args['seed'])
    mu = args['variance']*np.random.normal(size=dimX_orig+[dimY])

    random.seed(args['seed'])
    np.random.seed(args['seed'])
    projection = np.random.normal(size=dimX_orig+dimX)

    def make_data(numdp,seed):
        random.seed(seed)
        np.random.seed(seed)

        if args['type']=='classification':
            Y = np.random.multinomial(1,[1/float(dimY)]*dimY,size=[numdp])
            X = np.zeros([numdp]+dimX_orig)
            for i in range(0,numdp):
                Yval = np.nonzero(Y[i])[0][0]
                X[i,...] = mu[...,Yval] + np.random.normal(size=[1]+dimX_orig)

        elif args['type']=='regression':
            X = np.random.uniform(size=[numdp]+dimX_orig)
            Y = np.dot(X,mu)

        X = np.dot(X,projection)

        Id = np.array(range(0,numdp))
        return np.float32(X),np.float32(Y),Id

    train_numdp=args['numdp']
    train_X,train_Y,Id=make_data(args['numdp'],args['seed'])
    train=tf.data.Dataset.from_tensor_slices((train_X,train_Y,Id))

    valid_numdp=args['numdp_valid']
    valid_X,valid_Y,Id=make_data(args['numdp_valid'],args['seed']+1)
    valid=tf.data.Dataset.from_tensor_slices((valid_X,valid_Y,Id))

    test_numdp=args['numdp_test']
    test_X,test_Y,Id=make_data(args['numdp_test'],args['seed']+1)
    test=tf.data.Dataset.from_tensor_slices((test_X,test_Y,Id))

