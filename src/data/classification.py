def modify_parser(subparsers):
    import argparse
    from interval import interval

    parser = subparsers.add_parser('classification', help='generate a synthetic dataset for binary classification')
    parser.add_argument('--variance',type=interval(float),default=1)
    parser.add_argument('--numdp',type=interval(int),default=100)
    parser.add_argument('--numdp_test',type=interval(int),default=100)
    parser.add_argument('--dimX',type=interval(int),default=[100],nargs='*')
    parser.add_argument('--dimY',type=interval(int),default=2)
    
    parser.add_argument('--dimX_nuisance',type=interval(int),default=[0],nargs='*')
    parser.add_argument('--noise',type=interval(int),default=0)
    parser.add_argument('--label_flip',type=interval(int),default=0)

    parser.add_argument('--seed',type=int,default=0)

def init(args):
    import tensorflow as tf
    import numpy as np
    import random

    global train
    global test
    global dimX 
    global dimY 

    dimX_orig=args['dimX']
    dimX_nuisance=args['dimX_nuisance']
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

        Y = np.random.multinomial(1,[1/float(dimY)]*dimY,size=[numdp])
        X = np.zeros([numdp]+dimX_orig)
        for i in range(0,numdp):
            Yval = np.nonzero(Y[i])[0][0]
            X[i,...] = mu[...,Yval] + np.random.normal(size=[1]+dimX_orig)
            #print('Yval=',Yval,'Y=',Y[i,...],'Xval=',X[i,...])
        X = np.dot(X,projection)
        #X += np.random.normal(size=[numdp]+dimX)*10
        return np.float32(X),np.float32(Y)

    X,Y=make_data(args['numdp'],args['seed'])
    X[0:args['label_flip'],...] *= 1 #args['numdp']
    Y[0:args['label_flip'],...] = Y[0:args['label_flip'],...]*(-1) + 1
    X[0:args['noise'],...] += 100*np.random.normal(size=dimX)
    train=tf.data.Dataset.from_tensor_slices((X,Y))

    X,Y=make_data(args['numdp_test'],args['seed']+1)
    test=tf.data.Dataset.from_tensor_slices((X,Y))
