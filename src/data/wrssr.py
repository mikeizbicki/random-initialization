def modify_parser(subparsers):
    import argparse
    from interval import interval

    parser = subparsers.add_parser('wrssr', help='an adversarial dataset from section 3.3 of "the marginal value of adaptive gradient methods in machine learning"')
    parser.add_argument('--data_dir',type=str,default='data/mnist')
    parser.add_argument('--numdp',type=interval(int),default=100)
    parser.add_argument('--numdp_test',type=interval(int),default=100)
    parser.add_argument('--p',type=interval(float),default=0.6,help='class imbalance; must be strictly greater than 0.5')

    parser.add_argument('--seed',type=interval(int),default=0)

class Data:
    def __init__(self,args):

        import tensorflow as tf
        from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
        import numpy as np
        import random


        class Dataset:
            def __init__(self):
                pass

        p = args['p']
        n = args['numdp']+args['numdp_test']
        d = 6*n
        X = np.zeros([n,d])

        random.seed(args['seed'])
        np.random.seed(args['seed'])

        Y = np.random.multinomial(1,[p,1-p],[n])

        X[:,0] = Y[:,0]-Y[:,1]
        X[:,1:3] = np.ones([n,2])
        for i in range (0,n):
            X[i,3+5*i] = 1
            if X[i,0] == -1:
                X[i,3+5*i+1] = 1
                X[i,3+5*i+2] = 1
                X[i,3+5*i+3] = 1
                X[i,3+5*i+4] = 1

        self.dimX=[d]
        self.dimY=2

        self.train=Dataset()
        self.train.numdp=args['numdp']
        self.train.X=X[:args['numdp'],:]
        self.train.Y=Y[:args['numdp'],:]

        self.test=Dataset()
        self.test.numdp=args['numdp_test']
        self.test.X=X[args['numdp']:,:]
        self.test.Y=Y[args['numdp']:,:]
