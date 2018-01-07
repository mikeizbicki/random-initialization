def modify_parser(subparsers):
    import argparse
    from interval import interval

    parser = subparsers.add_parser('classification', help='generate a synthetic dataset for binary classification')
    parser.add_argument('--variance',type=interval(float),default=0.1)
    parser.add_argument('--numdp',type=interval(int),default=100)
    parser.add_argument('--numdp_test',type=interval(int),default=100)
    parser.add_argument('--numdim',type=interval(int),default=100)
    parser.add_argument('--numclass',type=interval(int),default=2)

    parser.add_argument('--response',type=str,default='gaussian')
    parser.add_argument('--seed',type=int,default=0)

class Data:
    def __init__(self,args):

        import tensorflow as tf
        import numpy as np
        import random

        self.args=args
        numdim=args['numdim']
        seed=args['seed']

        class Dataset:
            def __init__(self,numdp,seed):
                random.seed(seed)
                np.random.seed(seed)
                self.mu = np.zeros(shape=[numdim])
                self.wstar = np.random.normal(size=[numdim])
                self.X = self.mu+np.random.normal(size=[numdp,numdim])
                self.Yprob = 1/(1+np.exp(-np.dot(self.X,self.wstar)))
                self.Y = np.random.binomial(1,self.Yprob).reshape([numdp,1])

        self.train=Dataset(args['numdp'],seed)
        self.test=Dataset(args['numdp_test'],seed+1)

        self.train.X[50,:] = np.ones([1,numdim])*args['numdp']
        self.train.Y[50,]  = 1
