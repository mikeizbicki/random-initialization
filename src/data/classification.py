def modify_parser(subparsers):
    import argparse
    from interval import interval

    parser = subparsers.add_parser('classification', help='generate a synthetic dataset for binary classification')
    parser.add_argument('--variance',type=interval(float),default=1)
    parser.add_argument('--numdp',type=interval(int),default=100)
    parser.add_argument('--numdp_test',type=interval(int),default=100)
    parser.add_argument('--dimX',type=interval(int),default=100)
    parser.add_argument('--dimY',type=interval(int),default=2)
    
    parser.add_argument('--noise',type=interval(float),default=0)

    parser.add_argument('--seed',type=int,default=0)

class Data:
    def __init__(self,args):

        import tensorflow as tf
        import numpy as np
        import random

        self.args=args
        self.dimX=args['dimX']; dimX=args['dimX']
        self.dimY=args['dimY']; dimY=args['dimY']
        seed=args['seed']

        random.seed(seed)
        np.random.seed(seed)
        mu = args['variance']*np.random.normal(size=[dimX,dimY])

        class Dataset:
            def __init__(self,numdp):
                self.Y = np.random.multinomial(1,[1/float(dimY)]*dimY,size=[numdp])
                self.X = np.zeros([numdp,dimX])
                self.numdp=args['numdp']
                for i in range(0,numdp):
                    Yval = np.nonzero(self.Y[i])[0][0]
                    self.X[i,:] = mu[:,Yval] + np.random.normal(size=[1,dimX])

        self.train=Dataset(args['numdp'])
        self.test=Dataset(args['numdp_test'])

        self.train.X[0,:] = np.ones([1,dimX])*args['noise']
