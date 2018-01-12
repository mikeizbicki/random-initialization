def modify_parser(subparsers):
    import argparse
    from interval import interval

    parser = subparsers.add_parser('mnist', help='a handwritten digit dataset')
    parser.add_argument('--data_dir',type=str,default='data/mnist')
    parser.add_argument('--numdp',type=interval(int),default=55000)
    parser.add_argument('--numdp_test',type=interval(int),default=10000)

    parser.add_argument('--seed',type=interval(int),default=0)
    parser.add_argument('--label_corruption',type=interval(float),default=0)
    parser.add_argument('--noise',type=interval(float),default=0)
    parser.add_argument('--loud',type=interval(int),default=0)
    parser.add_argument('--ones',type=interval(int),default=0)

class Data:
    def __init__(self,args):

        import tensorflow as tf
        from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
        import numpy as np
        import random

        datasets = read_data_sets(args['data_dir'],False)

        class Dataset:
            def __init__(self):
                pass

        self.dimX = [28,28,1]
        self.dimY = 10

        self.train = Dataset()
        self.train.numdp = args['numdp']
        self.train.X = datasets.train._images.reshape([55000]+self.dimX)
        self.train.X = self.train.X[0:args['numdp'],...]
        self.train.Y = np.eye(10)[datasets.train._labels]
        self.train.Y = self.train.Y[0:args['numdp']]

        random.seed(args['seed'])
        np.random.seed(args['seed'])
        noise_binary=np.random.binomial(1,args['noise'],size=self.train.X.shape)
        noise_exponential=np.random.exponential(size=self.train.X.shape)
        noise=noise_binary*noise_exponential
        self.train.X=np.maximum(noise,self.train.X)

        #import code; code.interact(local=locals())

        random.seed(args['seed'])
        np.random.seed(args['seed'])
        num_corrupted=int(args['label_corruption']*self.train.numdp)
        Y_shift=np.random.randint(1,9,size=[num_corrupted])
        Y_shifted=(np.argmax(self.train.Y[0:num_corrupted],axis=1)+Y_shift)%10
        Y_corrupted=np.eye(10)[Y_shifted]
        self.train.Y[0:num_corrupted] = Y_corrupted

        self.train.X[0:args['loud']] *= args['numdp']
        self.train.Y[0:args['loud']] = self.train.Y[0]

        self.train.X[0:args['ones']] = 1+0*self.train.X[0:args['ones']]

        self.test = Dataset()
        self.test.numdp = 10000
        self.test.X = datasets.test._images.reshape([10000]+self.dimX)
        self.test.X = self.test.X[0:args['numdp_test'],...]
        self.test.Y = np.eye(10)[datasets.test._labels]
        self.test.Y = self.test.Y[0:args['numdp_test']]

