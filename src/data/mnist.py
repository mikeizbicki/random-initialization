import argparse

def modify_parser(subparsers):
    parser = subparsers.add_parser('mnist', help='a handwritten digit dataset')
    parser.add_argument('--data_dir',type=str,default='data/mnist')

class Data:
    def __init__(self,args):

        import tensorflow as tf
        from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
        import numpy as np
        #import random

        print('loading dataset mnist')
        datasets = read_data_sets(args['data_dir'],False)

        class Dataset:
            def __init__(self):
                pass

        self.dimX = 784
        self.dimY = 10
        self.train = Dataset()
        self.train.X = datasets.train._images
        self.train.Y = np.eye(10)[datasets.train._labels]
        self.train.numdp = 55000
        self.test = Dataset()
        self.test.X = datasets.test._images
        self.test.Y = np.eye(10)[datasets.test._labels]
        self.test.numdp = 20000
