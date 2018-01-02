def modify_parser(subparsers):
    import argparse
    from interval import interval

    parser = subparsers.add_parser('synthetic', help='generate a synthetic dataset')
    parser.add_argument('--variance',type=interval(float),default=0.1)
    parser.add_argument('--numdp',type=interval(int),default=30)
    parser.add_argument('--target',type=str,default='sin(x)')

class Data:
    def __init__(self,args):
        import tensorflow as tf
        self.args=args
        tfvars = dict([(item,eval('tf.'+item)) for item in dir(tf)]) 
        target = eval('lambda x: '+args.target,tfvars)
        self.target_true = target

    def eval_target(self,X):
        import tensorflow as tf
        with tf.Graph().as_default():
            x_ = tf.placeholder(tf.float32, [None,1])
            y_true = self.target_true(x_)
            sess = tf.Session()
            Y_true = sess.run(y_true,feed_dict={x_:X})
            return Y_true

    def generate_data(self,opts):
        import numpy as np
        import random
        random.seed(opts['seed_np'])
        np.random.seed(opts['seed_np'])
        xmin=-10
        xmax=10
        X = np.random.uniform(xmin,xmax,size=[opts['numdp'],1])
        Y_true = self.eval_target(X)
        random.seed(opts['seed_np'])
        np.random.seed(opts['seed_np'])
        Y = Y_true + np.random.normal(scale=opts['variance'],size=[opts['numdp'],1])
        return X,Y,Y_true
