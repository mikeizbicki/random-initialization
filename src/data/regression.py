def modify_parser(subparsers):
    import argparse
    from interval import interval

    parser = subparsers.add_parser('regression', help='generate a synthetic dataset')
    parser.add_argument('--variance',type=interval(float),default=0.1)
    parser.add_argument('--numdp_test',type=interval(int),default=100)
    parser.add_argument('--numdp',type=interval(int),default=30)
    parser.add_argument('--dimX',type=interval(int),default=[1],nargs='*')
    parser.add_argument('--dimY',type=interval(int),default=1)
    parser.add_argument('--target',type=str,default='sin(x)')
    parser.add_argument('--seed',type=int,default=0)

class Data:
    def __init__(self,args):
        import tensorflow as tf
        self.args=args
        tfvars = dict([(item,eval('tf.'+item)) for item in dir(tf)]) 
        target = eval('lambda x: '+args['target'],tfvars)
        self.target_true = target

        def eval_target(X):
            import tensorflow as tf
            with tf.Graph().as_default():
                x_ = tf.placeholder(tf.float32, [None,1])
                y_true = self.target_true(x_)
                sess = tf.Session()
                Y_true = sess.run(y_true,feed_dict={x_:X})
                return Y_true

        class Dataset:
            def __init__(self,numdp,seed):
                import numpy as np
                import random

                random.seed(seed)
                np.random.seed(seed)
                xmin=-10
                xmax=10
                X = np.random.uniform(xmin,xmax,size=[numdp,1])
                Y_true = eval_target(X)

                random.seed(seed)
                np.random.seed(seed)
                Y = Y_true + np.random.normal(scale=args['variance'],size=[numdp,1])
                self.X=X
                self.Y=Y
        
        seed=args['seed']
        self.train=Dataset(args['numdp'],seed)
        self.test=Dataset(args['numdp_test'],seed+1)


