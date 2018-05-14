from __future__ import print_function

def modify_parser(subparsers):
    import argparse
    from interval import interval

    parser = subparsers.add_parser('mean', help='generate a synthetic dataset for mean estimation')
    parser.add_argument('--mu_scale',type=interval(float),default=1.0)
    parser.add_argument('--mu_offset',type=interval(float),default=0.0)
    parser.add_argument('--scale',type=interval(float),default=1.0)
    parser.add_argument('--random_type',choices=['scale','entries','full'],default='entries')
    parser.add_argument('--randomness',choices=['normal','exponential','pareto','cauchy'],default='normal')
    parser.add_argument('--corruption',type=interval(float),default=0.0)
    parser.add_argument('--numdp',type=interval(int),default=100)
    parser.add_argument('--numdp_test',type=interval(int),default=100)
    parser.add_argument('--numdp_valid',type=interval(int),default=100)
    parser.add_argument('--dimY',type=interval(int),default=100)

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
    global mu

    dimX=[1]
    dimY=args['dimY']

    random.seed(args['seed'])
    np.random.seed(args['seed'])

    if args['random_type']=='scale':
        def random_angle():
            v = np.random.normal(size=[1,dimY])
            return v/np.linalg.norm(v)
        normal      = lambda:random_angle()*np.random.normal()
        exponential = lambda:random_angle()*np.random.exponential()
        pareto      = lambda:random_angle()*np.random.pareto(1)
        cauchy      = lambda:random_angle()*np.random.standard_cauchy()
        random_generator = eval(args['randomness'])
    elif args['random_type']=='entries':
        normal      = lambda:args['scale']*np.random.normal(size=[1,dimY])
        exponential = lambda:np.random.exponential(args['scale'],size=[1,dimY])
        pareto      = lambda:np.random.pareto(args['scale'],size=[1,dimY])
        cauchy      = lambda:args['scale']*np.random.standard_cauchy(size=[1,dimY])
    elif args['random_type']=='full':
        import scipy.stats as stats
        rot = stats.ortho_group.rvs(dimY)
        normal      = lambda:np.dot(args['scale']*np.random.normal(size=[1,dimY]),rot)
        exponential = lambda:np.dot(np.random.exponential(args['scale'],size=[1,dimY]),rot)
        pareto      = lambda:np.dot(np.random.pareto(args['scale'],size=[1,dimY]),rot)
        cauchy      = lambda:np.dot(args['scale']*np.random.standard_cauchy(size=[1,dimY]),rot)
    random_generator = eval(args['randomness'])

    scale=args['mu_scale']
    offset=args['mu_offset']*np.ones([1,dimY])
    mu = offset + np.zeros([1,dimY]) #scale*normal()
    #mu /= np.linalg.norm(mu)/scale
    mu_corrupt = offset + np.ones([1,dimY]) #scale*normal()
    #mu_corrupt /= np.linalg.norm(mu_corrupt)/scale

    print('|mu-mu_corrupt|^2=',np.sum((mu-mu_corrupt)**2))

    def make_data(numdp,corruption,seed):
        random.seed(seed)
        np.random.seed(seed)

        Y = np.zeros([numdp,dimY])
        numdp_bad=int(numdp*corruption)
        for i in range(0,numdp):
            if i<numdp_bad:
                Y[i,...] = random_generator() + mu_corrupt
            else:
                Y[i,...] = random_generator() + mu

        Id = np.array(range(0,numdp))
        return np.float32(Id),np.float32(Y),Id

    train_numdp=args['numdp']
    train_X,train_Y,Id=make_data(args['numdp'],args['corruption'],args['seed'])
    train=tf.data.Dataset.from_tensor_slices((train_X,train_Y,Id))

    valid_numdp=args['numdp_valid']
    valid_X,valid_Y,Id=make_data(args['numdp_valid'],0,args['seed']+1)
    valid=tf.data.Dataset.from_tensor_slices((valid_X,valid_Y,Id))

    test_numdp=args['numdp_test']
    test_X,test_Y,Id=make_data(args['numdp_test'],0,args['seed']+1)
    test=tf.data.Dataset.from_tensor_slices((test_X,test_Y,Id))

    global mu_hat
    global mu_hat_mse
    mu_hat=np.average(train_Y,axis=0).reshape([1,dimY])
    mu_hat_mse=np.sum((mu-mu_hat)**2)
    print('|mu-mu_hat|^2=',mu_hat_mse)

    global mu_med
    global mu_med_mse
    mu_med=np.median(train_Y,axis=0).reshape([1,dimY])
    mu_med_mse=np.sum((mu-mu_med)**2)
    print('|mu-mu_med|^2=',mu_med_mse)

    global mu_hat_star
    global mu_hat_star_mse
    train_Y_star = train_Y
    train_Y_star[0:int(train_numdp*args['corruption'])] -= mu_corrupt - mu
    mu_hat_star=np.average(train_Y_star,axis=0).reshape([1,dimY])
    mu_hat_star_mse=np.sum((mu-mu_hat_star)**2)
    print('|mu-mu_hat_star|^2=',mu_hat_star_mse)

    global mu_hat_star2
    global mu_hat_star2_mse
    train_Y_star2 = train_Y[int(train_numdp*args['corruption']):]
    mu_hat_star2=np.average(train_Y_star2,axis=0).reshape([1,dimY])
    mu_hat_star2_mse=np.sum((mu-mu_hat_star2)**2)
    print('|mu-mu_hat_star2|^2=',mu_hat_star2_mse)

    global naive_accuracies
    naive_accuracies=[mu_hat_star_mse,mu_hat_star2_mse,mu_hat_mse,mu_med_mse]
