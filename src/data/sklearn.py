def modify_parser(subparsers):
    import argparse
    from interval import interval

    parser = subparsers.add_parser('sklearn', help='load a dataset from scikit-learn')
    parser.add_argument('--name',choices=['boston','iris','diabetes','digits','linnerud','wine','breast_cancer'])
    parser.add_argument('--train_frac',type=interval(float),default=0.5)
    parser.add_argument('--seed',type=int,default=0)

def init(args):
    import tensorflow as tf
    import numpy as np
    import random

    random.seed(args['seed'])
    np.random.seed(args['seed'])
    from sklearn import datasets

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

    regression=['boston','linnerud','diabetes']
    classification=['iris','digits','wine','breast_cancer']

    load_data=eval('datasets.load_'+args['name'])
    Xall,Yargmax=load_data(return_X_y=True)
    numdp=Yargmax.shape[0]
    dimX=list(Xall.shape)[1:]
    if args['name'] in classification:
        dimY=np.amax(Yargmax,axis=0)+1
        Yall = np.eye(dimY)[Yargmax]
    elif args['name'] in regression:
        print('Yargmax=',Yargmax.shape)
        try:
            dimY=Yargmax.shape[1]
        except:
            dimY=1
        Yall = Yargmax.reshape([Yargmax.shape[0],dimY])

    random.seed(args['seed'])
    np.random.seed(args['seed'])
    np.random.shuffle(Xall)

    random.seed(args['seed'])
    np.random.seed(args['seed'])
    np.random.shuffle(Yall)

    train_numdp=int(float(numdp)*args['train_frac'])
    print('dimX=',dimX)
    print('dimY=',dimY)
    print('numdp=',numdp)
    print('train_numdp=',train_numdp)
    train_X=Xall[0:train_numdp,...]
    train_Y=Yall[0:train_numdp,...]
    train_Id = np.array(range(0,train_numdp))
    train=tf.data.Dataset.from_tensor_slices((np.float32(train_X),np.float32(train_Y),train_Id))

    test_numdp=numdp-train_numdp
    test_X=Xall[train_numdp:,...]
    test_Y=Yall[train_numdp:,...]
    test_Id = np.array(range(0,test_numdp))
    test=tf.data.Dataset.from_tensor_slices((np.float32(test_X),np.float32(test_Y),test_Id))

    valid_numdp=test_numdp
    valid_X=test_X
    valid_Y=test_Y
    valid_Id=test_Id
    valid=test

    global train_Y_max
    train_Y_max = np.float32(np.amax(train_Y,axis=0))
