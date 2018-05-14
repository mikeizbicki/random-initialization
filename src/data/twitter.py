"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

def modify_parser(subparsers):
    import argparse
    from interval import interval

    parser = subparsers.add_parser('twitter', help='twitter json data from the firehose')
    parser.add_argument('--data_dir',type=str,default='/rhome/mizbicki/twitter')
    parser.add_argument('--seed',type=interval(int),default=0)

def init(args):

    global train
    global valid
    global test
    global dimX
    global dimY

    import tensorflow as tf
    import tflearn
    import numpy as np
    import random

    random.seed(args['seed'])
    np.random.seed(args['seed'])


    # constructing filenames

    import os
    files_train=[]
    files_validation=[]
    files_test=[]
    files_all=[]

    for path_date in os.listdir(args['data_dir']):
        path_date_full=os.path.join(args['data_dir'],path_date)
        if os.path.isdir(path_date_full):
            for path_hour in os.listdir(path_date_full):
                files_all.append(os.path.join(path_date_full,path_hour))

    files_all.sort()
    files_all=files_all
    print('files_all=',len(files_all))

    files_train=files_all[1]
    files_validation=files_all[0]

    # make tensors

    import simplejson as json

    dimX=[2]
    dimY=2

    def parse(str):
        data=json.loads(str)

        try:
            lat=data['geo']['coordinates'][0]
            lon=data['geo']['coordinates'][1]
        except:
            try:
                def centroid(xs):
                    coords=xs[0]
                    lats=[coord[0] for coord in coords]
                    lons=[coord[1] for coord in coords]
                    lat=sum(lats)/float(len(lats))
                    lon=sum(lons)/float(len(lons))
                    coord=(lat,lon)
                    return coord

                coord=centroid(data['place']['bounding_box']['coordinates'])
                lat=coord[1]
                lon=coord[0]
            except:
                lat=0
                lon=0

        x=np.float32(np.array([1.0,0.0]))
        y=np.float32(np.array([lat,lon]))
        id=np.int32(np.array([1]))
        return (x,y,id)


    train=tf.data.TextLineDataset(files_train)
    train=train.map(lambda x: tf.py_func(parse,[x],(tf.float32,tf.float32,tf.int32)))

    valid=tf.data.TextLineDataset(files_validation)
    valid=valid.map(lambda x: tf.py_func(parse,[x],(tf.float32,tf.float32,tf.int32)))
    #valid=valid.map(parse)
    #valid=train
    test=valid

    #raise ValueError('test')
