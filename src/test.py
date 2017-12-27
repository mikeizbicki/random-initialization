#!/usr/bin/env python

from __future__ import print_function

import sys

########################################
print('processing cmdline args')
import argparse

parser=argparse.ArgumentParser('random initialization tests')

parser.add_argument('--seed_tf',type=int,default=0)
parser.add_argument('--seed_np',type=int,default=0)
parser.add_argument('--trials',type=int,default=1)
parser.add_argument('--scale',type=float,default=1.0)
parser.add_argument('--randomness',choices=['normal','uniform','gamma'],default='normal')
parser.add_argument('--abs',action='store_true',default=False)
parser.add_argument('--layers',type=int,nargs='*',default=[100])
parser.add_argument('--activation',choices=['relu','relu6','crelu','elu','softplus','softsign','sigmoid','tanh'],default='relu')

args=parser.parse_args()

########################################
print('setting tensorflow options')
import tensorflow as tf

activation=eval('tf.nn.'+args.activation)

def randomness(shape):
    ret=None
    if args.randomness=='normal':
        ret=tf.random_normal(shape,stddev=args.scale)
    if args.randomness=='uniform':
        ret=tf.random_uniform(shape,minval=-args.scale,maxval=args.scale)
    if args.randomness=='gamma':
        alpha=1
        beta=alpha/args.scale**2
        ret=tf.sign(tf.random_uniform(shape,minval=-args.scale,maxval=args.scale)) * tf.random_gamma(shape,alpha=alpha,beta=beta)
    if args.abs:
        ret=tf.abs(ret)
    return ret

########################################
print('initializing tensorflow graph')
with tf.name_scope('inputs'):
    x_ = tf.placeholder(tf.float32, [None,1])
    y_ = tf.placeholder(tf.float32, [None,1])

    bias=1.0
    layer=0
    y = x_
    n0 = 1
    for n in args.layers:
        print('layer'+str(layer)+' nodes: '+str(n))
        with tf.name_scope('layer'+str(layer)):
            w = tf.Variable(randomness([n0,n]))
            b = tf.Variable(bias)
            y = activation(tf.matmul(y,w)+b)
            print('w=',w.get_shape())
            print('y=',y.get_shape())
        n0 = n
        if args.activation=='crelu':
            n0*=2
        layer+=1

    with tf.name_scope('layer_final'):
        print('layer_final ')
        w = tf.Variable(randomness([n0,1]))
        b = tf.Variable(bias)
        print('w=',w.get_shape())
        print('y=',y.get_shape())
        y = tf.matmul(y,w)+b

sess = tf.Session()
tf.set_random_seed(args.seed_tf)
sess.run(tf.global_variables_initializer())

def eval_graph(x):
    yhat = sess.run(y,feed_dict={x_:x})
    return yhat

########################################
print('plotting')
import matplotlib.pyplot as plt
import numpy as np

xmin=-10*args.scale**(-1/2)
xmax=10*args.scale**(-1/2)
xs = np.linspace(xmin,xmax,100).reshape(100,1)
for i in range(0,args.trials):
    print('plotting '+str(i))
    sess.run(tf.global_variables_initializer())
    ys = eval_graph(xs)
    plt.plot(xs,ys, '-')

plt.show()
