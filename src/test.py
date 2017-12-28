#!/usr/bin/env python

from __future__ import print_function

import math
import sys

########################################
print('processing cmdline args')
import argparse

parser=argparse.ArgumentParser('random initialization tests')

parser.add_argument('--seed_tf',type=int,default=0)
parser.add_argument('--seed_np',type=int,default=0)
parser.add_argument('--trials',type=int,default=1)
parser.add_argument('--target',choices=['quadratic','sin'])
parser.add_argument('--variance',type=float,default=0.1)

parser.add_argument('--batchsize',type=int,default=5)
parser.add_argument('--numdp',type=int,default=30)
parser.add_argument('--epochs',type=int,default=0)
parser.add_argument('--stepplot',type=int,default=10)
parser.add_argument('--learningrate',type=float,default=0.001)
parser.add_argument('--trainop',choices=['sgd','momentum','adam'])

parser.add_argument('--mean',type=float,default=0.0)
parser.add_argument('--scale',type=float,default=1.0)
parser.add_argument('--randomness',choices=['normal','uniform','gamma'],default='normal')
parser.add_argument('--abs',action='store_true',default=False)

parser.add_argument('--layers',type=int,nargs='*',default=[100])
parser.add_argument('--activation',choices=['relu','relu6','crelu','elu','softplus','softsign','sigmoid','tanh'],default='relu')

args=parser.parse_args()

########################################
print('setting tensorflow options')
import tensorflow as tf
tf.set_random_seed(args.seed_tf)

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
    return args.mean+ret

########################################
print('initializing tensorflow graph')

with tf.name_scope('inputs'):
    x_ = tf.placeholder(tf.float32, [None,1])
    y_ = tf.placeholder(tf.float32, [None,1])

with tf.name_scope('true'):
    if args.target=='sin':
        y_true = tf.sin(x_)
    else:
        y_true = x_**2/10

with tf.name_scope('model'):
    bias=1.0
    layer=0
    y = x_
    n0 = 1
    for n in args.layers:
        print('layer'+str(layer)+' nodes: '+str(n))
        with tf.name_scope('layer'+str(layer)):
            w = tf.Variable(randomness([n0,n]))
            b = tf.Variable(bias*tf.ones([1,n]))
            y = activation(tf.matmul(y,w)+b)
        n0 = n
        if args.activation=='crelu':
            n0*=2
        layer+=1

    with tf.name_scope('layer_final'):
        #w = tf.Variable(randomness([n0,1]))
        w = tf.ones([n0,1])
        b = tf.Variable(bias)
        y = tf.matmul(y,w)+b

with tf.name_scope('eval'):
    loss = (y_-y)**2
    loss_true = (y_true-y)**2

    loss_ave = tf.reduce_sum(loss)/tf.cast(tf.size(x_),tf.float32)
    loss_true_ave = tf.reduce_sum(loss_true)/tf.cast(tf.size(x_),tf.float32)

if args.trainop=='sgd':
    train_op = tf.train.GradientDescentOptimizer(args.learningrate).minimize(loss)
elif args.trainop=='momentum':
    train_op = tf.train.MomentumOptimizer(args.learningrate, 0.9).minimize(loss)
elif args.trainop=='adam':
    train_op = tf.train.AdamOptimizer(args.learningrate).minimize(loss)

sess = tf.Session()

########################################
print('generating data')
import random
import numpy as np
random.seed(args.seed_np)
np.random.seed(args.seed_np)

xmin=-10*args.scale**(-1/2)
xmax=10*args.scale**(-1/2)

X = np.random.uniform(xmin,xmax,size=[args.numdp,1])
Y_true = sess.run(y_true,feed_dict={x_:X})
Y = Y_true + np.random.normal(scale=args.variance,size=[args.numdp,1])

########################################
print('plotting')
import matplotlib.pyplot as plt
from label_lines import *


steps=256
xmargin=0.1*(xmax-xmin)/2
ymargin=0.5*(np.amax(Y)-np.amin(Y))/2
colorstep=0.5/float(args.epochs)
xs = np.linspace(xmin-xmargin,xmax+xmargin,steps).reshape(steps,1)

for i in range(0,args.trials):
    print('plotting '+str(i))
    try:
        for epoch in range(0,args.epochs+1):
            if epoch==0:
                tf.set_random_seed(args.seed_tf+i)
                sess.run(tf.global_variables_initializer())
                ys = sess.run(y,feed_dict={x_:xs})
            else:
                rng_state = np.random.get_state()
                np.random.shuffle(X)
                np.random.set_state(rng_state)
                np.random.shuffle(Y)
                for batchstart in range(0,args.numdp,args.batchsize):
                    batchstop=batchstart+args.batchsize
                    Xbatch=X[batchstart:batchstop]
                    Ybatch=Y[batchstart:batchstop]
                    sess.run(train_op,feed_dict={x_:Xbatch,y_:Ybatch})
                    ys = sess.run(y,feed_dict={x_:xs})

            if epoch%args.stepplot==0:
                res_ave,res_true_ave=sess.run([loss_ave,loss_true_ave],feed_dict={x_:X,y_:Y})
                [res_int]=sess.run([loss_true_ave],feed_dict={x_:xs})
                print('  epoch: %4d ; y_ave: %e ; y_true_ave: %e ; integral_ave %e'%(epoch,res_ave,res_true_ave,res_int))
                plt.plot(xs,ys, '-',label=str(epoch),color=(0.5-colorstep*epoch,1,0.5-colorstep*epoch))
    except KeyboardInterrupt:
        print('>>>>>>>>>>>>>> KeyboardInterupt <<<<<<<<<<<<<<')
    plt.plot(xs,ys, '-',label=str(epoch),color=(0,0.5,0))

#labelLines(plt.gca().get_lines(),zorder=2.5)
plt.xlim([xmin-xmargin,xmax+xmargin])
plt.ylim([np.amin(Y)-ymargin,np.amax(Y)+ymargin])

ys = sess.run(y_true,feed_dict={x_:xs})
plt.plot(xs,ys,'-',color=(0,0,1))
plt.plot(X,Y,'.',color=(0,0,1))
plt.show()
