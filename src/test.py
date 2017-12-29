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
parser.add_argument('--numdp',type=int,default=30)

parser.add_argument('--scale',type=float,default=1.0)
parser.add_argument('--mean',type=float,default=0.0)
parser.add_argument('--randomness',choices=['normal','uniform','gamma'],default='normal')
parser.add_argument('--abs',action='store_true',default=False)

parser.add_argument('--batchsize',type=int,default=5)
parser.add_argument('--epochs',type=int,default=0)
parser.add_argument('--plotevery',type=int,default=10)
parser.add_argument('--learningrate',type=float,default=0.001)
parser.add_argument('--trainop',choices=['sgd','momentum','adam'])

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

fig, axs = plt.subplots(ncols=2)
fig.set_tight_layout(True)

ax_data=axs[0]
ax_data_lines=[]

steps=256
xmargin=0.1*(xmax-xmin)/2
ymargin=0.5*(np.amax(Y)-np.amin(Y))/2
colorstep=0.5/float(args.epochs)
ax_data_xs = np.linspace(xmin-xmargin,xmax+xmargin,steps).reshape(steps,1)

ax_data_xs2 = np.linspace(xmin,xmax,steps).reshape(steps,1)
ax_data_ys2 = sess.run(y_true,feed_dict={x_:ax_data_xs2})

ax_loss=axs[1]
ax_loss.set_yscale('log')
ax_loss_lines_ave=[]
ax_loss_lines_true=[]
ax_loss_lines_integral=[]
ax_loss_xs=[]
ax_loss_ys_ave=[]
ax_loss_ys_true=[]
ax_loss_ys_integral=[]

for i in range(0,args.trials):
    print('plotting '+str(i))
    try:
        for epoch in range(0,args.epochs+1):
            if epoch==0:
                tf.set_random_seed(args.seed_tf+i)
                sess.run(tf.global_variables_initializer())
                ax_data_ys = sess.run(y,feed_dict={x_:ax_data_xs})
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
                    ax_data_ys = sess.run(y,feed_dict={x_:ax_data_xs})

            if epoch%args.plotevery==0:
                res_ave,res_true=sess.run([loss_ave,loss_true_ave],feed_dict={x_:X,y_:Y})
                [res_integral]=sess.run([loss_true_ave],feed_dict={x_:ax_data_xs2})
                print('  epoch: %4d ; y_ave: %e ; y_true_ave: %e ; integral_ave %e'
                    %(epoch,res_ave,res_true,res_integral))

                ax_data_line,=ax_data.plot(ax_data_xs,ax_data_ys,'-')
                ax_data_lines.append(ax_data_line)

                ax_loss_xs.append(epoch)

                ax_loss_ys_ave.append(res_ave)
                ax_loss_line_ave,=ax_loss.plot(ax_loss_xs,ax_loss_ys_ave,'-',color=(1,0,0))
                ax_loss_lines_ave.append(ax_loss_line_ave)

                ax_loss_ys_true.append(res_true)
                ax_loss_line_true,=ax_loss.plot(ax_loss_xs,ax_loss_ys_true,'-',color=(0,1,0))
                ax_loss_lines_true.append(ax_loss_line_true)

                ax_loss_ys_integral.append(res_integral)
                ax_loss_line_integral,=ax_loss.plot(ax_loss_xs,ax_loss_ys_integral,'-',color=(0,0,1))
                ax_loss_lines_integral.append(ax_loss_line_integral)

    except KeyboardInterrupt:
        print('>>>>>>>>>>>>>> KeyboardInterupt <<<<<<<<<<<<<<')

#labelLines(plt.gca().get_lines(),zorder=2.5)
ax_data.set_xlim([xmin-xmargin,xmax+xmargin])
ax_data.set_ylim([np.amin(Y)-ymargin,np.amax(Y)+ymargin])

ltrue,=ax_data.plot(ax_data_xs2,ax_data_ys2,'-',color=(0,0,1))
ldata,=ax_data.plot(X,Y,'.',color=(0,0,1))

##############################
print('saving/displaying result')
from matplotlib.animation import FuncAnimation

def init_animation():
    for ax_data_line in ax_data_lines:
        ax_data_line.set_visible(False)
    
    for ax_loss_line_ave in ax_loss_lines_ave:
        ax_loss_line_ave.set_visible(False)

    for ax_loss_line_true in ax_loss_lines_true:
        ax_loss_line_true.set_visible(False)

    for ax_loss_line_integral in ax_loss_lines_integral:
        ax_loss_line_integral.set_visible(False)

def update_animation(i):
    ax_loss_lines_ave[i].set_visible(True)
    ax_loss_lines_true[i].set_visible(True)
    ax_loss_lines_integral[i].set_visible(True)
    ax_data_lines[i].set_visible(True)
    plt.setp(ax_data_lines[i],color=(0,0.5,0))

    gradientlen=5
    colorstep=0.5/gradientlen
    for j in range(1,gradientlen):
        if i-j>=0:
            plt.setp(ax_data_lines[i-j],color=(j*colorstep,0.5+j*colorstep,j*colorstep))

ani = FuncAnimation(fig, update_animation, frames=range(0,len(ax_data_lines)),
                    init_func=init_animation, blit=True)

#basename=' '.join(sys.argv[1:])
basename='test'
fig.set_size_inches(6,3)
#ani.save(basename+'.gif', dpi=96, writer='imagemagick')
ani.save(basename+'.mp4', dpi=96, writer='ffmpeg')
plt.savefig(basename+'.png',dpi=96)
#plt.show()
