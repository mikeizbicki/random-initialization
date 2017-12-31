#!/usr/bin/env python

from __future__ import print_function

import copy
import math
import sys

########################################
print('processing cmdline args')
import argparse

class Interval:
    def __init__(self,a,b):
        self.start=a
        self.stop=b
        self.is_trivial=(a==b)

def interval(t):
    def mkinterval(str):
        try:
            val=t(str)
            return Interval(val,val)
        except ValueError:
            str1,str2=str.split(':')
            val1=t(str1)
            val2=t(str2)
            return Interval(val1,val2)
    return mkinterval

parser=argparse.ArgumentParser()

parser_network = parser.add_argument_group('network options')
parser_network.add_argument('--epochs',type=int,default=100)
parser_network.add_argument('--batchsize',type=interval(int),default=5)
parser_network.add_argument('--learningrate',type=interval(float),default=0.001)
parser_network.add_argument('--trainop',choices=['sgd','momentum','adam'],default='adam')
parser_network.add_argument('--layers',type=interval(int),nargs='*',default=[interval(int)(100)])
parser_network.add_argument('--activation',choices=['relu','relu6','crelu','elu','softplus','softsign','sigmoid','tanh'],default='relu')

parser_weights = parser.add_argument_group('weight initialization')
parser_weights.add_argument('--scale',type=interval(float),default=1.0)
parser_weights.add_argument('--mean',type=interval(float),default=0.0)
parser_weights.add_argument('--randomness',choices=['normal','uniform','gamma'],default='normal')
parser_weights.add_argument('--abs',type=bool,default=False)
parser_weights.add_argument('--normalize',type=bool,default=True)

parser_data = parser.add_argument_group('data options')
parser_data.add_argument('--target',choices=['quadratic','sin'])
parser_data.add_argument('--variance',type=interval(float),default=0.1)
parser_data.add_argument('--numdp',type=interval(int),default=30)

parser_graphing = parser.add_argument_group('graphing options')
parser_graphing.add_argument('--steps',type=int,default=0)
parser_graphing.add_argument('--plotevery',type=int,default=10)
parser_graphing.add_argument('--zoomtarget',type=bool,default=True)
parser_graphing.add_argument('--seed_tf',type=int,default=0)
parser_graphing.add_argument('--seed_node',type=int,default=0)
parser_graphing.add_argument('--seed_np',type=int,default=0)

args=parser.parse_args()

########################################

import tensorflow as tf
tf.Session()
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation

################################################################################
print('initialize global plotting variables')

fig = plt.figure()
plt.tight_layout()
gs = gridspec.GridSpec(2,1)

#fig, axs = plt.subplots(ncols=2,nrows=args.steps+1)
#fig.set_tight_layout(True)

xmin=-10
xmax=10

steps=256
xmargin=0.1*(xmax-xmin)/2
ax_data_xs = np.linspace(xmin-xmargin,xmax+xmargin,steps).reshape(steps,1)

plotinfo={}
for step in range(0,args.steps+1):
    plotinfo[step]={}
    plotinfo[step]['ax_data']=fig.add_subplot(gs[0],label='ax_data step '+str(step))
    plotinfo[step]['ax_data_lines']=[]

    plotinfo[step]['ax_loss']=fig.add_subplot(gs[1],label='ax_loss step '+str(step))
    plotinfo[step]['ax_loss'].set_yscale('log')
    plotinfo[step]['ax_loss_lines_ave']=[]
    plotinfo[step]['ax_loss_lines_true']=[]
    plotinfo[step]['ax_loss_lines_integral']=[]
    plotinfo[step]['ax_loss_xs']=[]
    plotinfo[step]['ax_loss_ys_ave']=[]
    plotinfo[step]['ax_loss_ys_true']=[]
    plotinfo[step]['ax_loss_ys_integral']=[]

titles_step=[]
titles_data=[]

################################################################################
try:
    for step in range(0,args.steps+1):
        print('step '+str(step))

        ########################################
        print('  processing commandline options for step')
        opts=copy.deepcopy(args)
        param_names=filter(lambda x: x[0]!='_', dir(args))

        title_step=[]
        for param_name in param_names:
            param=eval('args.'+param_name)
            if isinstance(param,Interval):
                res=param.start+step*(param.stop-param.start)/(args.steps+1)
                exec('opts.'+param_name+' = '+str(res))
                if not param.is_trivial:
                    title_step.append(param_name+' = '+str(res))
            elif type(param) is list:
                ress=[]
                all_trivial=True
                for p in param:
                    res=p.start+step*(p.stop-p.start)/(args.steps+1)
                    ress.append(res)
                    if not p.is_trivial:
                        all_trivial=False
                if not all_trivial:
                    title_step.append(param_name+' = '+str(ress))
                exec('opts.'+param_name+' = '+str(ress))
        titles_step.append(' ; '.join(title_step))

        tf.set_random_seed(opts.seed_tf)
        random.seed(opts.seed_np)
        np.random.seed(opts.seed_np)

        ########################################
        print('  setting tensorflow options')
        with tf.Graph().as_default():

            activation=eval('tf.nn.'+opts.activation)

            def randomness(shape):
                ret=None
                if opts.randomness=='normal':
                    ret=tf.random_normal(shape,seed=opts.seed_node)
                if opts.randomness=='uniform':
                    ret=tf.random_uniform(shape,minval=-1,maxval=1,seed=opts.seed_node)
                if opts.randomness=='gamma':
                    alpha=1
                    beta=1
                    sign=tf.sign(tf.random_uniform(shape,minval=-1,maxval=1,seed=opts.seed_node))
                    ret=sign*tf.random_gamma(shape,alpha=alpha,beta=beta,seed=opts.seed_node)
                if opts.abs:
                    ret=tf.abs(ret)
                if opts.normalize:
                    ret/=tf.cast(tf.size(ret),tf.float32)
                return opts.mean+opts.scale*ret

            ########################################
            print('  creating tensorflow graph')

            with tf.name_scope('inputs'):
                x_ = tf.placeholder(tf.float32, [None,1])
                y_ = tf.placeholder(tf.float32, [None,1])

            with tf.name_scope('true'):
                if opts.target=='sin':
                    y_true = tf.sin(x_)
                else:
                    y_true = x_**2/10

            with tf.name_scope('model'):
                bias=1.0
                layer=0
                y = x_
                n0 = 1
                for n in opts.layers:
                    print('    layer'+str(layer)+' nodes: '+str(n))
                    with tf.name_scope('layer'+str(layer)):
                        w = tf.Variable(randomness([n0,n]))
                        b = tf.Variable(bias*tf.ones([1,n]))
                        y = activation(tf.matmul(y,w)+b)
                    n0 = n
                    if opts.activation=='crelu':
                        n0*=2
                    layer+=1

                with tf.name_scope('layer_final'):
                    w = tf.Variable(randomness([n0,1]))
                    #w = tf.ones([n0,1])
                    b = tf.Variable(bias)
                    y = tf.matmul(y,w)+b

            with tf.name_scope('eval'):
                loss = (y_-y)**2
                loss_true = (y_true-y)**2

                loss_ave = tf.reduce_sum(loss)/tf.cast(tf.size(x_),tf.float32)
                loss_true_ave = tf.reduce_sum(loss_true)/tf.cast(tf.size(x_),tf.float32)

            learningrate=10**opts.learningrate
            if opts.trainop=='sgd':
                train_op = tf.train.GradientDescentOptimizer(learningrate).minimize(loss)
            elif opts.trainop=='momentum':
                train_op = tf.train.MomentumOptimizer(learningrate, 0.9).minimize(loss)
            elif opts.trainop=='adam':
                train_op = tf.train.AdamOptimizer(learningrate).minimize(loss)

            sess = tf.Session()

            ########################################
            print('  generating data')

            X = np.random.uniform(xmin,xmax,size=[opts.numdp,1])
            Y_true = sess.run(y_true,feed_dict={x_:X})
            Y = Y_true + np.random.normal(scale=opts.variance,size=[opts.numdp,1])

            ax_data_xs2 = np.linspace(xmin,xmax,steps).reshape(steps,1)
            ax_data_ys2 = sess.run(y_true,feed_dict={x_:ax_data_xs2})

            ########################################
            print('  training')

            for epoch in range(0,opts.epochs+1):
                if epoch==0:
                    sess.run(tf.global_variables_initializer())
                    ax_data_ys = sess.run(y,feed_dict={x_:ax_data_xs})
                else:
                    rng_state = np.random.get_state()
                    np.random.shuffle(X)
                    np.random.set_state(rng_state)
                    np.random.shuffle(Y)
                    for batchstart in range(0,opts.numdp,opts.batchsize):
                        batchstop=batchstart+opts.batchsize
                        Xbatch=X[batchstart:batchstop]
                        Ybatch=Y[batchstart:batchstop]
                        sess.run(train_op,feed_dict={x_:Xbatch,y_:Ybatch})
                        ax_data_ys = sess.run(y,feed_dict={x_:ax_data_xs})

                if epoch%opts.plotevery==0:
                    titles_data.append('epoch = '+str(epoch))
                    res_ave,res_true=sess.run([loss_ave,loss_true_ave],feed_dict={x_:X,y_:Y})
                    [res_integral]=sess.run([loss_true_ave],feed_dict={x_:ax_data_xs2})
                    print('    epoch: %4d ; y_ave: %e ; y_true_ave: %e ; integral_ave %e'
                        %(epoch,res_ave,res_true,res_integral))

                    ax_data_line,=plotinfo[step]['ax_data'].plot(ax_data_xs,ax_data_ys,'-')
                    plotinfo[step]['ax_data_lines'].append(ax_data_line)

                    plotinfo[step]['ax_loss_xs'].append(epoch)

                    plotinfo[step]['ax_loss_ys_ave'].append(res_ave)
                    plotinfo[step]['ax_loss_line_ave'],=plotinfo[step]['ax_loss'].plot(plotinfo[step]['ax_loss_xs'],plotinfo[step]['ax_loss_ys_ave'],'-',color=(1,0,0))
                    plotinfo[step]['ax_loss_lines_ave'].append(plotinfo[step]['ax_loss_line_ave'])

                    plotinfo[step]['ax_loss_ys_true'].append(res_true)
                    plotinfo[step]['ax_loss_line_true'],=plotinfo[step]['ax_loss'].plot(plotinfo[step]['ax_loss_xs'],plotinfo[step]['ax_loss_ys_true'],'-',color=(0,1,0))
                    plotinfo[step]['ax_loss_lines_true'].append(plotinfo[step]['ax_loss_line_true'])

                    plotinfo[step]['ax_loss_ys_integral'].append(res_integral)
                    plotinfo[step]['ax_loss_line_integral'],=plotinfo[step]['ax_loss'].plot(plotinfo[step]['ax_loss_xs'],plotinfo[step]['ax_loss_ys_integral'],'-',color=(0,0,1))
                    plotinfo[step]['ax_loss_lines_integral'].append(plotinfo[step]['ax_loss_line_integral'])

            plotinfo[step]['ax_data'].set_xlim([xmin-xmargin,xmax+xmargin])

            if opts.zoomtarget:
                ymargin=0.5*(np.amax(Y)-np.amin(Y))/2
                plotinfo[step]['ax_data'].set_ylim([np.amin(Y)-ymargin,np.amax(Y)+ymargin])

            ltrue,=plotinfo[step]['ax_data'].plot(ax_data_xs2,ax_data_ys2,'-',color=(0,0,1))
            ldata,=plotinfo[step]['ax_data'].plot(X,Y,'.',color=(0,0,1))

except KeyboardInterrupt:
    print('>>>>>>>>>>>>>> KeyboardInterupt <<<<<<<<<<<<<<')

################################################################################
print('saving/displaying result')

class animate_data:
    @staticmethod
    def init():
        for step in range(0,args.steps+1):
            for ax_data_line in plotinfo[step]['ax_data_lines']:
                ax_data_line.set_visible(False)

            for ax_loss_line_ave in plotinfo[step]['ax_loss_lines_ave']:
                ax_loss_line_ave.set_visible(False)

            for ax_loss_line_true in plotinfo[step]['ax_loss_lines_true']:
                ax_loss_line_true.set_visible(False)

            for ax_loss_line_integral in plotinfo[step]['ax_loss_lines_integral']:
                ax_loss_line_integral.set_visible(False)

        return []

    @staticmethod
    def update(frame):
        plt.suptitle(titles_data[frame])
        for step in range(0,args.steps+1):
            plotinfo[step]['ax_data_lines'][frame].set_visible(True)
            plt.setp(plotinfo[step]['ax_data_lines'][frame],color=(0,0.5,0))

        gradientlen=5
        colorstep=0.5/gradientlen
        for j in range(1,gradientlen):
            if frame-j>=0:
                for step in range(0,args.steps+1):
                    plt.setp(plotinfo[step]['ax_data_lines'][frame-j],color=(j*colorstep,0.5+j*colorstep,j*colorstep))

        for step in range(0,args.steps+1):
            plotinfo[step]['ax_loss_lines_ave'][frame].set_visible(True)
            plotinfo[step]['ax_loss_lines_true'][frame].set_visible(True)
            plotinfo[step]['ax_loss_lines_integral'][frame].set_visible(True)

        return []

class animate_steps:
    @staticmethod 
    def init():
        for step in range(0,args.steps+1):
            animate_data.init()
            frames = range(0,len(plotinfo[0]['ax_data_lines']))
            for frame in frames:
                animate_data.update(frame)
            plotinfo[step]['ax_data'].set_visible(False)
            plotinfo[step]['ax_loss'].set_visible(False)
        return []

    @staticmethod 
    def update(frame):
        plt.suptitle(titles_step[frame])
        print(titles_step[frame])
        for step in range(0,args.steps+1):
            plotinfo[step]['ax_data'].set_visible(step==frame)
            plotinfo[step]['ax_loss'].set_visible(step==frame)
        return []

class animate:
    @staticmethod
    def init():
        print('  initializing animation')
        if args.steps==0:
            animate_data.init()
        else:
            animate_steps.init()

    @staticmethod
    def update(frame):
        print('  animating frame '+str(frame))
        if args.steps==0:
            animate_data.update(frame)
        else:
            animate_steps.update(frame)

if args.steps==0:
    frames=len(plotinfo[0]['ax_data_lines'])
    animate.init()
    ani = FuncAnimation(fig, animate.update, frames=frames)
else:
    animate.init()
    ani = FuncAnimation(fig, animate.update, frames=args.steps+1)

#basename=' '.join(sys.argv[1:])
basename='test'
fig.set_size_inches(6,3)
ani.save(basename+'.gif', dpi=96, writer='imagemagick')
#ani.save(basename+'.mp4', dpi=96, writer='ffmpeg')
#plt.savefig(basename+'.png',dpi=96)
#plt.show()
