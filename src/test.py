#!/usr/bin/env python

from __future__ import print_function

import copy
import itertools
import math
import os
import pkgutil
import sys

########################################
print('processing cmdline args')
import argparse
from interval import interval,Interval

parser=argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest='command')

modules={}
modules['data']={}
modules['graph']={}

import data
subparser_data = subparsers.add_parser('data')
subparsers_data = subparser_data.add_subparsers(dest='subcommand')
for loader, name, is_pkg in pkgutil.walk_packages(data.__path__):
    modules['data'][name]=loader.find_module(name).load_module(name)
    modules['data'][name].modify_parser(subparsers_data)

subparser_common = subparsers.add_parser('common')
subparser_common.add_argument('--steps',type=int,default=0)
#subparser_common.add_argument('--plotevery',type=int,default=10)
#subparser_common.add_argument('--zoomtarget',type=bool,default=True)
#subparser_common.add_argument('--zoomlevel',type=float,default=1.0)
subparser_common.add_argument('--seed_tf',type=int,default=0)
subparser_common.add_argument('--seed_node',type=int,default=0)
subparser_common.add_argument('--seed_np',type=int,default=0)

import graph
subparser_graph = subparsers.add_parser('graph')
subparsers_graph = subparser_graph.add_subparsers(dest='subcommand')
for loader, name, is_pkg in pkgutil.walk_packages(graph.__path__):
    modules['graph'][name]=loader.find_module(name).load_module(name)
    modules['graph'][name].modify_parser(subparsers_graph)

subparser_model = subparsers.add_parser('model')
parser_network = subparser_model.add_argument_group('network options')
parser_network.add_argument('--epochs',type=int,default=100)
parser_network.add_argument('--batchsize',type=interval(int),default=5)
parser_network.add_argument('--learningrate',type=interval(float),default=-3)
parser_network.add_argument('--trainop',choices=['sgd','momentum','adam'],default='adam')
parser_network.add_argument('--layers',type=interval(int),nargs='*',default=[interval(int)(100)])
parser_network.add_argument('--activation',choices=['relu','relu6','crelu','elu','softplus','softsign','sigmoid','tanh'],default='relu')
parser_weights = subparser_model.add_argument_group('weight initialization')
parser_weights.add_argument('--scale',type=interval(float),default=1.0)
parser_weights.add_argument('--mean',type=interval(float),default=0.0)
parser_weights.add_argument('--randomness',choices=['normal','uniform','gamma'],default='normal')
parser_weights.add_argument('--abs',type=bool,default=False)
parser_weights.add_argument('--normalize',type=bool,default=True)

####################

argvv = [list(group) for is_key, group in itertools.groupby(sys.argv[1:], lambda x: x=='--') if not is_key]

args={}
args['data'] = parser.parse_args(['data','synthetic'])
args['model'] = parser.parse_args(['model'])
args['common'] = parser.parse_args(['common'])
args['graph'] = []

for argv in argvv:
    if argv[0]=='graph':
        args['graph'].append(parser.parse_args(argv))
    else:
        args[argv[0]]=parser.parse_args(argv)

########################################
print('importing modules')

import tensorflow as tf
tf.Session()
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation

################################################################################
print('initialize graphs')

fig = plt.figure()
plt.tight_layout()
gs = gridspec.GridSpec(len(args['graph']),1)

graphs={}

titles_step=[]

################################################################################
try:
    for step in range(0,args['common'].steps+1):
        print('step '+str(step))

        ########################################
        print('  processing range args')

        opts={}
        title_step=[]
        for command in ['model','data','common']:
            param_names=filter(lambda x: x[0]!='_', dir(args[command]))
            for param_name in param_names:
                param=eval('args["'+command+'"].'+param_name)
                if isinstance(param,Interval):
                    res=param.start+step*(param.stop-param.start)/(args['common'].steps+1)
                    opts[param_name]=res
                    if not param.is_trivial:
                        title_step.append(param_name+' = '+str(res))
                elif type(param) is list:
                    ress=[]
                    all_trivial=True
                    for p in param:
                        if isinstance(p,Interval):
                            res=p.start+step*(p.stop-p.start)/(args['common'].steps+1)
                            ress.append(res)
                            if not p.is_trivial:
                                all_trivial=False
                    if not all_trivial:
                        title_step.append(param_name+' = '+str(ress))
                    opts[param_name] = ress
                else:
                    opts[param_name] = eval('args[command].'+param_name)
        titles_step.append(' ; '.join(title_step))

        tf.set_random_seed(opts['seed_tf'])
        random.seed(opts['seed_np'])
        np.random.seed(opts['seed_np'])

        data = modules['data']['synthetic'].Data(args['data'])

        ######################################## 
        print('  initializing graphs')
        graphs[step]=[]
        for i in range(0,len(args['graph'])):
            arg = args['graph'][i]
            g = modules['graph'][arg.subcommand].Graph(fig,gs[i],str(step),arg,opts)
            graphs[step].append(g)

        ########################################
        print('  setting tensorflow options')
        with tf.Graph().as_default():

            activation=eval('tf.nn.'+opts['activation'])

            def randomness(shape,seed):
                ret=None
                if opts['randomness']=='normal':
                    ret=tf.random_normal(shape,seed=seed+opts['seed_node'])
                if opts['randomness']=='uniform':
                    ret=tf.random_uniform(shape,minval=-1,maxval=1,seed=seed+opts['seed_node'])
                if opts['randomness']=='gamma':
                    alpha=1
                    beta=1
                    sign=tf.sign(tf.random_uniform(shape,minval=-1,maxval=1,seed=seed+opts['seed_node']))
                    ret=sign*tf.random_gamma(shape,alpha=alpha,beta=beta,seed=seed+opts['seed_node'])
                if opts['abs']:
                    ret=tf.abs(ret)
                if opts['normalize']:
                    ret/=tf.cast(tf.size(ret),tf.float32)
                return opts['mean']+opts['scale']*ret

            ########################################
            print('  creating tensorflow graph')

            with tf.name_scope('inputs'):
                x_ = tf.placeholder(tf.float32, [None,1])
                y_ = tf.placeholder(tf.float32, [None,1])

            with tf.name_scope('true'):
                #y_true = modules['data']['synthetic'].target_true(x_)
                y_true = data.target_true(x_)

            with tf.name_scope('model'):
                bias=1.0
                layer=0
                y = x_
                n0 = 1
                for n in opts['layers']:
                    print('    layer'+str(layer)+' nodes: '+str(n))
                    with tf.name_scope('layer'+str(layer)):
                        w = tf.Variable(randomness([n0,n],n),name='w')
                        b = tf.Variable(bias*tf.ones([1,n]),name='b')
                        y = activation(tf.matmul(y,w)+b,name='y')
                    n0 = n
                    if opts['activation']=='crelu':
                        n0*=2
                    layer+=1

                with tf.name_scope('layer_final'):
                    w = tf.Variable(randomness([n0,1],n+1),name='w')
                    #w = tf.ones([n0,1])
                    b = tf.Variable(bias,name='b')
                    y = tf.matmul(y,w)+b

            with tf.name_scope('eval'):
                loss = (y_-y)**2
                loss_true = (y_true-y)**2

                loss_ave = tf.reduce_sum(loss)/tf.cast(tf.size(x_),tf.float32)
                loss_true_ave = tf.reduce_sum(loss_true)/tf.cast(tf.size(x_),tf.float32)

            learningrate=10**opts['learningrate']
            if opts['trainop']=='sgd':
                train_op = tf.train.GradientDescentOptimizer(learningrate).minimize(loss)
            elif opts['trainop']=='momentum':
                train_op = tf.train.MomentumOptimizer(learningrate, 0.9).minimize(loss)
            elif opts['trainop']=='adam':
                train_op = tf.train.AdamOptimizer(learningrate).minimize(loss)

            sess = tf.Session()

            ########################################
            print('  generating data')

            xmin=-10
            xmax=10
            #steps=256
            xmargin=0.1*(xmax-xmin)/2
            #ax_data_xs = np.linspace(xmin-xmargin,xmax+xmargin,steps).reshape(steps,1)

            X,Y,Y_true = data.generate_data(opts)

            ########################################
            print('  training')

            for graph in graphs[step]:
                graph.init_step(dict(globals(),**locals()))

            for epoch in range(0,opts['epochs']+1):
                if epoch==0:
                    sess.run(tf.global_variables_initializer())
                    #ax_data_ys = sess.run(y,feed_dict={x_:ax_data_xs})
                else:
                    rng_state = np.random.get_state()
                    np.random.shuffle(X)
                    np.random.set_state(rng_state)
                    np.random.shuffle(Y)
                    for batchstart in range(0,opts['numdp'],opts['batchsize']):
                        batchstop=batchstart+opts['batchsize']
                        Xbatch=X[batchstart:batchstop]
                        Ybatch=Y[batchstart:batchstop]
                        sess.run(train_op,feed_dict={x_:Xbatch,y_:Ybatch})
                        #ax_data_ys = sess.run(y,feed_dict={x_:ax_data_xs})

                if epoch!=0:
                    print('\033[F',end='')
                print('  epoch: %d'%epoch)

                for graph in graphs[step]:
                    graph.record_epoch(dict(globals(),**locals()))

            for graph in graphs[step]:
                graph.finalize(dict(globals(),**locals()))

except KeyboardInterrupt:
    print('>>>>>>>>>>>>>> KeyboardInterupt <<<<<<<<<<<<<<')

################################################################################
print('saving/displaying result')

epochframes=0
for graph in graphs[0]:
    epochframes=max(epochframes,graph.get_num_frames())

if args['common'].steps==0:
    def update(frame):
        if frame!=0:
            print('\033[F',end='')
        print('  animating frame '+str(frame))
        for graph in graphs[0]:
            graph.update(frame)
    ani = FuncAnimation(fig, update, frames=epochframes, init_func=lambda:[])

else:
    for step in range(0,args['common'].steps+1):
        for graph in graphs[step]:
            for frame in range(0,epochframes):
                graph.update(frame)

    def update(frame):
        if frame!=0:
            print('\033[F',end='')
        print('  animating frame '+str(frame))
        plt.suptitle(titles_step[frame])
        print(titles_step[frame])
        for step in range(0,args['common'].steps+1):
            for graph in graphs[step]:
                graph.set_visible(step==frame)
    ani = FuncAnimation(fig, update, frames=args['common'].steps+1, init_func=lambda:[])

#basename=' '.join(sys.argv[1:])
basename='test'
fig.set_size_inches(6,3)
ani.save(basename+'.gif', dpi=96, writer='imagemagick')
#ani.save(basename+'.mp4', dpi=96, writer='ffmpeg')
