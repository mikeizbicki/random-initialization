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
parser_network.add_argument('--nodecay',action='store_true')
parser_network.add_argument('--robust',action='store_true')
parser_network.add_argument('--loss',choices=['xentropy','mse'],default='xentropy')
parser_network.add_argument('--optimizer',choices=['sgd','momentum','adam','adagrad','adagradda','adadelta','ftrl','psgd','padagrad','rmsprop'],default='adam')
parser_network.add_argument('--layers',type=interval(int),nargs='*',default=[interval(int)(100)])
parser_network.add_argument('--activation',choices=['none','relu','relu6','crelu','elu','softplus','softsign','sigmoid','tanh','logistic'],default='relu')
parser_weights = subparser_model.add_argument_group('weight initialization')
parser_weights.add_argument('--scale',type=interval(float),default=1.0)
parser_weights.add_argument('--mean',type=interval(float),default=0.0)
parser_weights.add_argument('--randomness',choices=['normal','uniform','laplace'],default='normal')
parser_weights.add_argument('--abs',type=bool,default=False)
parser_weights.add_argument('--normalize',type=str,default='False')

####################

argvv = [list(group) for is_key, group in itertools.groupby(sys.argv[1:], lambda x: x=='--') if not is_key]

args={}
args['data'] = parser.parse_args(['data','regression'])
args['model'] = parser.parse_args(['model'])
args['common'] = parser.parse_args(['common'])
args['graph'] = []

for argv in argvv:
    if argv[0]=='graph':
        args['graph'].append(parser.parse_args(argv))
    else:
        args[argv[0]]=parser.parse_args(argv)

datamodule = modules['data'][args['data'].subcommand]

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
        stepargs={}
        title_step=[]
        for command in ['model','data','common']:
            stepargs[command]={}
            param_names=filter(lambda x: x[0]!='_', dir(args[command]))
            for param_name in param_names:
                param=eval('args["'+command+'"].'+param_name)
                if isinstance(param,Interval):
                    res=param.start+step*(param.stop-param.start)/(args['common'].steps+1)
                    stepargs[command][param_name]=res
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
                    stepargs[command][param_name] = ress
                    opts[param_name] = ress
                else:
                    stepargs[command][param_name] = eval('args[command].'+param_name)
                    opts[param_name] = eval('args[command].'+param_name)
        titles_step.append(' ; '.join(title_step))

        tf.set_random_seed(opts['seed_tf'])
        random.seed(opts['seed_np'])
        np.random.seed(opts['seed_np'])

        ######################################## 
        print('  initializing graphs')
        graphs[step]=[]
        for i in range(0,len(args['graph'])):
            arg = args['graph'][i]
            g = modules['graph'][arg.subcommand].Graph(fig,gs[i],str(step),arg,opts)
            graphs[step].append(g)

        ########################################
        print('  generating data')

        xmin=-500
        xmax=500
        xmargin=0.1*(xmax-xmin)/2
        data = datamodule.Data(stepargs['data'])

        ########################################
        print('  setting tensorflow options')
        with tf.Graph().as_default():

            #tf.nn.logistic = lambda x: tf.log(1+tf.exp(-x))
            tf.nn.logistic = lambda x: tf.log(1+tf.exp(-tf.maximum(-88.0,x)))

            if opts['activation']=='none':
                activation=tf.identity
            else:
                activation=eval('tf.nn.'+opts['activation'])

            def randomness(size,seed):
                from stable_random import stable_random
                r=stable_random(size,opts['seed_np']+seed,dist=opts['randomness']).astype(np.float32)
                if opts['normalize']=='True':
                    r/=np.amax(np.abs(r))
                if opts['abs']:
                    r=np.abs(r)
                return opts['mean']+opts['scale']*r

            ########################################
            print('  creating tensorflow graph')

            with tf.name_scope('inputs'):
                #x_ = tf.train.shuffle_batch(
                        #[tf.cast(tf.constant(data.train.X),tf.float32)],
                        #batch_size=opts['batchsize'],
                        #capacity=opts['numdp'],
                        #min_after_dequeue=0, 
                        #num_threads=1, 
                        #seed=opts['seed_tf'], 
                        #enqueue_many=True)
                x_ = tf.placeholder(tf.float32, [None,opts['dimX']])
                y_ = tf.placeholder(tf.float32, [None,opts['dimY']])

            with tf.name_scope('model'):
                bias=1.0
                layer=0
                y = x_
                n0 = opts['dimX']
                for n in opts['layers']:
                    print('    layer'+str(layer)+' nodes: '+str(n))
                    with tf.name_scope('layer'+str(layer)):
                        w = tf.Variable(randomness([n0,n],layer),name='w')
                        b = tf.Variable(randomness([1,n],layer+1),name='b')
                        y = activation(tf.matmul(y,w)+b)
                    n0 = n
                    if opts['activation']=='crelu':
                        n0*=2
                    layer+=1

                with tf.name_scope('layer_final'):
                    w = tf.Variable(randomness([n0,opts['dimY']],layer+1),name='w')
                    b = tf.Variable(bias,name='b')
                    y = tf.matmul(y,w)+b

            with tf.name_scope('eval'):
                if opts['loss']=='xentropy':
                    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
                elif opts['loss']=='mse':
                    loss = tf.losses.mean_squared_error(y_,y)
                #loss_ave = loss

            alpha_=tf.placeholder(tf.float32, [])
            learningrate=10**opts['learningrate']
            if opts['optimizer']=='sgd':
                optimizer = tf.train.GradientDescentOptimizer(alpha_)
            elif opts['optimizer']=='momentum':
                optimizer = tf.train.MomentumOptimizer(alpha_, 0.9)
            elif opts['optimizer']=='adam':
                optimizer = tf.train.AdamOptimizer(alpha_)
            elif opts['optimizer']=='adagrad':
                optimizer = tf.train.AdagradOptimizer(alpha_)
            elif opts['optimizer']=='adagradda':
                optimizer = tf.train.AdagradDAOptimizer(alpha_)
            elif opts['optimizer']=='adadelta':
                optimizer = tf.train.AdadeltaOptimizer(alpha_)
            elif opts['optimizer']=='ftrl':
                optimizer = tf.train.FtrlOptimizer(alpha_)
            elif opts['optimizer']=='psgd':
                optimizer = tf.train.ProximalOptimizer(alpha_)
            elif opts['optimizer']=='padagrad':
                optimizer = tf.train.ProximalAdagradOptimizer(alpha_)
            elif opts['optimizer']=='rmsprop':
                optimizer = tf.train.RMSPropOptimizer(alpha_)

            grads_and_vars=optimizer.compute_gradients(loss)
            gradients, variables = zip(*optimizer.compute_gradients(loss))
            update_clipper = tf.group()
            if opts['robust']:
                with tf.name_scope('clipper'):
                    ave_alpha = 0.9
                    var_alpha = 0.9
                    global_norm = tf.global_norm(gradients)
                    ave = tf.Variable(0.0,trainable=False)
                    var = tf.Variable(1.0,trainable=False)
                    #global_norm = tf.Print(global_norm,[ave,var,global_norm])
                    ave_unbiased = ave/(1-ave_alpha)
                    var_unbiased = var/(1-var_alpha)
                    clip = ave_unbiased+tf.sqrt(var_unbiased) #+1e-6
                    #ave2 = ave_alpha*ave + (1-ave_alpha)*global_norm 
                    #var2 = var_alpha*var + (1-var_alpha)*global_norm**2
                    ave2 = ave_alpha*ave + (1-ave_alpha)*tf.minimum(global_norm,clip)
                    var2 = var_alpha*var + (1-var_alpha)*tf.minimum(global_norm,clip)**2
                    gradients, _ = tf.clip_by_global_norm(gradients, clip, use_norm=global_norm)
                    ave_update = tf.assign(ave,ave2)
                    var_update = tf.assign(var,var2)
                    update_clipper = tf.group(ave_update,var_update)
            grad_updates=optimizer.apply_gradients(zip(gradients,variables))
            train_op = tf.group(grad_updates,update_clipper)

            sess = tf.Session()

            ########################################
            print('  training')

            for graph in graphs[step]:
                graph.init_step(dict(globals(),**locals()))

            for epoch in range(0,opts['epochs']+1):
                if epoch==0:
                    sess.run(tf.global_variables_initializer())

                else:
                    rng_state = np.random.get_state()
                    np.random.shuffle(data.train.X)
                    np.random.set_state(rng_state)
                    np.random.shuffle(data.train.Y)
                    for batchstart in range(0,opts['numdp'],opts['batchsize']):
                        batchstop=batchstart+opts['batchsize']
                        Xbatch=data.train.X[batchstart:batchstop]
                        Ybatch=data.train.Y[batchstart:batchstop]

                        if False:
                            print('-----')
                            print('batchstart=',batchstart,'; batchstop=',batchstop)
                            print('Xbatch=',Xbatch)
                            #print('Ybatch=',Ybatch)
                            feed_dict={x_:data.train.X[0:1],y_:data.train.Y[0:1]}
                            for gv in grads_and_vars:
                                val=sess.run(gv[0],feed_dict=feed_dict)
                                print('grad:',gv[1],'=',val)
                            print('loss=',sess.run(loss,feed_dict=feed_dict))
                            print('w=',sess.run(w))
                            print('b=',sess.run(b))
                            sys.exit(1)

                        stepsize=learningrate
                        if not opts['nodecay']:
                            stepsize/=math.sqrt(epoch)

                        sess.run(train_op,feed_dict={x_:Xbatch,y_:Ybatch,alpha_:stepsize})

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
print('visualizing')

epochframes=0
for graph in graphs[0]:
    epochframes=max(1,epochframes,graph.get_num_frames())

if args['common'].steps==0:
    def update(frame):
        if frame!=0:
            print('\033[F',end='')
        print('  animating frame '+str(frame))
        for graph in graphs[0]:
            graph.update(frame)
    ani = FuncAnimation(fig, update, frames=epochframes, init_func=lambda:[])

else:
    print('  rendering step frames')
    for step in range(0,args['common'].steps+1):
        print('    step: ',step)
        for graph in graphs[step]:
            for frame in range(0,epochframes):
                graph.update(frame)

    print('  animating')
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
fig.set_size_inches(6,2*len(args['graph']))
ani.save(basename+'.gif', dpi=96, writer='imagemagick')
#ani.save(basename+'.mp4', dpi=96, writer='ffmpeg')
