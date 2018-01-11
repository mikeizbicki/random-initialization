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
import data
subparser_data = subparsers.add_parser('data')
subparsers_data = subparser_data.add_subparsers(dest='subcommand')
for loader, name, is_pkg in pkgutil.walk_packages(data.__path__):
    modules['data'][name]=loader.find_module(name).load_module(name)
    modules['data'][name].modify_parser(subparsers_data)

modules['graph']={}
import graph
subparser_graph = subparsers.add_parser('graph')
subparsers_graph = subparser_graph.add_subparsers(dest='subcommand')
for loader, name, is_pkg in pkgutil.walk_packages(graph.__path__):
    modules['graph'][name]=loader.find_module(name).load_module(name)
    modules['graph'][name].modify_parser(subparsers_graph)

modules['model']={}
import model
subparser_model = subparsers.add_parser('model')
subparsers_model = subparser_model.add_subparsers(dest='subcommand')
for loader, name, is_pkg in pkgutil.walk_packages(model.__path__):
    modules['model'][name]=loader.find_module(name).load_module(name)
    modules['model'][name].modify_parser(subparsers_model)

subparser_common = subparsers.add_parser('common')
subparser_common.add_argument('--partitions',type=int,default=0)
subparser_common.add_argument('--seed_tf',type=int,default=0)
subparser_common.add_argument('--seed_node',type=int,default=0)
subparser_common.add_argument('--seed_np',type=int,default=0)
subparser_common.add_argument('--log_dir',type=str,default='log')

subparser_optimizer = subparser_common.add_argument_group('optimizer options')
subparser_optimizer.add_argument('--loss',choices=['xentropy','mse'],default='xentropy')
subparser_optimizer.add_argument('--nodecay',action='store_true')
subparser_optimizer.add_argument('--robust',choices=['none','global','local'])
subparser_optimizer.add_argument('--m_alpha',type=interval(float),default=0.5)
subparser_optimizer.add_argument('--v_alpha',type=interval(float),default=0.5)
subparser_optimizer.add_argument('--epochs',type=int,default=100)
subparser_optimizer.add_argument('--batchsize',type=interval(int),default=5)
subparser_optimizer.add_argument('--learningrate',type=interval(float),default=-3)
subparser_optimizer.add_argument('--optimizer',choices=['sgd','momentum','adam','adagrad','adagradda','adadelta','ftrl','psgd','padagrad','rmsprop'],default='adam')

####################

argvv = [list(group) for is_key, group in itertools.groupby(sys.argv[1:], lambda x: x=='--') if not is_key]

args={}
args['data'] = parser.parse_args(['data','regression'])
args['model'] = parser.parse_args(['model','custom'])
args['common'] = parser.parse_args(['common'])
args['graph'] = []

for argv in argvv:
    if argv[0]=='graph':
        args['graph'].append(parser.parse_args(argv))
    else:
        args[argv[0]]=parser.parse_args(argv)

module_data = modules['data'][args['data'].subcommand]
module_model = modules['model'][args['model'].subcommand]

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

titles_partition=[]

################################################################################
try:
    for partition in range(0,args['common'].partitions+1):
        print('partition '+str(partition))

        ########################################
        print('  processing range args')

        opts={}
        partitionargs={}
        title_partition=[]
        for command in ['model','data','common']:
            partitionargs[command]={}
            param_names=filter(lambda x: x[0]!='_', dir(args[command]))
            for param_name in param_names:
                param=eval('args["'+command+'"].'+param_name)
                if isinstance(param,Interval):
                    res=param.start+partition*(param.stop-param.start)/(args['common'].partitions+1)
                    partitionargs[command][param_name]=res
                    opts[param_name]=res
                    if not param.is_trivial:
                        title_partition.append(param_name+' = '+str(res))
                elif type(param) is list:
                    ress=[]
                    all_trivial=True
                    for p in param:
                        if isinstance(p,Interval):
                            res=p.start+partition*(p.stop-p.start)/(args['common'].partitions+1)
                            ress.append(res)
                            if not p.is_trivial:
                                all_trivial=False
                    if not all_trivial:
                        title_partition.append(param_name+' = '+str(ress))
                    partitionargs[command][param_name] = ress
                    opts[param_name] = ress
                else:
                    partitionargs[command][param_name] = eval('args[command].'+param_name)
                    opts[param_name] = eval('args[command].'+param_name)
        titles_partition.append(' ; '.join(title_partition))

        tf.set_random_seed(opts['seed_tf'])
        random.seed(opts['seed_np'])
        np.random.seed(opts['seed_np'])

        ########################################
        print('  initializing graphs')
        graphs[partition]=[]
        for i in range(0,len(args['graph'])):
            arg = args['graph'][i]
            g = modules['graph'][arg.subcommand].Graph(fig,gs[i],str(partition),arg,opts)
            graphs[partition].append(g)

        ########################################
        print('  generating data')

        xmin=-500
        xmax=500
        xmargin=0.1*(xmax-xmin)/2
        data = module_data.Data(partitionargs['data'])

        ########################################
        print('  setting tensorflow options')
        with tf.Graph().as_default():

            ########################################
            print('  creating tensorflow model')

            with tf.name_scope('inputs'):
                #x_ = tf.train.shuffle_batch(
                        #[tf.cast(tf.constant(data.train.X),tf.float32)],
                        #batch_size=opts['batchsize'],
                        #capacity=opts['numdp'],
                        #min_after_dequeue=0,
                        #num_threads=1,
                        #seed=opts['seed_tf'],
                        #enqueue_many=True)
                x_ = tf.placeholder(tf.float32, [None]+data.dimX)
                y_ = tf.placeholder(tf.float32, [None,data.dimY])

            y = module_model.inference(x_,data,opts)

            with tf.name_scope('losses'):
                xentropy = tf.reduce_mean(
                            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y),
                            name='xentropy')
                tf.add_to_collection(tf.GraphKeys.LOSSES,xentropy)

                mse = tf.losses.mean_squared_error(y_,y)

                results = tf.cast(tf.equal(tf.argmax(y,axis=1),tf.argmax(y_,axis=1)),tf.float32)
                accuracy = tf.reduce_mean(results,name='accuracy')
                tf.add_to_collection(tf.GraphKeys.LOSSES,accuracy)

                loss = eval(opts['loss'])

            ########################################
            print('  creating tensorflow optimizer')

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

            with tf.name_scope('robust'):
                grads_and_vars=optimizer.compute_gradients(loss)
                gradients, variables = zip(*grads_and_vars)

                m_alpha = opts['m_alpha']
                v_alpha = opts['v_alpha']
                m_init=0.0
                v_init=1.0

                if opts['robust']=='none':
                    gradients2=gradients
                    update_clipper = tf.group()

                elif opts['robust']=='global':
                    global_norm = tf.global_norm(gradients)
                    m = tf.Variable(m_init,trainable=False)
                    v = tf.Variable(v_init,trainable=False)
                    #global_norm = tf.Print(global_norm,[m,v,global_norm])
                    m_unbiased = m/(1.0-m_alpha)
                    v_unbiased = v/(1.0-v_alpha)
                    clip = m_unbiased+tf.sqrt(v_unbiased) #+1e-6
                    #m2 = m_alpha*m + (1-m_alpha)*global_norm
                    #v2 = v_alpha*v + (1-v_alpha)*global_norm**2
                    m2 = m_alpha*m + (1.0-m_alpha)*tf.minimum(global_norm,clip)
                    v2 = v_alpha*v + (1.0-v_alpha)*tf.minimum(global_norm,clip)**2
                    gradients2, _ = tf.clip_by_global_norm(gradients, clip, use_norm=global_norm)
                    m_update = tf.assign(m,m2)
                    v_update = tf.assign(v,v2)
                    update_clipper = tf.group(m_update,v_update)

                elif opts['robust']=='local':
                    gradients2=[]
                    update_clipper=tf.group()
                    for grad,var in grads_and_vars:
                        rawname=var.name.split(':')[0]
                        ones = np.ones(var.get_shape())
                        m = tf.Variable(m_init*ones,name=rawname,trainable=False,dtype=tf.float32)
                        v = tf.Variable(v_init*ones,name=rawname,trainable=False,dtype=tf.float32)
                        m_unbiased = m/(1.0-m_alpha)
                        v_unbiased = v/(1.0-v_alpha)
                        clip = m_unbiased+tf.sqrt(v_unbiased) #*50+1e6
                        grad2_abs = tf.minimum(tf.abs(grad),clip)
                        m2 = m_alpha*m + (1.0-m_alpha)*tf.minimum(grad2_abs,clip)
                        v2 = v_alpha*v + (1.0-v_alpha)*tf.minimum(grad2_abs,clip)**2
                        grad2=tf.sign(grad)*grad2_abs
                        gradients2.append(grad2)
                        m_update=tf.assign(m,m2)
                        v_update=tf.assign(v,v2)
                        update_clipper=tf.group(update_clipper,m_update,v_update)
                        #print('var=',var.name)

            grad_updates=optimizer.apply_gradients(zip(gradients2,variables))
            train_op = tf.group(grad_updates,update_clipper)

            ########################################
            print('  creating tensorflow session')

            for graph in graphs[partition]:
                graph.add_summary()

            merged = tf.summary.merge_all()
            sess = tf.Session()

            import hashlib
            import shutil
            optshash=hashlib.sha224(str(opts)).hexdigest()
            dirname = 'optshash='+optshash
            log_dir = args['common'].log_dir+'/'+dirname
            shutil.rmtree(log_dir,ignore_errors=True)
            writer_train = tf.summary.FileWriter(log_dir+'/train',sess.graph)
            writer_test = tf.summary.FileWriter(log_dir+'/test',sess.graph)
            print('    log_dir = '+log_dir)

            symlink=args['common'].log_dir+'/recent'
            try:
                os.remove(symlink)
            except:
                pass
            os.symlink(dirname,symlink)
            print('    symlink = '+symlink)

            with open(log_dir+'/opts.txt','w') as f:
                f.write('opts = '+str(opts))

            ########################################
            print('  training')

            step=0
            for graph in graphs[partition]:
                graph.init_step(dict(globals(),**locals()))

            for epoch in range(0,opts['epochs']+1):
                if epoch==0:
                    sess.run(tf.global_variables_initializer())

                else:
                    rng_state = np.random.get_state()
                    np.random.shuffle(data.train.X)
                    np.random.set_state(rng_state)
                    np.random.shuffle(data.train.Y)
                    for batchstart in range(0,data.train.numdp,opts['batchsize']):
                        if batchstart!=0:
                            print('\033[F',end='')
                        print('    step=',step,'    ')
                        step+=1
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

                        partitionsize=learningrate
                        if not opts['nodecay']:
                            partitionsize/=math.sqrt(step)
                            #partitionsize/=math.sqrt(epoch)

                        feed_dict={x_:Xbatch,y_:Ybatch,alpha_:partitionsize}
                        summary,_=sess.run([merged,train_op],feed_dict=feed_dict)

                        writer_train.add_summary(summary, step)

                if epoch!=0:
                    print('\033[F',end='')
                    print('\033[F',end='')
                print('  epoch: %d     '%epoch)

                feed_dict={x_:data.test.X,y_:data.test.Y}
                summary=sess.run(merged,feed_dict=feed_dict)
                writer_test.add_summary(summary, step)

                for graph in graphs[partition]:
                    graph.record_epoch(dict(globals(),**locals()))

            for graph in graphs[partition]:
                graph.finalize(dict(globals(),**locals()))

except KeyboardInterrupt:
    print('>>>>>>>>>>>>>> KeyboardInterupt <<<<<<<<<<<<<<')

################################################################################
print('visualizing          ')

epochframes=0
for graph in graphs[0]:
    epochframes=max(1,epochframes,graph.get_num_frames())

if args['common'].partitions==0:
    def update(frame):
        if frame!=0:
            print('\033[F',end='')
        print('  animating frame '+str(frame))
        for graph in graphs[0]:
            graph.update(frame)
    ani = FuncAnimation(fig, update, frames=epochframes, init_func=lambda:[])

else:
    print('  rendering partition frames')
    for partition in range(0,args['common'].partitions+1):
        print('    partition: ',partition)
        for graph in graphs[partition]:
            for frame in range(0,epochframes):
                graph.update(frame)

    print('  animating')
    def update(frame):
        if frame!=0:
            print('\033[F',end='')
        print('  animating frame '+str(frame))
        plt.suptitle(titles_partition[frame])
        print(titles_partition[frame])
        for partition in range(0,args['common'].partitions+1):
            for graph in graphs[partition]:
                graph.set_visible(partition==frame)
    ani = FuncAnimation(fig, update, frames=args['common'].partitions+1, init_func=lambda:[])

#basename=' '.join(sys.argv[1:])
basename='test'
fig.set_size_inches(6,2*len(args['graph']))
ani.save(basename+'.gif', dpi=96, writer='imagemagick')
#ani.save(basename+'.mp4', dpi=96, writer='ffmpeg')
