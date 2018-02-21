#!/usr/bin/env python

from __future__ import print_function

import collections
import copy
import itertools
import math
import os
import pkgutil
import sys
import sklearn
import tflearn
import time
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

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
    if not name[0] == '_':
        modules['model'][name]=loader.find_module(name).load_module(name)
        modules['model'][name].modify_parser(subparsers_model)

subparser_common = subparsers.add_parser('common')
subparser_common.add_argument('--partitions',type=int,default=0)
subparser_common.add_argument('--seed_tf',type=int,default=0)
subparser_common.add_argument('--seed_node',type=int,default=0)
subparser_common.add_argument('--seed_np',type=int,default=0)
subparser_common.add_argument('--verbose',action='store_true',default=False)
subparser_common.add_argument('--do_sklearn',action='store_true')
subparser_common.add_argument('--naive_mean',action='store_true')

subparser_log = subparser_common.add_argument_group('logging options')
subparser_log.add_argument('--log_dir',type=str,default='log')
subparser_log.add_argument('--tensorboard',action='store_true')
subparser_log.add_argument('--dirname_opts',type=str,default=[],nargs='*')
subparser_log.add_argument('--dump_data',action='store_true')

subparser_preprocess = subparser_common.add_argument_group('data preprocessing options')
subparser_preprocess.add_argument('--unit_norm',action='store_true')
subparser_preprocess.add_argument('--triangle',type=interval(int),default=0)
subparser_preprocess.add_argument('--label_corruption',type=interval(int),default=0)
subparser_preprocess.add_argument('--gaussian_X',type=interval(int),default=0)
subparser_preprocess.add_argument('--zero_Y',type=interval(int),default=0)
subparser_preprocess.add_argument('--max_Y',type=interval(int),default=0)
subparser_preprocess.add_argument('--pad_dim',type=interval(int),default=0)
subparser_preprocess.add_argument('--pad_dim_numbad',type=interval(int),default=0)

subparser_optimizer = subparser_common.add_argument_group('optimizer options')
subparser_optimizer.add_argument('--batch_size',type=interval(int),default=1)
subparser_optimizer.add_argument('--batch_size_test',type=interval(int),default=100)
subparser_optimizer.add_argument('--learning_rate',type=interval(float),default=-3)
subparser_optimizer.add_argument('--decay',choices=['inverse_time','natural_exp','piecewise_constant','polynomial','exponential','none','sqrt'],default='sqrt')
subparser_optimizer.add_argument('--decay_steps',type=interval(float),default=100000)
subparser_optimizer.add_argument('--optimizer',choices=['sgd','momentum','adam','adagrad','adagradda','adadelta','ftrl','psgd','padagrad','rmsprop'],default='sgd')

subparser_optimizer.add_argument('--early_stop_check',type=int,default=3)
subparser_optimizer.add_argument('--epochs',type=int,default=100)

subparser_robust = subparser_common.add_argument_group('robustness options')
subparser_robust.add_argument('--disable_robust',action='store_true')
subparser_robust.add_argument('--clip_method',choices=['batch','batch_naive','dp','dp_naive'],default='batch')
subparser_robust.add_argument('--clip_type',choices=['none','global','local'],default='none')
subparser_robust.add_argument('--clip_activation',choices=['soft','hard'],default='soft')
subparser_robust.add_argument('--clip_percentile',type=interval(float),default=99)
subparser_robust.add_argument('--window_size',type=interval(int),default=None)

####################

argvv = [list(group) for is_key, group in itertools.groupby(sys.argv[1:], lambda x: x=='--') if not is_key]

args={}
args['data'] = parser.parse_args(['data','synthetic'])
args['model'] = parser.parse_args(['model','glm'])
args['common'] = parser.parse_args(['common'])
args['graph'] = []

for argv in argvv:
    if argv[0]=='graph':
        args['graph'].append(parser.parse_args(argv))
    else:
        print('argv[0]=',argv[0])
        args[argv[0]]=parser.parse_args(argv)

data = modules['data'][args['data'].subcommand]
module_model = modules['model'][args['model'].subcommand]

########################################
print('importing modules')

import tensorflow as tf
tf.Session()
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

    ########################################
    print('  initializing graphs')
    graphs[partition]=[]
    for i in range(0,len(args['graph'])):
        arg = args['graph'][i]
        g = modules['graph'][arg.subcommand].Graph(fig,gs[i],str(partition),arg,opts)
        graphs[partition].append(g)

    ########################################
    print('  setting tensorflow options')
    with tf.Graph().as_default():

        tf.set_random_seed(opts['seed_tf'])
        random.seed(opts['seed_np'])
        np.random.seed(opts['seed_np'])

        global_step = tf.Variable(0, name='global_step',trainable=False)
        global_step_float=tf.cast(global_step,tf.float32)
        is_training = tf.placeholder(tf.bool)

        ########################################
        print('  creating tensorflow model')

        with tf.name_scope('inputs'):
            data.init(partitionargs['data'])

            def unit_norm(x,y,id):
                if opts['unit_norm']:
                    return (x/tf.norm(x),y,id)
                else:
                    return (x,y,id)

            gaussian_X_norms=tf.random_normal([opts['gaussian_X']]+data.dimX)
            def gaussian_X(x,y,id):
                x2=tf.cond(
                        opts['gaussian_X']>id,
                        lambda:gaussian_X_norms[id,...]*tf.norm(x),
                        lambda:x
                        )
                return (x2,y,id)

            def zero_Y(x,y,id):
                y2=tf.cond(
                        opts['zero_Y']>id,
                        lambda:0*y,
                        lambda:y
                        )
                return (x,y2,id)

            def max_Y(x,y,id):
                try:
                    y2=tf.cond(
                            opts['max_Y']>id,
                            lambda:data.train_Y_max,
                            lambda:y
                            )
                except AttributeError:
                    y2=y
                return (x,y2,id)

            def label_corruption(x,y,id):
                if opts['label_corruption']>0:
                    y2=tf.cond(
                            id>=opts['label_corruption'],
                            lambda: y,
                            lambda: tf.concat([y[1:],y[:1]],axis=0)
                            )
                else:
                    y2=y
                return (x,y2,id)

            def pad_dim(x,y,id):
                if opts['pad_dim']>0:
                    x2=tf.pad(
                            x,
                            [[0,opts['pad_dim']],[0,opts['pad_dim']],[0,0]],
                            constant_values=0.5,
                            )
                    return (x2,y,id)
                else:
                    return (x,y,id)

            def triangle(x,y,id):
                if opts['triangle']>0:
                    dimXnew=[data.dimX[0]+opts['pad_dim'],data.dimX[1]+opts['pad_dim']]
                    triangle=np.float32(1.414*np.tril(np.ones(dimXnew, dtype=int), -1))
                    triangle=triangle.reshape(dimXnew+[data.dimX[2]])
                    x2=tf.cond(
                            id<opts['triangle'],
                            lambda:triangle,
                            lambda:x
                            )
                    return (x2,y,id)
                else:
                    return (x,y,id)

            data.train = data.train.map(unit_norm)
            data.train = data.train.map(label_corruption)
            data.train = data.train.map(gaussian_X)
            data.train = data.train.map(zero_Y)
            data.train = data.train.map(max_Y)
            data.train = data.train.map(pad_dim)
            data.train = data.train.map(triangle)
            data.train = data.train.shuffle(opts['batch_size']*20,seed=0)
            data.train = data.train.batch(opts['batch_size'])

            data.valid = data.valid.map(unit_norm)
            data.valid = data.valid.map(pad_dim)
            data.valid = data.valid.batch(opts['batch_size_test'])

            data.test = data.test.map(unit_norm)
            data.test = data.test.map(pad_dim)
            data.test = data.test.batch(opts['batch_size_test'])

            iterator = tf.data.Iterator.from_structure(
                    data.train.output_types,
                    data.train.output_shapes
                    )
            x_,y_,z_ = iterator.get_next()
            y_argmax_=tf.argmax(y_,axis=1)

            if opts['pad_dim']>0:
                data.dimX=[
                        data.dimX[0]+opts['pad_dim'],
                        data.dimX[1]+opts['pad_dim'],
                        data.dimX[2]
                        ]
            print('    padded dimX=',x_.get_shape())

        y = module_model.inference(x_,data,opts,is_training)
        y_argmax = tf.argmax(y)
        loss,loss_per_dp = module_model.loss(partitionargs['model'],y_,y)

        if opts['tensorboard']:
            with tf.name_scope('batch/'):
                vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                for var in vars:
                    tf.summary.histogram(var.name,var)

        ########################################
        print('  creating tensorflow optimizer')

        # set learning rate

        learning_rate=10**opts['learning_rate']
        #decay_steps=opts['decay_steps']
        if data.train_numdp:
            decay_steps=data.train_numdp
        else:
            decay_steps=100000

        if opts['decay']=='exponential':
            learning_rate = tf.train.exponential_decay(learning_rate,global_step,decay_steps,0.96)
        elif opts['decay']=='natural_exp':
            learning_rate=tf.train.natural_exp_decay(learning_rate,global_step,decay_steps,0.96)
        elif opts['decay']=='inverse_time':
            learning_rate=tf.train.inverse_time_decay(learning_rate,global_step,decay_steps,0.5)
        elif opts['decay']=='polynomial':
            learning_rate=tf.train.polynomial_decay(learning_rate,global_step,decay_steps,learning_rate/100)
        elif opts['decay']=='sqrt':
            learning_rate=learning_rate/tf.sqrt(global_step_float+100)
        elif opts['decay']=='piecewise_constant':
            raise ValueError('piecewise_constant rate decay needs work')

        if opts['tensorboard']:
            tf.summary.scalar('learning_rate',learning_rate)

        # set optimizer

        if opts['optimizer']=='sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        elif opts['optimizer']=='momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
        elif opts['optimizer']=='adam':
            optimizer = tf.train.AdamOptimizer(learning_rate)
        elif opts['optimizer']=='adagrad':
            optimizer = tf.train.AdagradOptimizer(learning_rate)
        elif opts['optimizer']=='adagradda':
            optimizer = tf.train.AdagradDAOptimizer(learning_rate)
        elif opts['optimizer']=='adadelta':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate)
        elif opts['optimizer']=='ftrl':
            optimizer = tf.train.FtrlOptimizer(learning_rate)
        elif opts['optimizer']=='psgd':
            optimizer = tf.train.ProximalOptimizer(learning_rate)
        elif opts['optimizer']=='padagrad':
            optimizer = tf.train.ProximalAdagradOptimizer(learning_rate)
        elif opts['optimizer']=='rmsprop':
            optimizer = tf.train.RMSPropOptimizer(learning_rate)

        # set robust modifier
        print('  robustifying optimizer')

        if opts['disable_robust']:
            train_op=optimizer.minimize(loss)
            global_norm=0
            clip=0
            m_unbiased=0

        else:
            if not opts['window_size']:
                try:
                    opts['window_size']=data.train_numdp
                except:
                    opts['window_size']=10000

            import robust
            train_op,global_norm,clip,m_unbiased=robust.robust_minimize(
                    optimizer,
                    loss,
                    loss_per_dp,
                    global_step,
                    opts['batch_size'],
                    clip_method=opts['clip_method'],
                    clip_type=opts['clip_type'],
                    clip_activation=opts['clip_activation'],
                    clip_percentile=opts['clip_percentile'],
                    window_size=opts['window_size']
                    )

        if opts['tensorboard']:
            with tf.name_scope('batch/'):
                for grad,var in grads_and_vars:
                    tf.summary.histogram(var.name+'_grad',grad)
                for grad,var in grads_and_vars2:
                    tf.summary.histogram(var.name+'_grad_rob',grad)

        ########################################
        print('  creating tensorflow session')

        loss_values=[]
        loss_updates=[]
        for loss in tf.get_default_graph().get_collection(tf.GraphKeys.LOSSES):
            with tf.name_scope('batch/'):
                tf.summary.scalar(loss.name,loss,collections=['batch'])
            with tf.name_scope('epoch/'):
                value,update=tf.contrib.metrics.streaming_mean(loss)
                loss_values.append(value)
                loss_updates.append(update)
                tf.summary.scalar(loss.name,value,collections=['epoch'])
        loss_updates=tf.group(*loss_updates)

        summary_batch = tf.summary.merge_all(key='batch')
        #summary_batch = tf.summary.merge_all()
        summary_epoch = tf.summary.merge_all(key='epoch')

        def reset_summary():
            vars=tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES,scope='epoch/')
            sess.run(tf.variables_initializer(vars))

        with tf.Session().as_default() as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord) #,sess=sess)

            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())

            # create a log dir

            dirname_opts={}
            for opt in args['common'].dirname_opts:
                dirname_opts[opt]=opts[opt]
            dirname_opts_str=str(dirname_opts).translate(None,"{}: '")

            import hashlib
            import shutil
            optshash=hashlib.sha224(str(opts)).hexdigest()
            dirname = 'optshash='+optshash+'-'+dirname_opts_str
            log_dir = args['common'].log_dir+'/'+dirname
            shutil.rmtree(log_dir,ignore_errors=True)
            writer_train = tf.summary.FileWriter(log_dir+'/train',sess.graph)
            writer_test = tf.summary.FileWriter(log_dir+'/test',sess.graph)
            print('    log_dir = '+log_dir)

            # create symlinks to the log_dir

            symlink=args['common'].log_dir+'/recent'
            try:
                os.remove(symlink)
            except:
                pass
            try:
                os.symlink(dirname,symlink)
                print('    symlink = '+symlink)
            except:
                print('    failed to create symlink')

            # misc files in log_dir

            with open(log_dir+'/opts.txt','w',1) as f:
                f.write('opts = '+str(opts))

            file_batch=open(log_dir+'/batch.txt','w',1)
            file_epoch=open(log_dir+'/epoch.txt','w',1)
            file_results=open(log_dir+'/results.txt','w',1)

            if opts['dump_data']:
                import scipy.io
                scipy.io.savemat(
                        log_dir+'/data.mat',
                        {'train_Y':data.train_Y,
                         'mu':data.mu
                        })

            ########################################
            if opts['do_sklearn']:
                from sklearn import linear_model

                train_X=[]
                train_Y=[]
                sess.run(iterator.make_initializer(data.train))
                try:
                    while True:
                        X,Y=sess.run([x_,y_])
                        train_X.append(X)
                        train_Y.append(np.argmax(Y,axis=1))
                except tf.errors.OutOfRangeError:
                    train_X=np.concatenate(train_X,axis=0)
                    train_Y=np.concatenate(train_Y,axis=0)

                logreg = linear_model.LogisticRegression(C=1/opts['l2'])
                logreg.fit(train_X,train_Y)
                print('logreg score=',logreg.score(data.valid_X,np.argmax(data.valid_Y,axis=1)))

                ransac = linear_model.LogisticRegression(C=1/opts['l2'])
                ransac.fit(train_X,train_Y)
                print('ransac score=',ransac.score(data.valid_X,np.argmax(data.valid_Y,axis=1)))

            if opts['naive_mean']:
                file_results.write(' '.join(map(str,data.naive_accuracies))+' ')

            ########################################
            print('training')

            validation_scores=[]
            validation_scores_diff=float('inf')

            for epoch in range(0,opts['epochs']+1):

                epoch_start=time.clock()

                # train one epoch on training set
                if epoch>0:
                    sess.run(iterator.make_initializer(data.train),feed_dict={is_training:True})
                    try:
                        reset_summary()
                        while True:
                            tracker_ops=[global_step,y_argmax_,z_,global_norm,clip,m_unbiased]+loss_values
                            batch_start=time.clock()
                            loss_res,tracker_res,_,_=sess.run([loss,tracker_ops,loss_updates,train_op],feed_dict={is_training:True})
                            if opts['verbose']:
                                print('    step=%d, loss=%g, sec=%g           '%(global_step.eval(),loss_res,time.clock()-batch_start))
                                print('\033[F',end='')
                            nextline=' '.join(map(str,tracker_res))
                            nextline=filter(lambda x: x not in '[],', nextline)
                            file_batch.write(nextline+'\n')
                            if opts['tensorboard']:
                                writer_train.add_summary(summary, global_step.eval())
                    except tf.errors.OutOfRangeError:
                        summary=sess.run(summary_epoch)
                        writer_train.add_summary(summary,global_step.eval())

                # evaluate on validation set
                sess.run(iterator.make_initializer(data.valid),feed_dict={is_training:False})
                try:
                    reset_summary()
                    while True:
                        sess.run(loss_updates,feed_dict={is_training:False})
                except tf.errors.OutOfRangeError:
                    res,summary=sess.run([loss_values,summary_epoch],feed_dict={is_training:False})
                    if opts['tensorboard']:
                        writer_test.add_summary(summary,global_step.eval())

                    nextline=str(epoch)+' '+' '.join(map(str,res))
                    nextline=filter(lambda x: x not in '[],', nextline)
                    file_epoch.write(nextline+'\n')

                    validation_scores.append(res[0])
                    if len(validation_scores)>=opts['early_stop_check']:
                        n=len(validation_scores)-1
                        validation_scores_diff=validation_scores[n-opts['early_stop_check']]-validation_scores[n]
                    print('  epoch: %d    '%epoch,'early_stop:',validation_scores_diff,' -- ',res,'         ')

                    if validation_scores_diff<0:
                        break


            file_results.write(' '.join(map(str,res))+'\n')

            # evaluate on test set
            print('evaluating on test set')
            sess.run(iterator.make_initializer(data.test),feed_dict={is_training:False})
            try:
                reset_summary()
                while True:
                    sess.run(loss_updates,feed_dict={is_training:False})
            except tf.errors.OutOfRangeError:
                res=sess.run(loss_values,feed_dict={is_training:False})
                print('  res=',res)
            with open(log_dir+'/eval.txt','w',1) as f:
                f.write(str(res)+'\n')
