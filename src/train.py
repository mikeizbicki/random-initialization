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
subparser_common.add_argument('--verbose',action='store_true',default=False)

subparser_optimizer = subparser_common.add_argument_group('optimizer options')
subparser_optimizer.add_argument('--loss',choices=['xentropy','mse'],default='xentropy')
subparser_optimizer.add_argument('--robust',choices=['none','global','local'],default='none')
subparser_optimizer.add_argument('--median',action='store_true')
subparser_optimizer.add_argument('--m_alpha',type=interval(float),default=0.9)
subparser_optimizer.add_argument('--v_alpha',type=interval(float),default=0.9)
subparser_optimizer.add_argument('--num_stddevs',type=interval(float),default=1)
subparser_optimizer.add_argument('--epochs',type=int,default=100)
subparser_optimizer.add_argument('--batch_size',type=interval(int),default=5)
subparser_optimizer.add_argument('--batch_size_test',type=interval(int),default=100)
subparser_optimizer.add_argument('--learning_rate',type=interval(float),default=-3)
subparser_optimizer.add_argument('--decay',choices=['inverse_time','natural_exp','piecewise_constant','polynomial','exponential','none'],default='inverse_time')
subparser_optimizer.add_argument('--decay_steps',type=interval(float),default=100000)
subparser_optimizer.add_argument('--optimizer',choices=['sgd','momentum','adam','adagrad','adagradda','adadelta','ftrl','psgd','padagrad','rmsprop'],default='adam')

####################

argvv = [list(group) for is_key, group in itertools.groupby(sys.argv[1:], lambda x: x=='--') if not is_key]

args={}
args['data'] = parser.parse_args(['data','regression'])
args['model'] = parser.parse_args(['model','glm'])
args['common'] = parser.parse_args(['common'])
args['graph'] = []

for argv in argvv:
    if argv[0]=='graph':
        args['graph'].append(parser.parse_args(argv))
    else:
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
    print('  setting tensorflow options')
    with tf.Graph().as_default():

        ########################################
        print('  creating tensorflow model')

        def unit_norm(x,y,id):
            return (x/tf.norm(x),y,id)

        def label_corruption(x,y,id):
            #y_corrupt=tf.ones(y.get_shape())-y
            #y_corrupt=tf.concat([y[1:],y[:1]],axis=0)
            #return (x,tf.cond(id>=opts['label_corruption'],lambda: y,lambda: y_corrupt),id)
            y2=tf.cond(
                    id>=opts['label_corruption'],
                    lambda: y,
                    lambda: tf.concat([y[1:],y[:1]],axis=0)
                    )
            return (x,y2,id)

        with tf.name_scope('inputs'):
            data.init(partitionargs['data'])
            data.train = data.train.shuffle(opts['batch_size']*10000) #.batch(opts['batch_size'])
            #data.train = data.train.map(unit_norm)
            data.train = data.train.map(label_corruption)
            data.train = data.train.batch(opts['batch_size'])

            #data.test = data.test.map(unit_norm)
            data.test = data.test.batch(opts['batch_size_test'])

            iterator = tf.data.Iterator.from_structure(
                    data.train.output_types,
                    data.train.output_shapes
                    )
            x_,y_,z_ = iterator.get_next()
            y_argmax_=tf.argmax(y_,axis=1)

        global_step = tf.Variable(0, name='global_step',trainable=False)
        global_step_float=tf.cast(global_step,tf.float32)

        y = module_model.inference(x_,data,opts)
        y_argmax = tf.argmax(y)
        loss = module_model.loss(partitionargs['model'],y_,y)

        with tf.name_scope('batch/'):
            vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            for var in vars:
                tf.summary.histogram(var.name,var)

        ########################################
        print('  creating tensorflow optimizer')

        # set learning rate

        learning_rate=10**opts['learning_rate']
        decay_steps=opts['decay_steps']

        if opts['decay']=='exponential':
            learning_rate = tf.train.exponential_decay(learning_rate,global_step,decay_steps,0.96)
        elif opts['decay']=='natural_exp':
            learning_rate=tf.train.natural_exp_decay(learning_rate,global_step,decay_steps,0.96)
        elif opts['decay']=='inverse_time':
            learning_rate=tf.train.inverse_time_decay(learning_rate,global_step,decay_steps,0.5)
        elif opts['decay']=='polynomial':
            learning_rate=tf.train.polynomial_decay(learning_rate,global_step,decay_steps,learning_rate/100)
        elif opts['decay']=='piecewise_constant':
            raise ValueError('piecewise_constant rate decay needs work')

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
        with tf.name_scope('robust'):
            grads_and_vars=optimizer.compute_gradients(loss)
            gradients, variables = zip(*grads_and_vars)
            global_norm = tf.global_norm(gradients)

            total_parameters=0
            for _,var in grads_and_vars:
                variable_parameters = 1
                for dim in var.get_shape():
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            print('    total_parameters=',total_parameters)
            stddev_modifier=math.sqrt(2.0*math.log(float(total_parameters)))
            print('    stddev_modifier=',stddev_modifier)
            opts['num_stddevs']*=stddev_modifier
            print('    opts[num_stddevs]=',opts['num_stddevs'])

            if not opts['median']:
                m_alpha = 0.999 # opts['m_alpha']
                v_alpha = 0.999 # opts['v_alpha']
                m_init=0.0
                v_init=0.0
                epsilon=1e-9
                burn_in=10

                m = tf.Variable(m_init,trainable=False)
                v = tf.Variable(v_init,trainable=False)
                m_unbiased = m/(1.0-m_alpha**(1+global_step_float))
                v_unbiased = v/(1.0-v_alpha**(1+global_step_float))

                clip = tf.cond(global_step<burn_in,
                        lambda: global_norm,
                        lambda: m_unbiased+epsilon+(tf.sqrt(v_unbiased))*opts['num_stddevs']/math.sqrt(opts['batch_size'])/2
                        )

                #m2 = m_alpha*m + (1-m_alpha)*global_norm
                #v2 = v_alpha*v + (1-v_alpha)*global_norm**2
                m2 = m_alpha*m + (1.0-m_alpha)*tf.minimum(global_norm,clip)
                v2 = v_alpha*v + (1.0-v_alpha)*tf.minimum(global_norm,clip)**2
                m_update = tf.assign(m,m2)
                v_update = tf.assign(v,v2)
                update_clipper = tf.group(m_update,v_update)

            elif opts['median']:
                burn_in = 1000
                epsilon = 1e-9
                def get_median(v):
                    v = tf.reshape(v, [-1])
                    m = v.get_shape()[0]//2
                    return tf.nn.top_k(v, m).values[m-1]
                print('global_norm=',tf.reshape(global_norm,[1]).get_shape())
                ms = tf.Variable(tf.zeros([burn_in]),trainable=False)
                ms_update = tf.assign(ms[tf.mod(global_step,burn_in)],global_norm)
                vs = ms*ms
                m = tf.contrib.distributions.percentile(ms,75)
                ms_median = tf.contrib.distributions.percentile(ms,50)
                v = tf.contrib.distributions.percentile(
                        tf.abs(ms - ms_median),
                        50
                        )
                #v = tf.contrib.distributions.percentile(ms*ms,50)
                #m = tf.reduce_mean(ms)
                #v = tf.reduce_mean(ms*ms)
                m_unbiased = m
                v_unbiased = v
                clip = tf.cond(global_step<burn_in,
                        lambda: global_norm,
                        lambda: m_unbiased+epsilon+(v_unbiased)*opts['num_stddevs']/math.sqrt(opts['batch_size'])/2
                        )
                update_clipper = tf.group(ms_update)

            tf.summary.scalar('m',m)
            tf.summary.scalar('v',v)
            tf.summary.scalar('m_unbiased',m_unbiased)
            tf.summary.scalar('v_unbiased',v_unbiased)
            tf.summary.scalar('global_norm',global_norm)
            tf.summary.scalar('clip',clip)

            if opts['robust']=='none':
                gradients2=gradients

            elif opts['robust']=='global':
                clip = tf.cond(
                        clip>=global_norm,
                        lambda:clip,
                        lambda:tf.Print(clip,[global_norm,clip,m_unbiased,m,v_unbiased,v],'clipped'),
                        )
                gradients2, _ = tf.clip_by_global_norm(gradients, clip, use_norm=global_norm)
                #gradients2 = tf.cond(
                        #clip>global_norm,
                        #lambda:list(gradients),
                        #lambda:map (lambda x:x*1000.0,gradients)
                        #)

            elif opts['robust']=='local':
                gradients2=[]
                update_clipper=tf.group()

                for grad,var in grads_and_vars:
                    rawname=var.name.split(':')[0]
                    ones = np.ones(var.get_shape())
                    m = tf.Variable(m_init*ones,name=rawname,trainable=False,dtype=tf.float32)
                    v = tf.Variable(v_init*ones,name=rawname,trainable=False,dtype=tf.float32)
                    #m_unbiased = m/(1.0-m_alpha)
                    #v_unbiased = v/(1.0-v_alpha)
                    m_unbiased = m/(1.0-m_alpha**(1+global_step_float))
                    v_unbiased = v/(1.0-v_alpha**(1+global_step_float))
                    #v_unbiased = (v-(1.0-v_alpha)*v_init)/(1.0-v_alpha**(1+global_step_float))
                    clip = m_unbiased+tf.sqrt(v_unbiased)*opts['num_stddevs']
                    grad2_abs = tf.minimum(tf.abs(grad),clip)
                    m2 = m_alpha*m + (1.0-m_alpha)*tf.minimum(grad2_abs,clip)
                    v2 = v_alpha*v + (1.0-v_alpha)*tf.minimum(grad2_abs,clip)**2
                    grad2=tf.sign(grad)*grad2_abs
                    gradients2.append(grad2)
                    m_update=tf.assign(m,m2)
                    v_update=tf.assign(v,v2)
                    update_clipper=tf.group(update_clipper,m_update,v_update)

        # apply gradients

        grads_and_vars2=zip(gradients2,variables)
        grad_updates=optimizer.apply_gradients(
                grads_and_vars2,
                global_step=global_step)
        train_op = tf.group(grad_updates,update_clipper)

        with tf.name_scope('batch/'):
            for grad,var in grads_and_vars:
                tf.summary.histogram(var.name+'_grad',grad)
            for grad,var in grads_and_vars2:
                tf.summary.histogram(var.name+'_grad_rob',grad)

        ########################################
        print('  creating tensorflow session')

        values=[]
        updates=[]
        for graph in graphs[partition]:
            value,update=graph.add_summary()
            updates+=update
            values+=value
        updates=tf.group(*updates)

        #summary_batch = tf.summary.merge_all(key='batch')
        summary_batch = tf.summary.merge_all()
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

            opts_important={}
            for opt in ['batch_size','robust','num_stddevs','m_alpha','v_alpha','learning_rate','optimizer']: #,'noise','label_corruption']:
                opts_important[opt]=opts[opt]
            opts_important_str=str(opts_important).translate(None,"{}: '")

            import hashlib
            import shutil
            optshash=hashlib.sha224(str(opts)).hexdigest()
            dirname = 'optshash='+optshash+'-'+opts_important_str
            log_dir = args['common'].log_dir+'/'+dirname
            shutil.rmtree(log_dir,ignore_errors=True)
            writer_train = tf.summary.FileWriter(log_dir+'/train',sess.graph)
            writer_test = tf.summary.FileWriter(log_dir+'/test',sess.graph)
            print('    log_dir = '+log_dir)

            # create a symlink to the logdir

            #symlink=args['common'].log_dir+'/recent'
            #try:
                #os.remove(symlink)
            #except:
                #pass
            #os.symlink(dirname,symlink)
            #print('    symlink = '+symlink)

            with open(log_dir+'/opts.txt','w') as f:
                f.write('opts = '+str(opts))

            norms_file=open(log_dir+'/norms.txt','w')

            ########################################
            print('  training')

            for epoch in range(0,opts['epochs']+1):

                # train one epoch on training set
                if epoch>0:
                    sess.run(iterator.make_initializer(data.train))
                    try:
                        reset_summary()
                        while True:
                            loss_res,summary,y,z,norm,norm2,ave,stddev,_,_=sess.run([loss,summary_batch,y_argmax_,z_,global_norm,clip,m_unbiased,v_unbiased,updates,train_op])
                            writer_train.add_summary(summary, global_step.eval())
                            print('    step=%d, loss=%g, id=%s, label=%s, norm=%g/%g               '%(global_step.eval(),loss_res,str(z),str(y),norm,norm2))
                            if not opts['verbose']:
                                print('\033[F',end='')
                            norms_file.write('%d %d %g %g %g %g\n'%(global_step.eval(),z,norm,norm2,ave,stddev))
                    except tf.errors.OutOfRangeError: 
                        summary=sess.run(summary_epoch)
                        writer_train.add_summary(summary,global_step.eval())

                # evaluate on test set
                sess.run(iterator.make_initializer(data.test))
                try:
                    reset_summary()
                    while True:
                        sess.run(updates)
                except tf.errors.OutOfRangeError: 
                    res,summary=sess.run([values,summary_epoch])
                    writer_test.add_summary(summary,global_step.eval())

                    #if not opts['verbose'] and epoch!=0:
                        #print('\033[F',end='')
                    print('  epoch: %d    '%epoch,res,'         ')


