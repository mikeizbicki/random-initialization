from __future__ import print_function

########################################
#print('importing modules')

import numpy as np
import random
import tensorflow as tf
import time

################################################################################

def modify_parser(subparsers):
    import argparse
    from interval import interval,Interval

    subparser = subparsers.add_parser('local')

    subparser_train = subparser.add_argument_group('train')
    subparser_train.add_argument('--seed_tf',type=int,default=0)
    subparser_train.add_argument('--seed_node',type=int,default=0)
    subparser_train.add_argument('--seed_np',type=int,default=0)
    subparser_train.add_argument('--verbose',action='store_true',default=False)
    subparser_train.add_argument('--savebatch',action='store_true',default=False)
    subparser_train.add_argument('--sklearn_classification',action='store_true')
    subparser_train.add_argument('--sklearn_regression',action='store_true')
    subparser_train.add_argument('--naive_mean',action='store_true')
    subparser_train.add_argument('--restore_from',type=str,default=None)

    subparser_log = subparser.add_argument_group('logging options')
    subparser_log.add_argument('--log_dir',type=str,default='log')
    subparser_log.add_argument('--delete_log',action='store_true')
    subparser_log.add_argument('--tensorboard',action='store_true')
    subparser_log.add_argument('--dirname_opts',type=str,default=[],nargs='*')
    subparser_log.add_argument('--dump_data',action='store_true')

    subparser_optimizer = subparser.add_argument_group('optimizer options')
    subparser_optimizer.add_argument('--batch_size',type=interval(int),default=1)
    subparser_optimizer.add_argument('--batch_size_valid',type=interval(int),default=100)
    subparser_optimizer.add_argument('--learning_rate',type=interval(float),default=-3)
    subparser_optimizer.add_argument('--weight_decay',type=interval(float),default=0)
    subparser_optimizer.add_argument('--decay',choices=['inverse_time','natural_exp','piecewise_constant','polynomial','exponential','none','sqrt'],default='none')
    subparser_optimizer.add_argument('--decay_steps',type=interval(float),default=100000)
    subparser_optimizer.add_argument('--optimizer',choices=['sgd','momentum','adam','adagrad','adagradda','adadelta','ftrl','psgd','padagrad','rmsprop'],default='sgd')
    subparser_optimizer.add_argument('--shuffle_all',action='store_true')

    subparser_optimizer.add_argument('--early_stop_check',type=int,default=10000)
    subparser_optimizer.add_argument('--epochs',type=int,default=100)

    subparser_robust = subparser.add_argument_group('robustness options')
    subparser_robust.add_argument('--robust_log',action='store_true')
    subparser_robust.add_argument('--disable_robust',action='store_true')
    subparser_robust.add_argument('--clip_method',choices=['batch','batch_naive','dp','dp_naive'],default='dp')
    subparser_robust.add_argument('--clip_type',choices=['none','global','local'],default='none')
    subparser_robust.add_argument('--clip_function',choices=['soft','hard'],default='soft')
    subparser_robust.add_argument('--clip_percentile',type=interval(float),default=99)
    subparser_robust.add_argument('--clip_threshold',type=interval(float),default=0)
    subparser_robust.add_argument('--disable_clip_perclass',action='store_true')
    subparser_robust.add_argument('--window_size',type=interval(int),default=None)

################################################################################

def train_with_hyperparams(model,data,partitionargs):

    ########################################
    print('  setup logging')

    dirname_opts={}
    dirname_opts_str=""
    for opt in partitionargs['train']['dirname_opts']:
        opt_split=opt.split('/')
        #dirname_opts[opt]=partitionargs[opt_split[0]][opt_split[1]]
        #dirname_opts[opt]=partitionargs['train'][opt]
        dirname_opts_str+='_'+opt_split[1]+'='+str(partitionargs[opt_split[0]][opt_split[1]])
    #dirname_opts_str=str(dirname_opts).translate(None,"{}: '")

    import hashlib
    import os
    import shutil
    optshash=hashlib.sha224(str(partitionargs)).hexdigest()
    dirname = 'optshash='+optshash+'-'+dirname_opts_str
    log_dir = partitionargs['train']['log_dir']+'/'+dirname
    log_dir_current=log_dir+'/'+'current'
    log_dir_best=log_dir+'/'+'best'

    # create directory

    if partitionargs['train']['delete_log']:
        shutil.rmtree(log_dir,ignore_errors=True)

    try:
        os.makedirs(log_dir)
    except OSError as e:
        if e.errno == os.errno.EEXIST and os.path.isdir(log_dir):
            pass
        else:
            raise

    print('    log_dir = '+log_dir)

    # create symlinks to the log_dir

    symlink=partitionargs['train']['log_dir']+'/recent'
    try:
        os.remove(symlink)
    except:
        pass
    try:
        os.symlink(dirname,symlink)
        print('    symlink = '+symlink)
    except Exception as e:
        print('    failed to create symlink ')
        print('    dirname: ',dirname)
        print('    symlink: ',symlink)
        print('    exception: ',e)

    ########################################
    print('  creating tensorflow model')

    tf.set_random_seed(partitionargs['train']['seed_tf'])
    random.seed(partitionargs['train']['seed_np'])
    np.random.seed(partitionargs['train']['seed_np'])

    epoch = tf.Variable(0, name='epoch',trainable=False)
    epoch_update = tf.assign(epoch,epoch+1)

    global_step = tf.Variable(0, name='global_step',trainable=False)
    global_step_float=tf.cast(global_step,tf.float32)
    is_training = tf.placeholder(tf.bool)

    shuffle_size=partitionargs['train']['batch_size']*20
    if partitionargs['train']['shuffle_all']:
        shuffle_size=data.train_numdp
    data.train = data.train.shuffle(shuffle_size,seed=0)
    data.train = data.train.batch(partitionargs['train']['batch_size'])
    data.valid = data.valid.batch(partitionargs['train']['batch_size_valid'])
    data.test = data.test.batch(partitionargs['train']['batch_size_valid'])

    iterator = tf.data.Iterator.from_structure(
            data.train.output_types,
            data.train.output_shapes
            )
    x_,y_,z_,mod_ = iterator.get_next()
    y_argmax_=tf.argmax(y_,axis=1)

    init_train=iterator.make_initializer(data.train),
    init_valid=iterator.make_initializer(data.valid),
    init_test=iterator.make_initializer(data.test),

    y = model.inference(x_,data,partitionargs['model'],is_training)
    y_argmax = tf.argmax(y)
    loss,loss_per_dp = model.loss(partitionargs['model'],y_,y)

    with tf.name_scope('weight_decay'):
        vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        weight=0
        for var in vars:
            weight+=tf.reduce_sum(var*var)
        weight*=partitionargs['train']['weight_decay']
        loss+=weight
        loss_per_dp+=weight
        #loss_per_dp=tf.map_fn(lambda l: l+weight,loss_per_dp)

    #if partitionargs['train']['tensorboard']:
        #with tf.name_scope('batch/'):
            #vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            #for var in vars:
                #tf.summary.histogram(var.name,var)

    ########################################
    print('  creating tensorflow optimizer')

    # set learning rate

    learning_rate=10**partitionargs['train']['learning_rate']
    try:
        decay_steps=data.train_numdp
    except:
        decay_steps=100000

    if partitionargs['train']['decay']=='exponential':
        learning_rate = tf.train.exponential_decay(learning_rate,global_step,decay_steps,0.96)
    elif partitionargs['train']['decay']=='natural_exp':
        learning_rate=tf.train.natural_exp_decay(learning_rate,global_step,decay_steps,0.96)
    elif partitionargs['train']['decay']=='inverse_time':
        learning_rate=tf.train.inverse_time_decay(learning_rate,global_step,decay_steps,0.5)
    elif partitionargs['train']['decay']=='polynomial':
        learning_rate=tf.train.polynomial_decay(learning_rate,global_step,decay_steps,learning_rate/100)
    elif partitionargs['train']['decay']=='sqrt':
        learning_rate=learning_rate/tf.sqrt(global_step_float+100)
    elif partitionargs['train']['decay']=='piecewise_constant':
        raise ValueError('piecewise_constant rate decay needs work')

    if partitionargs['train']['tensorboard']:
        tf.summary.scalar('learning_rate',learning_rate)

    # set optimizer

    if partitionargs['train']['optimizer']=='sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    elif partitionargs['train']['optimizer']=='momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    elif partitionargs['train']['optimizer']=='adam':
        optimizer = tf.train.AdamOptimizer(learning_rate)
    elif partitionargs['train']['optimizer']=='adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate)
    elif partitionargs['train']['optimizer']=='adagradda':
        optimizer = tf.train.AdagradDAOptimizer(learning_rate)
    elif partitionargs['train']['optimizer']=='adadelta':
        optimizer = tf.train.AdadeltaOptimizer(learning_rate)
    elif partitionargs['train']['optimizer']=='ftrl':
        optimizer = tf.train.FtrlOptimizer(learning_rate)
    elif partitionargs['train']['optimizer']=='psgd':
        optimizer = tf.train.ProximalOptimizer(learning_rate)
    elif partitionargs['train']['optimizer']=='padagrad':
        optimizer = tf.train.ProximalAdagradOptimizer(learning_rate)
    elif partitionargs['train']['optimizer']=='rmsprop':
        optimizer = tf.train.RMSPropOptimizer(learning_rate)

    # set robust modifier
    print('  robustifying optimizer')

    if partitionargs['train']['disable_robust']:
        train_op=optimizer.minimize(loss)
        global_norm=0
        clip=0
        m_unbiased=0

    else:
        import robust

        if not partitionargs['train']['window_size']:
            try:
                partitionargs['train']['window_size']=data.train_numdp
            except:
                partitionargs['train']['window_size']=10000

        robust_log_dir=None
        if partitionargs['train']['robust_log']:
            robust_log_dir=log_dir

        train_op=robust.robust_minimize(
                optimizer,
                loss,
                loss_per_dp,
                global_step,
                partitionargs['train']['batch_size'],
                y_,
                clip_method=partitionargs['train']['clip_method'],
                clip_type=partitionargs['train']['clip_type'],
                clip_function=partitionargs['train']['clip_function'],
                clip_percentile=partitionargs['train']['clip_percentile'],
                clip_threshold=partitionargs['train']['clip_threshold'],
                clip_perclass=not partitionargs['train']['disable_clip_perclass'],
                window_size=partitionargs['train']['window_size'],
                log_dir=robust_log_dir,
                marks=[z_,y_argmax_,mod_],
                )

    #if partitionargs['train']['tensorboard']:
        #with tf.name_scope('batch/'):
            #for grad,var in grads_and_vars:
                #tf.summary.histogram(var.name+'_grad',grad)
            #for grad,var in grads_and_vars2:
                #tf.summary.histogram(var.name+'_grad_rob',grad)

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

    saver=tf.train.Saver(max_to_keep=1)
    with tf.Session().as_default() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord) #,sess=sess)

        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        # create a log dir

        #import hashlib
        #import os
        #import shutil
        #optshash=hashlib.sha224(str(partitionargs)).hexdigest()
        #dirname = 'optshash='+optshash+'-'+dirname_opts_str
        #log_dir = partitionargs['train']['log_dir']+'/'+dirname
        #log_dir_current=log_dir+'/'+'current'
        #log_dir_best=log_dir+'/'+'best'
#
        if partitionargs['train']['delete_log']:
            shutil.rmtree(log_dir,ignore_errors=True)
        else:
            try:
                if partitionargs['train']['restore_from'] is None:
                    restore_dir=log_dir
                    restore_subdir='/current'
                else:
                    restore_dir=partitionargs['train']['restore_from']
                    restore_subdir='/best'
                    print('  restoring from ',restore_dir)
                chpt=tf.train.latest_checkpoint(restore_dir+restore_subdir)
                saver.restore(sess,chpt)
                with open(restore_dir+'/best/loss.txt','r') as f:
                    best_loss=float(f.readline())
                with open(restore_dir+'/best/epoch.txt','r') as f:
                    best_epoch=float(f.readline())
            except Exception as e:
                print('  failed to restore: ',e)

        # populate log_dir

        writer_train = tf.summary.FileWriter(log_dir+'/train',sess.graph)
        writer_test = tf.summary.FileWriter(log_dir+'/test',sess.graph)

        with open(log_dir+'/opts.txt','w',1) as f:
            import json
            f.write(json.dumps(partitionargs, sort_keys=True, indent=4))

        file_batch=open(log_dir+'/batch.txt','w',1)
        file_epoch=open(log_dir+'/epoch.txt','w',1)
        file_results=open(log_dir+'/results.txt','w',1)

        if partitionargs['train']['dump_data']:
            import scipy.io
            scipy.io.savemat(
                    log_dir+'/data.mat',
                    {'train_Y':data.train_Y,
                     'mu':data.mu
                    })

        ########################################
        if partitionargs['train']['sklearn_regression'] or partitionargs['train']['sklearn_classification']:
            print('sklearn')
            from sklearn import linear_model
            from sklearn.metrics import mean_squared_error,accuracy_score
            file_sklearn=open(log_dir+'/results-sklearn.txt','w',1)

            train_X=[]
            train_Y=[]
            sess.run(iterator.make_initializer(data.train))
            try:
                while True:
                    X,Y=sess.run([x_,y_])
                    train_X.append(X)
                    if partitionargs['train']['sklearn_regression']:
                        train_Y.append(Y)
                        test_Y=data.test_Y
                    else:
                        train_Y.append(np.argmax(Y,axis=1))
                        test_Y=np.argmax(data.test_Y,axis=1)
            except tf.errors.OutOfRangeError:
                train_X=np.concatenate(train_X,axis=0)
                train_Y=np.concatenate(train_Y,axis=0)

            if partitionargs['train']['sklearn_regression']:
                models=[('ols',linear_model.LinearRegression())
                       ,('Theil-Sen', linear_model.TheilSenRegressor(random_state=42))
                       ,('RANSAC', linear_model.RANSACRegressor(random_state=42))
                       ,('HuberRegressor', linear_model.HuberRegressor())]
                skloss=mean_squared_error
            else:
                models=[('logreg',linear_model.LogisticRegression(C=1e6))]
                skloss=accuracy_score

            for (name,model) in models:
                model.fit(train_X,train_Y)
                result_model=skloss(model.predict(data.test_X),test_Y)
                print('  model:',name,'; loss:',result_model)
                file_sklearn.write(str(result_model)+' ')

            file_sklearn.write('\n')
            file_sklearn.close()

        if partitionargs['train']['naive_mean']:
            file_results.write(' '.join(map(str,data.naive_accuracies))+' ')

        ########################################
        print('training')

        local_vars=tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES,scope='epoch/')
        reset_summary_vars=tf.variables_initializer(local_vars)
        sess.graph.finalize()

        _epoch = sess.run(epoch)
        while _epoch <= partitionargs['train']['epochs']:

            epoch_start=time.clock()

            # train one epoch on training set
            if _epoch>0:
                sess.run(init_train,feed_dict={is_training:True})
                try:
                    sess.run(reset_summary_vars)
                    while True:
                        batch_start=time.clock()
                        loss_res,_,summary,_=sess.run(
                                [loss,loss_updates,summary_batch,train_op],
                                feed_dict={is_training:True}
                                )
                        if partitionargs['train']['verbose']:
                            print('    step=%d, loss=%g, sec=%g           '%(global_step.eval(),loss_res,time.clock()-batch_start))
                            print('\033[F',end='')
                        if partitionargs['train']['savebatch']:
                            file_batch.write(str(global_step.eval())+' '+str(loss_res)+'\n')
                        if partitionargs['train']['tensorboard']:
                            writer_train.add_summary(summary, global_step.eval())
                except tf.errors.OutOfRangeError:
                    summary=sess.run(summary_epoch)
                    writer_train.add_summary(summary,global_step.eval())

            # evaluate on validation set
            sess.run(init_valid,feed_dict={is_training:False})
            try:
                sess.run(reset_summary_vars)
                validation_step=0
                while True:
                    sess.run(loss_updates,feed_dict={is_training:False})
                    if partitionargs['train']['verbose']:
                        print('    validation_step=%d'%validation_step)
                        print('\033[F',end='')
                    validation_step+=1
            except tf.errors.OutOfRangeError:
                res,summary=sess.run([loss_values,summary_epoch],feed_dict={is_training:False})
                if partitionargs['train']['tensorboard']:
                    writer_test.add_summary(summary,global_step.eval())

                nextline=str(_epoch)+' '+' '.join(map(str,res))
                nextline=filter(lambda x: x not in '[],', nextline)
                file_epoch.write(nextline+'\n')

            # save current model
            try:
                os.stat(log_dir_current)
            except:
                os.mkdir(log_dir_current)
            saver.save(sess,log_dir_current+'/model.chpt',global_step=global_step)

            # update best model
            import shutil
            try:
                with open(log_dir_best+'/loss.txt','r') as f:
                    best_loss=float(f.readline())
                with open(log_dir_best+'/epoch.txt','r') as f:
                    best_epoch=float(f.readline())
            except:
                best_loss=float('inf')
                best_epoch=0

            if res[0]<best_loss:
                shutil.rmtree(log_dir_best,ignore_errors=True)
                shutil.copytree(log_dir_current, log_dir_best)
                with open(log_dir_best+'/loss.txt','w') as f:
                    f.write('%f\n'%res[0])
                with open(log_dir_best+'/epoch.txt','w') as f:
                    f.write('%d\n'%_epoch)
                    best_epoch=_epoch

            # print results
            if best_epoch==_epoch:
                highlight='*'
            else:
                highlight=' '
            print('  '+time.strftime('%Y-%m-%d %H:%M:%S ',time.localtime(time.time()))
                      +highlight
                      +'epoch: %d    '%_epoch,' -- ',res,'         ')

            # early stopping
            if _epoch>=best_epoch+partitionargs['train']['early_stop_check']:
                print('early stopping')
                break

            # update epoch counter
            sess.run(epoch_update)
            _epoch=sess.run(epoch)

        file_results.write(' '.join(map(str,res))+'\n')

        # evaluate on test set
        print('evaluating on test set on epoch ',best_epoch)
        chpt=tf.train.latest_checkpoint(log_dir_best)
        saver.restore(sess,chpt)
        sess.run(init_test,feed_dict={is_training:False})
        try:
            sess.run(reset_summary_vars)
            while True:
                sess.run(loss_updates,feed_dict={is_training:False})
        except tf.errors.OutOfRangeError:
            res=sess.run(loss_values,feed_dict={is_training:False})
            print('  res=',res)
        with open(log_dir+'/eval.txt','w',1) as f:
            f.write(str(res)+'\n')
