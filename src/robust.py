from __future__ import print_function

def update_tensor(tensor,indices,updates):
    import tensorflow as tf
    newvals=tf.SparseTensor(indices,tf.stack(updates),tensor.get_shape())

    updates2=map(lambda i: tensor.__getitem__(i),indices)
    oldvals=tf.SparseTensor(indices,tf.stack(updates2),tensor.get_shape())
    #return tensor+tf.sparse_tensor_to_dense(newvals-oldvals)
    return tensor+tf.sparse_tensor_to_dense(newvals)-tf.sparse_tensor_to_dense(oldvals)

def robust_minimize(
        optimizer,
        loss,
        loss_per_dp,
        global_step,
        batch_size,
        y_,
        clip_method='dp', #dp,batch,batch_naive
        clip_type='global',
        clip_function='soft',
        clip_threshold=0.0, # implies doing it automatically
        clip_percentile=99,
        clip_perclass=True,
        window_size=1000,
        log_dir=None,
        marks=[],
        ):

    import tensorflow as tf
    import numpy as np
    import math

    if batch_size==1 or clip_type=='none':
        clip_method='batch'

    if clip_method=='batch' or clip_method=='batch_naive':
        clip_perclass=False

    window_size=int(batch_size*math.ceil(window_size/batch_size))

    with tf.name_scope('robust_minimize'):

        # setup clipping
        epsilon=1e-6

        if clip_perclass:
            num_windows=y_.get_shape()[1]
            label_steps=tf.Variable(tf.zeros([num_windows]),trainable=False,name='label_steps')
            label_steps_update=tf.assign(label_steps,label_steps+tf.reduce_sum(y_,axis=0))
            label_steps_int=tf.cast(label_steps,tf.int32)
            y_window_ = tf.argmax(y_,axis=1)
        else:
            num_windows=1
            label_steps=tf.Variable(tf.zeros([num_windows]),trainable=False,name='label_steps')
            label_steps_update=tf.assign(label_steps,label_steps+batch_size)
            label_steps_int=tf.cast(label_steps,tf.int32)
            #label_steps=tf.cast(tf.reshape(global_step,[1]),tf.float32)
            #label_steps_update=tf.group()
            #label_steps_int=label_steps
            y_window_ = tf.zeros([batch_size])

        ms = tf.Variable(tf.zeros([num_windows,window_size]),trainable=False,name='ms')
        trim_factor=tf.minimum(1.0,label_steps/window_size)

        def get_percentile(dist,p):
            xs=map(lambda i: tf.contrib.distributions.percentile(dist[i],p[i])+epsilon,range(0,dist.get_shape()[0]))
            return tf.stack(xs)
        m=get_percentile(ms,50*trim_factor)

        if clip_threshold<=0.0:
            clip=get_percentile(ms,clip_percentile*trim_factor)
            #clip=tf.contrib.distributions.percentile(ms,clip_percentile)+epsilon
            #clip=tf.contrib.distributions.percentile(ms_trimmed,clip_percentile_modified)
        else:
            clip=clip_threshold*tf.ones([num_windows])

        def clip_gradients(gradients,norm,clip_mod):

            if clip_type=='none':
                gradients2=gradients

            elif clip_type=='global':
                #if opts['verbose']:
                    #clip = tf.cond(
                            #clip>=global_norm,
                            #lambda:clip,
                            #lambda:tf.Print(clip,[global_norm,clip],'clipped'),
                            #)
                if clip_function=='soft':
                    tf.Print(clip_mod,[clip_mod])
                    gradients2, _ = tf.clip_by_global_norm(gradients, clip_mod, use_norm=norm)
                elif clip_function=='hard':
                    gradients2=[]
                    for grad in gradients:
                        if grad==None:
                            grad2=None
                        else:
                            grad2=tf.cond(
                                    norm>clip_mod,
                                    lambda:tf.zeros(grad.get_shape()),
                                    lambda:grad
                                    )
                        gradients2.append(grad2)

            return gradients2

        # calculate gradients

        if clip_method=='dp':
            # FIXME: this method makes no effort to place the variables on appropriate devices
            # when multiple devices are available
            variables = (
                    tf.trainable_variables() +
                    tf.get_collection(tf.GraphKeys.TRAINABLE_RESOURCE_VARIABLES) +
                    tf.get_collection(tf.GraphKeys._STREAMING_MODEL_PORTS)
                    )

            loop_vars = [
                    tf.constant(0,tf.int32),
                    tf.TensorArray(tf.float32,size=batch_size,clear_after_read=False),
                    map(lambda _: tf.TensorArray(tf.float32,size=batch_size,clear_after_read=False),variables)
                    ]

            def go(i,arr_norm,arr_vars):
                grad=tf.gradients(loss_per_dp[i],variables)
                norm=tf.global_norm(grad)
                clip=clip_gradients(grad,norm,tf.reduce_sum(clip*y_[i]))
                return [
                        i+1,
                        arr_norm.write(i,norm),
                        map(lambda (arr,g): arr.write(i,g),zip(arr_vars,clip))
                        ]

            _,norms,clips=tf.while_loop(
                    lambda i,arr_norm,arr_vars: i<batch_size,
                    go,
                    loop_vars
                    )

            gradients2 = [ tf.reduce_mean(g.stack(),axis=0) for g in clips ]
            all_norms=norms.stack()
            global_norm= tf.reduce_mean(norms.stack())

            i1,i2,ms_new = tf.while_loop(
                lambda batch_index,window_index,ms_new: batch_index<batch_size,
                lambda batch_index,window_index,ms_new:
                    (batch_index+1
                    ,window_index+y_[batch_index]
                    ,update_tensor(
                        ms_new,
                        [(y_window_[batch_index],tf.mod(tf.cast(window_index[y_window_[batch_index]],tf.int64),window_size))],
                        [all_norms[batch_index]]
                        )
                    ),
                [0,tf.mod(label_steps,window_size),ms]
                )
            ms_update=tf.assign(ms,ms_new)

        elif clip_method=='dp_naive':
            all_gradients = []
            all_norms = []
            for i in range(0,batch_size):
                grads_and_vars=optimizer.compute_gradients(loss_per_dp[i,...])
                dp_gradients,variables = zip(*grads_and_vars)
                dp_norm = tf.global_norm(dp_gradients)
                dp_gradients2 = clip_gradients(dp_gradients,dp_norm,tf.reduce_sum(clip*y_[i]))
                all_gradients.append(dp_gradients2)
                all_norms.append(dp_norm)
            gradients2 = [ sum(i)/batch_size for i in zip(*all_gradients) ]
            global_norm= sum(all_norms)/batch_size

            #index_start=tf.mod( global_step   *batch_size,window_size)
            #index_stop =tf.mod((global_step+1)*batch_size,window_size)
            #all_norms=tf.stack(all_norms)
            #ms_update = tf.assign(ms[index_start:index_stop],all_norms)

            all_norms=tf.stack(all_norms)
            i1,i2,ms_new = tf.while_loop(
                lambda batch_index,window_index,ms_new: batch_index<batch_size,
                lambda batch_index,window_index,ms_new:
                    (batch_index+1
                    ,window_index+y_[batch_index]
                    ,update_tensor(
                        ms_new,
                        [(y_window_[batch_index],tf.mod(tf.cast(window_index[y_window_[batch_index]],tf.int64),window_size))],
                        [all_norms[batch_index]]
                        )
                    ),
                [0,tf.mod(label_steps,window_size),ms]
                )
            ms_update=tf.assign(ms,ms_new)

        elif clip_method=='batch_naive':
            all_gradients = []
            for i in range(0,batch_size):
                grads_and_vars=optimizer.compute_gradients(loss_per_dp[i,...])
                dp_gradients,variables = zip(*grads_and_vars)
                all_gradients.append(dp_gradients)
            gradients = [ sum(i)/batch_size for i in zip(*all_gradients) ]
            global_norm = tf.global_norm(gradients)
            ms_update = tf.assign(ms[0,tf.mod(global_step,window_size)],global_norm)
            gradients2 = clip_gradients(gradients,global_norm,clip)
            all_norms=tf.tile(tf.reshape(global_norm,shape=[1]),[batch_size])

        elif clip_method=='batch':
            grads_and_vars=optimizer.compute_gradients(loss)
            gradients, variables = zip(*grads_and_vars)
            global_norm = tf.global_norm(gradients)
            gradients2 = clip_gradients(gradients,global_norm,clip)
            ms_update = tf.assign(ms[0,tf.mod(global_step,window_size)],global_norm)
            all_norms=tf.tile(tf.reshape(global_norm,shape=[1]),[batch_size])

    # setup logging

    if log_dir is not None:
        log_file=log_dir+'/robust.log'
        import os
        print('    robust log file = ',os.path.abspath(log_file))
        log=open(log_file,'a',1)

        def update_log(global_step,clip,m,norms,*marks):
            for i in range(0,norms.shape[0]):
                log.write(str(global_step)+' ')
                log.write(str(clip)+' ')
                log.write(str(m)+' ')
                log.write(str(norms[i])+' ')
                #log.write(str(id_[i])+' ')
                for mark in marks:
                    log.write(str(mark[i])+' ')
                log.write('\n')
            return []

        log_update=tf.py_func(update_log,[global_step,clip,m,all_norms]+marks,[])

    else:
        log_update=tf.group()

    # apply gradients

    grads_and_vars2=zip(gradients2,variables)
    grad_updates=optimizer.apply_gradients(
            grads_and_vars2,
            global_step=global_step)
    train_op = tf.group(grad_updates,log_update,label_steps_update,ms_update)

    return train_op
