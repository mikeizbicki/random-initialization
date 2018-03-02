from __future__ import print_function

def robust_minimize(
        optimizer,
        loss,
        loss_per_dp,
        global_step,
        batch_size,
        clip_method='dp', #dp,batch,batch_naive
        clip_type='global',
        clip_activation='soft',
        clip_percentile=99,
        window_size=1000,
        log_dir=None,
        marks=[],
        ):

    import tensorflow as tf
    import numpy as np
    import math

    window_size=int(batch_size*math.ceil(window_size/batch_size))

    with tf.name_scope('robust_minimize'):

        # setup clipping

        trim_size=global_step
        if clip_method=='dp' or clip_method=='dp_naive':
            trim_size=global_step*batch_size

        ms = tf.Variable(tf.zeros([window_size]),trainable=False)
        ms_trimmed = ms[0:tf.minimum(window_size,trim_size+1)]
        m=tf.contrib.distributions.percentile(ms_trimmed,50)

        clip=tf.contrib.distributions.percentile(ms,clip_percentile)
        #clip=tf.contrib.distributions.percentile(ms_trimmed,clip_percentile_modified)

        def clip_gradients(gradients,norm):

            if clip_type=='none':
                gradients2=gradients

            elif clip_type=='global':
                #if opts['verbose']:
                    #clip = tf.cond(
                            #clip>=global_norm,
                            #lambda:clip,
                            #lambda:tf.Print(clip,[global_norm,clip],'clipped'),
                            #)
                if clip_activation=='soft':
                    gradients2, _ = tf.clip_by_global_norm(gradients, clip, use_norm=norm)
                elif clip_activation=='hard':
                    gradients2=[]
                    for grad in gradients:
                        if grad==None:
                            grad2=None
                        else:
                            grad2=tf.cond(
                                    global_norm>clip,
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
                clip=clip_gradients(grad,norm)
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

            index_start=tf.mod( global_step   *batch_size,window_size)
            index_stop =tf.mod((global_step+1)*batch_size,window_size)
            ms_update = tf.assign(ms[index_start:index_stop],norms.stack())

        elif clip_method=='dp_naive':
            all_gradients = []
            all_norms = []
            for i in range(0,batch_size):
                grads_and_vars=optimizer.compute_gradients(loss_per_dp[i,...])
                dp_gradients,variables = zip(*grads_and_vars)
                dp_norm = tf.global_norm(dp_gradients)
                dp_gradients2 = clip_gradients(dp_gradients,dp_norm)
                all_gradients.append(dp_gradients2)
                all_norms.append(dp_norm)
            gradients2 = [ sum(i)/batch_size for i in zip(*all_gradients) ]
            global_norm= sum(all_norms)/batch_size 

            index_start=tf.mod( global_step   *batch_size,window_size)
            index_stop =tf.mod((global_step+1)*batch_size,window_size)
            all_norms=tf.stack(all_norms)
            ms_update = tf.assign(ms[index_start:index_stop],all_norms)

        elif clip_method=='batch_naive':
            all_gradients = []
            for i in range(0,batch_size):
                grads_and_vars=optimizer.compute_gradients(loss_per_dp[i,...])
                dp_gradients,variables = zip(*grads_and_vars)
                all_gradients.append(dp_gradients)
            gradients = [ sum(i)/batch_size for i in zip(*all_gradients) ]
            global_norm = tf.global_norm(gradients)
            ms_update = tf.assign(ms[tf.mod(global_step,window_size)],global_norm)
            gradients2 = clip_gradients(gradients,global_norm)
            all_norms=tf.tile(tf.reshape(global_norm,shape=[1]),[batch_size])

        elif clip_method=='batch':
            grads_and_vars=optimizer.compute_gradients(loss)
            gradients, variables = zip(*grads_and_vars)
            global_norm = tf.global_norm(gradients)
            gradients2 = clip_gradients(gradients,global_norm)
            ms_update = tf.assign(ms[tf.mod(global_step,window_size)],global_norm)
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
    train_op = tf.group(grad_updates,log_update,ms_update)

    return train_op 
