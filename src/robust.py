from __future__ import print_function

def robust_minimize(
        optimizer,
        loss,
        loss_per_dp,
        global_step,
        batch_size,
        clip_method='dp_naive', #dp_naive,batch
        clip_type='global',
        clip_activation='soft',
        clip_percentile=99,
        burn_in=0,
        window_size=1000
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
        clip=tf.contrib.distributions.percentile(ms_trimmed,clip_percentile)

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
            ms_update = tf.assign(ms[index_start:index_stop],tf.stack(all_norms))

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

        elif clip_method=='batch':
            grads_and_vars=optimizer.compute_gradients(loss)
            gradients, variables = zip(*grads_and_vars)
            global_norm = tf.global_norm(gradients)
            gradients2 = clip_gradients(gradients,global_norm)
            ms_update = tf.assign(ms[tf.mod(global_step,window_size)],global_norm)

    # apply gradients

    grads_and_vars2=zip(gradients2,variables)
    grad_updates=optimizer.apply_gradients(
            grads_and_vars2,
            global_step=global_step)
    train_op = tf.group(grad_updates,ms_update)

    #train_op = tf.cond(
            #global_step<burn_in,
            #lambda:tf.group(
                #update_clipper,
                #tf.assign(global_step,global_step+1)
                #),
            #lambda:tf.group(
                #update_clipper,
                #optimizer.apply_gradients(
                    #grads_and_vars2,
                    #global_step=global_step)
                #)
            #)

    return [train_op,global_norm,clip,m]

################################################################################

        #total_parameters=0
        #for _,var in grads_and_vars:
            #variable_parameters = 1
            #for dim in var.get_shape():
                #variable_parameters *= dim.value
            #total_parameters += variable_parameters
        #print('    total_parameters=',total_parameters)

        #if opts['no_median']:
            #m_alpha = 0.999 # opts['m_alpha']
            #v_alpha = 0.999 # opts['v_alpha']
            #m_init=0.0
            #v_init=0.0
            #epsilon=1e-9
            #burn_in=10
#
            #m = tf.Variable(m_init,trainable=False)
            #v = tf.Variable(v_init,trainable=False)
            #m_unbiased = m/(1.0-m_alpha**(1+global_step_float))
            #v_unbiased = v/(1.0-v_alpha**(1+global_step_float))
#
            #clip = tf.cond(global_step<burn_in,
                    #lambda: global_norm,
                    #lambda: m_unbiased+epsilon+(tf.sqrt(v_unbiased))*opts['num_stddevs']/math.sqrt(opts['batch_size'])/2
                    #)
#
            ##m2 = m_alpha*m + (1-m_alpha)*global_norm
            ##v2 = v_alpha*v + (1-v_alpha)*global_norm**2
            #m2 = m_alpha*m + (1.0-m_alpha)*tf.minimum(global_norm,clip)
            #v2 = v_alpha*v + (1.0-v_alpha)*tf.minimum(global_norm,clip)**2
            #m_update = tf.assign(m,m2)
            #v_update = tf.assign(v,v2)
            #update_clipper = tf.group(m_update,v_update)
#
            #if opts['tensorboard']:
                #tf.summary.scalar('m',m)
                #tf.summary.scalar('v',v)
                #tf.summary.scalar('m_unbiased',m_unbiased)
                #tf.summary.scalar('v_unbiased',v_unbiased)
#
        #else:
        #elif opts['robust']=='local':
            #gradients2=[]
            #update_clipper=tf.group()
#
            #for grad,var in grads_and_vars:
                #rawname=var.name.split(':')[0]
                #ones = np.ones(var.get_shape())
                #m = tf.Variable(m_init*ones,name=rawname,trainable=False,dtype=tf.float32)
                #v = tf.Variable(v_init*ones,name=rawname,trainable=False,dtype=tf.float32)
                ##m_unbiased = m/(1.0-m_alpha)
                ##v_unbiased = v/(1.0-v_alpha)
                #m_unbiased = m/(1.0-m_alpha**(1+global_step_float))
                #v_unbiased = v/(1.0-v_alpha**(1+global_step_float))
                ##v_unbiased = (v-(1.0-v_alpha)*v_init)/(1.0-v_alpha**(1+global_step_float))
                #clip = m_unbiased+tf.sqrt(v_unbiased)*opts['num_stddevs']
                #grad2_abs = tf.minimum(tf.abs(grad),clip)
                #m2 = m_alpha*m + (1.0-m_alpha)*tf.minimum(grad2_abs,clip)
                #v2 = v_alpha*v + (1.0-v_alpha)*tf.minimum(grad2_abs,clip)**2
                #grad2=tf.sign(grad)*grad2_abs
                #gradients2.append(grad2)
                #m_update=tf.assign(m,m2)
                #v_update=tf.assign(v,v2)
                #update_clipper=tf.group(update_clipper,m_update,v_update)

