
def classification(args,y_,y):
    import tensorflow as tf

    with tf.name_scope('losses'):
        xentropy_per_dp = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
        xentropy = tf.reduce_mean(xentropy_per_dp,name='xentropy')
        tf.add_to_collection(tf.GraphKeys.LOSSES,xentropy)

        robustify = lambda x: tf.cond(
                tf.linalg.norm(x)>1,
                lambda:x/tf.sqrt(tf.linalg.norm(x)),
                lambda:x)
        xentropy_robust_per_dp=tf.nn.softmax_cross_entropy_with_logits(labels=robustify(y_), logits=y)
        xentropy_robust = tf.reduce_mean(xentropy_robust_per_dp,name='xentropy_robust')
        tf.add_to_collection(tf.GraphKeys.LOSSES,xentropy_robust)

        argmax_y =tf.argmax(y ,axis=1)
        argmax_y_=tf.argmax(y_,axis=1)
        results = tf.cast(tf.equal(argmax_y,argmax_y_),tf.float32)
        accuracy = tf.reduce_mean(results,name='accuracy')
        tf.add_to_collection(tf.GraphKeys.LOSSES,accuracy)

        #regularization=0
        #for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            #regularization+=tf.nn.l2_loss(var)*args['l2']

        if args['loss']=='default':
            args['loss']='xentropy'

        loss = eval(args['loss'])
        loss_per_dp = eval(args['loss']+'_per_dp')

    return loss,loss_per_dp

def regression(args,y_,y):
    import tensorflow as tf

    with tf.name_scope('losses'):
        diff_y = tf.abs(y_-y)
        mse_per_dp = diff_y**2
        mse = tf.reduce_mean(mse_per_dp)
        huber_per_dp = tf.where(diff_y<1,diff_y**2,diff_y)
        huber = tf.reduce_mean(huber_per_dp)
        absdiff_per_dp = diff_y
        absdiff = tf.reduce_mean(diff_y)

        tf.add_to_collection(tf.GraphKeys.LOSSES,mse)
        tf.add_to_collection(tf.GraphKeys.LOSSES,huber)
        tf.add_to_collection(tf.GraphKeys.LOSSES,absdiff)

        if args['loss']=='default':
            args['loss']='mse'

        loss = eval(args['loss'])
        loss_per_dp = eval(args['loss']+'_per_dp')

    return loss,loss_per_dp

