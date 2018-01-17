def modify_parser(subparsers):
    parser = subparsers.add_parser('loss', help='plots the loss')

class Graph:
    def __init__(self,fig,pos,label,args,opts):
        self.args=args

    def add_summary(self):
        import tensorflow as tf
        updates=[]
        values=[]
        for loss in tf.get_default_graph().get_collection(tf.GraphKeys.LOSSES):
            with tf.name_scope('batch/'):
                tf.summary.scalar(loss.name,loss,collections=['batch'])
            with tf.name_scope('epoch/'):
                value,update=tf.contrib.metrics.streaming_mean(loss)
                updates.append(update)
                tf.summary.scalar(loss.name,value,collections=['epoch'])
            values.append(value)
        return values,updates
