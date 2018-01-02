def modify_parser(subparsers):
    parser = subparsers.add_parser('weights', help='plots a histogram of the network\'s weights')
    parser.add_argument('--epoch_resolution',type=int,default=10)
    parser.add_argument('--layers',type=str,nargs='*',default=[])
    parser.add_argument('--vars',type=str,nargs='*',default=[])

class Graph():
    def __init__(self,fig,pos,label,args):
        import copy
        import matplotlib.gridspec as gridspec
        if len(args.layers)!=1:
            if args.layers==[]:
                args.layers=['layer0','layer_final']
            gs = gridspec.GridSpecFromSubplotSpec(len(args.layers),1,subplot_spec=pos)
            self.subgraphs=[]
            pos2 = 0
            for layer in args.layers:
                args2=copy.deepcopy(args)
                args2.layers=[layer]
                self.subgraphs.append(Graph(fig,gs[pos2],label,args2))
                pos2+=1

        elif len(args.vars)!=1:
            if args.vars==[]:
                args.vars=['w','b']
            gs = gridspec.GridSpecFromSubplotSpec(1,len(args.vars),subplot_spec=pos)
            self.subgraphs=[]
            pos2 = 0
            for var in args.vars:
                args2=copy.deepcopy(args)
                args2.vars=[var]
                self.subgraphs.append(Graph(fig,gs[pos2],label,args2))
                pos2+=1

        else:
            self.subgraphs=None
            layer=args.layers[0]
            var=args.vars[0]

            self.args=args
            self.subplot=fig.add_subplot(pos,label=layer+'/'+var+' ('+label+')')
            #self.subplot2=fig.add_subplot(gs[1],label='weights:'+label)
            self.args=args
            self.hists=[]

    def get_num_frames(self):
        return 1

    def record_epoch(self,vars):
        if self.subgraphs:
            for subgraph in self.subgraphs:
                subgraph.record_epoch(vars)

        else:
            import matplotlib.pyplot as plt
            import tensorflow as tf
            tensor=tf.get_default_graph().get_tensor_by_name('model/'+self.args.layers[0]+'/'+self.args.vars[0]+':0')
            w_,=vars['sess'].run([tensor],feed_dict={})
            [_,_,hist]=self.subplot.hist(w_.flatten(), normed=1, facecolor='green', alpha=0.75,linewidth=1,edgecolor=(0,0,0),histtype='stepfilled')
            self.hists.append(hist)

    def finalize(self,vars):
        pass

    def update(self,frame):
        if self.subgraphs:
            for subgraph in self.subgraphs:
                subgraph.update(frame)

        else:
            for i in range(0,len(self.hists)):
                for line in self.hists[i]:
                    line.set_visible(i==frame)
            self.subplot.relim()
            self.subplot.autoscale_view()
            self.subplot.autoscale()


    def set_visible(self,b):
        if self.subgraphs:
            for subgraph in self.subgraphs:
                subgraph.set_visible(b)
        else:
            self.subplot.set_visible(b)



