def modify_parser(subparsers):
    parser = subparsers.add_parser('loss', help='plots the loss')
    parser.add_argument('--epoch_resolution',type=int,default=1)

class Graph:
    def __init__(self,fig,pos,label,args,opts):
        self.args=args
        self.fig=fig
        self.pos=pos
        self.label=label
        self.args=args
        self.num_records=0
        self.aves=[]
        self.integrals=[]

    def get_num_frames(self):
        return 0

    def record_epoch(self,vars):
        if vars['epoch']%self.args.epoch_resolution==0:
            self.num_records+=1

            dict_ave={vars['x_']:vars['X'],vars['y_']:vars['Y']}
            res_ave,=vars['sess'].run([vars['loss_ave']],feed_dict=dict_ave)
            self.aves.append(res_ave)

            dict_integral={vars['x_']:vars['ax_data_xs2']}
            res_integral,=vars['sess'].run([vars['loss_true_ave']],feed_dict=dict_integral)
            self.integrals.append(res_integral)
        
    def finalize(self,vars):
        self.subplot=self.fig.add_subplot(self.pos,label='loss:'+self.label)
        self.subplot.plot(range(0,self.num_records),self.aves,':',color=(0,0,1))
        self.subplot.plot(range(0,self.num_records),self.integrals,'-',color=(0,0,1))
        self.subplot.set_yscale('log')

    def update(self,frame):
        pass

    def set_visible(self,b):
        self.subplot.set_visible(b)
