def modify_parser(subparsers):
    parser = subparsers.add_parser('loss', help='plots the loss')
    parser.add_argument('--epoch_resolution',type=int,default=1)
    parser.add_argument('--log_scale',action='store_true')

class Graph:
    def __init__(self,fig,pos,label,args,opts):
        self.args=args
        self.fig=fig
        self.pos=pos
        self.label=label
        self.args=args
        self.num_records=0
        self.trains=[]
        self.tests=[]

    def get_num_frames(self):
        return 0

    def init_step(self,vars):
        import numpy as np
        xmin=vars['xmin']
        xmax=vars['xmax']
        xmargin=vars['xmargin']
        xresolution=100
        self.ax_data_xs2 = np.linspace(xmin,xmax,xresolution).reshape(xresolution,1)
        pass

    def record_epoch(self,vars):
        if vars['epoch']%self.args.epoch_resolution==0:
            self.num_records+=1

            data=vars['data']
            dict_train={vars['x_']:data.train.X,vars['y_']:data.train.Y}
            res_train,=vars['sess'].run([vars['loss']],feed_dict=dict_train)
            self.trains.append(res_train)

            dict_test={vars['x_']:data.test.X,vars['y_']:data.test.Y}
            res_test,=vars['sess'].run([vars['loss']],feed_dict=dict_test)
            self.tests.append(res_test)
        
    def finalize(self,vars):
        self.subplot=self.fig.add_subplot(self.pos,label='loss:'+self.label)
        self.subplot.plot(range(0,self.num_records),self.trains,':',color=(0,0,1))
        self.subplot.plot(range(0,self.num_records),self.tests,'-',color=(0,0,1))

        if self.args.log_scale:
            self.subplot.set_yscale('log')
        else:
            self.subplot.set_ylim([0,1])

    def update(self,frame):
        pass

    def set_visible(self,b):
        self.subplot.set_visible(b)
