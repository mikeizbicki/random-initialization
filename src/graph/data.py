def modify_parser(subparsers):
    parser = subparsers.add_parser('data', help='plots the data')
    parser.add_argument('--epoch_resolution',type=int,default=10)

class Graph():
    def __init__(self,fig,pos,label,args):
        self.args=args
        self.subplot=fig.add_subplot(pos,label='data:'+label)
        #self.subplot.set_xlabel('input values')
        #self.subplot.set_ylabel('output values')
        self.lines=[]

    def get_num_frames(self):
        return len(self.lines)

    def record_epoch(self,vars):
        if vars['epoch']%self.args.epoch_resolution==0:
            line,=self.subplot.plot(vars['ax_data_xs'],vars['ax_data_ys'],'-',visible=False)
            self.lines.append(line)

    def finalize(self,vars):
        ltrue,=self.subplot.plot(vars['ax_data_xs2'],vars['ax_data_ys2'],'-',color=(0,0,1))
        ldata,=self.subplot.plot(vars['X'],vars['Y'],'.',color=(0,0,1))

    def update(self,frame):
        import matplotlib.pyplot as plt
        self.lines[frame].set_visible(True)
        plt.setp(self.lines[frame],color=(0,0.5,0))
        plt.suptitle('epoch = '+str(frame*self.args.epoch_resolution))

        gradientlen=5
        colorstep=0.5/gradientlen
        for j in range(1,gradientlen):
            if frame-j>=0:
                plt.setp(self.lines[frame-j],color=(j*colorstep,0.5+j*colorstep,j*colorstep))

    def set_visible(self,b):
        self.subplot.set_visible(b)
