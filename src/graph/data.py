def modify_parser(subparsers):
    parser = subparsers.add_parser('data', help='plots the data')
    parser.add_argument('--epoch_resolution',type=int,default=10)
    parser.add_argument('--plot_data',type=str,default='True')
    parser.add_argument('--xmin',type=float,default=-10)
    parser.add_argument('--xmax',type=float,default=10)
    parser.add_argument('--xresolution',type=float,default=256)

class Graph():
    def __init__(self,fig,pos,label,args,opts):
        self.args=args
        self.subplot=fig.add_subplot(pos,label='data:'+label)
        self.data=[]
        self.lines=[]

    def get_num_frames(self):
        return len(self.data)/self.args.epoch_resolution

    def init_step(self,vars):
        import numpy as np
        xresolution=self.args.xresolution
        xmin=self.args.xmin
        xmax=self.args.xmax
        xmargin=0.1*(xmax-xmin)/2
        self.ax_data_xs = np.linspace(xmin-xmargin,xmax+xmargin,xresolution).reshape(xresolution,1)

    def record_epoch(self,vars):
        ax_data_ys = vars['sess'].run(vars['y'],feed_dict={vars['x_']:self.ax_data_xs})
        self.data.append(ax_data_ys)

    def finalize(self,vars):
        import numpy as np
        data=vars['data']

        if self.args.plot_data=='True':
            xresolution=self.args.xresolution
            xmin=self.args.xmin
            xmax=self.args.xmax
            ax_data_xs2 = np.linspace(xmin,xmax,xresolution).reshape(xresolution,1)
            #ax_data_ys2 = vars['sess'].run(vars['y_true'],feed_dict={vars['x_']:ax_data_xs2})
            #self.subplot.plot(ax_data_xs2,ax_data_ys2,'-',color=(0,0,1),zorder=100000)
            self.subplot.plot(data.train.X,data.train.Y,'.',color=(0,0,1),zorder=100001)
            self.subplot.plot(data.test.X,data.test.Y,'.',color=(1,0,1),zorder=100001)

    def update(self,frame):
        import matplotlib.pyplot as plt
        plt.suptitle('epoch = '+str(frame*self.args.epoch_resolution))

        index=frame*self.args.epoch_resolution
        line,=self.subplot.plot(self.ax_data_xs,self.data[index],'-',color=(0,0.5,0))
        self.lines.append(line)

        gradientlen=5
        colorstep=0.5/gradientlen
        for j in range(1,gradientlen):
            if frame-j>=0:
                plt.setp(self.lines[frame-j],color=(j*colorstep,0.5+j*colorstep,j*colorstep))

    def set_visible(self,b):
        self.subplot.set_visible(b)
