def modify_parser(subparsers):
    parser = subparsers.add_parser('loss', help='plots the loss')
    parser.add_argument('--epoch_resolution',type=int,default=1)

class Graph:
    def __init__(self,subplot,args):
        self.subplot=subplot
        subplot.set_xlabel('epoch')
        subplot.set_ylabel('loss')

        self.args=args
        self.num_records=0
        self.aves=[]
        self.integrals=[]

    def get_num_frames(self):
        return 0

    def record_epoch(self,vars):
        if vars['epoch']%self.args.epoch_resolution==0:
            self.num_records+=1

            res_ave,=vars['sess'].run([vars['loss_ave']],feed_dict={vars['x_']:vars['X'],vars['y_']:vars['Y']})
            self.aves.append(res_ave)

            res_integral,=vars['sess'].run([vars['loss_true_ave']],feed_dict={vars['x_']:vars['ax_data_xs2']})
            self.integrals.append(res_integral)
        
    def finalize(self,vars):
        self.subplot.plot(range(0,self.num_records),self.aves,'-',color=(1,0,0))
        self.subplot.plot(range(0,self.num_records),self.integrals,'-',color=(0,1,0))

    def update(self,frame):
        pass
