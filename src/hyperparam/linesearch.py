from __future__ import print_function

import argparse
from interval import Interval

def modify_parser(subparsers):
    subparser = subparsers.add_parser('linesearch')
    subparser.add_argument('--resolution',type=int,default=1)

def hyper_opt(model,data,args):
    partitionargs={}
    for command in ['model','data','common']:
        partitionargs[command]={}
        param_names=filter(lambda x: x[0]!='_', dir(args[command]))
        for param_name in param_names:
            param=eval('args["'+command+'"].'+param_name)
            if isinstance(param,Interval):
                res=param.start+partition*(param.stop-param.start)/(args['hyperparam'].resolution+1)
                partitionargs[command][param_name]=res
            elif type(param) is list:
                ress=[]
                all_trivial=True
                for p in param:
                    if isinstance(p,Interval):
                        res=p.start+partition*(p.stop-p.start)/(args['hyperparam'].partitions+1)
                        ress.append(res)
                        if not p.is_trivial:
                            all_trivial=False
                partitionargs[command][param_name] = ress
            else:
                partitionargs[command][param_name] = eval('args[command].'+param_name)

    import train
    train.train_with_hyperparams(model,data,partitionargs)

