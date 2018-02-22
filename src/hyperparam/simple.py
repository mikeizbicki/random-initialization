from __future__ import print_function

import argparse
from interval import Interval

def modify_parser(subparsers):
    subparser = subparsers.add_parser('simple')

def hyper_opt(modules,args):
    partitionargs={}
    for command in modules.keys():
        partitionargs[command]={}
        param_names=filter(lambda x: x[0]!='_', dir(args[command]))
        for param_name in param_names:
            param=eval('args["'+command+'"].'+param_name)
            if isinstance(param,Interval):
                res=param.start+(param.stop-param.start)/2
                partitionargs[command][param_name]=res
            elif type(param) is list:
                ress=[]
                all_trivial=True
                for p in param:
                    if isinstance(p,Interval):
                        res=p.start+(p.stop-p.start)/2
                        ress.append(res)
                        if not p.is_trivial:
                            all_trivial=False
                partitionargs[command][param_name] = ress
            else:
                partitionargs[command][param_name] = eval('args[command].'+param_name)

    data = modules['data'][args['data'].subcommand]
    model = modules['model'][args['model'].subcommand]

    import preprocess.simple
    preprocess.simple.preprocess_data(data,partitionargs)

    import train.local
    train.local.train_with_hyperparams(model,data,partitionargs)
