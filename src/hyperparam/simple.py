from __future__ import print_function

import argparse
from interval import interval,Interval

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
                if not param.start == param.stop:
                    raise ValueError('args['+command+'].'+param_name+' cannot be an interval in the simple hyperparam optimizer')
                partitionargs[command][param_name]=param.start
            else:
                partitionargs[command][param_name] = eval('args[command].'+param_name)

    data = modules['data'][args['data'].subcommand]
    model = modules['model'][args['model'].subcommand]

    import preprocess.simple
    preprocess.simple.preprocess_data(data,partitionargs)

    import train.local
    train.local.train_with_hyperparams(model,data,partitionargs)
