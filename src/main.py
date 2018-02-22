#!/usr/bin/env python

from __future__ import print_function

import collections
import copy
import itertools
import math
import os
import pkgutil
import sys
import sklearn
import tflearn
import time
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

########################################
print('processing cmdline args')
import argparse
from interval import interval,Interval

parser=argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest='command')

default_functions={
        'data'              :'synthetic',
        'model'             :'glm',
        'hyperparam'        :'simple',
        'preprocess'        :'simple',
        'train'             :'local',
        }

modules={}
path=os.path.dirname(__file__)
for function in default_functions.keys():
    modules[function]={}
    exec('import '+function)
    pkg=eval(function)
    subparser = subparsers.add_parser(function)
    subsubparsers = subparser.add_subparsers(dest='subcommand')
    for loader, name, is_pkg in pkgutil.walk_packages(pkg.__path__):
        if not name[0]=='_':
            modules[function][name]=loader.find_module(name).load_module(name)
            modules[function][name].modify_parser(subsubparsers)

args={}
for k,v in default_functions.iteritems():
    args[k] = parser.parse_args([k,v])

argvv = [list(group) for is_key, group in itertools.groupby(sys.argv[1:], lambda x: x=='--') if not is_key]

for argv in argvv:
    args[argv[0]]=parser.parse_args(argv)

########################################
print('launching hyperparam optimizer')

hyperparam = modules['hyperparam'][args['hyperparam'].subcommand]
hyperparam.hyper_opt(modules,args)
