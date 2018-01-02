import argparse

def modify_parser(subparsers):
    parser = subparsers.add_parser('mnist', help='a handwritten digit dataset')


