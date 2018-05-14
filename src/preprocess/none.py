def modify_parser(subparsers):
    import argparse
    from interval import interval

    subparser = subparsers.add_parser('none')
