# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import os
import sys
import argparse
import re

def parse_arguments():
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--src', help='source python file path; can contain {n}', required=True)
    parser.add_argument('--dst', help='output directory; can contain {n}', required=True)
    parser.add_argument('--name', help='name - will be substituted into {n} in src and dst')
    return parser.parse_args()

def run(args):
    src = args.src
    if args.name is not None:
        src = src.format(n=args.name)

    dst = args.dst
    if args.name is not None:
        dst = dst.format(n=args.name)

    with open(src, 'r') as fin:
        lines = fin.readlines()

    ext = os.path.splitext(src)[1]
    if ext == '.py':
        hide_start_re = re.compile(r"# HIDE_IN_TEMPLATE_START")
        hide_stop_re = re.compile(r"# HIDE_IN_TEMPLATE_STOP")
        show_re = re.compile(r"(\s*)# SHOW_IN_TEMPLATE (.*)", re.DOTALL)
    elif ext == '.ipynb':
        hide_start_re = re.compile(r"# HIDE_IN_TEMPLATE_START")
        hide_stop_re = re.compile(r"# HIDE_IN_TEMPLATE_STOP")
        show_re = re.compile(r"(.*)# SHOW_IN_TEMPLATE (.*)", re.DOTALL)
    else:
        raise NotImplementedError("Conversion of {} files not supported yet".format(ext))


    output = []

    showing = True
    for linenum, line in enumerate(lines):
        if hide_start_re.search(line):
            if not showing:
                raise RuntimeError("Nested hiding start on line {}".format(linenum + 1))
            showing = False

        if hide_stop_re.search(line):
            if showing:
                raise RuntimeError("Hiding stop without hiding start on line {}".format(linenum + 1))
            showing = True
            continue

        if show_re.match(line):
            match = show_re.match(line)
            line = match.group(1)+match.group(2)

        if showing:
            output.append(line)

    contents = ''.join(output)
    filename = os.path.basename(src)
    out_path = os.path.join(dst, filename)

    with open(out_path, 'w') as fout:
        fout.write(contents)

    return 0

def main():
    args = parse_arguments()
    return run(args)

if __name__ == '__main__':
    sys.exit(main())
