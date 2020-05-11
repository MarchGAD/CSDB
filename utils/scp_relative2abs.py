# written by wyc.hit 2020
# This script will replace the relative path to absolute path in .scp file
# from all .scp files under the specified directory.
import os
import re
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('path', type=str)
args = parser.parse_args()
path = args.path
if not os.path.isdir(path):
    raise Exception('Please give a path of a directory.')
sta = [os.path.abspath(path)]
while len(sta) != 0:
    path = sta.pop()
    ls = os.listdir(path)
    for name in ls:
        new_path = os.path.join(path, name)
        if os.path.isdir(new_path):
            sta.append(new_path)
        elif os.path.isfile(new_path):
            if '.scp' not in new_path:
                continue
            pre = None
            with open(new_path, 'r') as fin:
                for line in fin:
                    pre = re.match(r'(.*?) (.*?.ark):([0-9]+)', line).group(2)
                    break
            rpath = os.path.join(os.path.dirname(new_path), pre)
            aft = os.path.abspath(rpath)
            tmp_path = new_path + '__tmp'
            with open(new_path, 'r') as fin:
                with open(tmp_path, 'w') as fout:
                    for line in fin:
                        fout.write(line.replace(pre, aft))
            os.remove(new_path)
            os.rename(tmp_path, new_path)
        else:
            raise Exception('What the Hell it is? %s' % new_path)
