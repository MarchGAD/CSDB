# written by wyc.hit 2020
# This script will replace specified strings
# from all files under the specified directory.
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', type=str)
parser.add_argument('-f', '--father', type=str)
parser.add_argument('-s', '--son', type=str)
args = parser.parse_args()
path = args.path
pre = args.father
aft = args.son
if not os.path.isdir(path):
    raise Exception('Please give a path of a directory.')
sta = [path]
while len(sta) != 0:
    path = sta.pop()
    ls = os.listdir(path)
    for name in ls:
        new_path = os.path.join(path, name)
        if os.path.isdir(new_path):
            sta.append(new_path)
        elif os.path.isfile(new_path):
            tmp_path = new_path + '__tmp'
            with open(new_path, 'r') as fin:
                with open(tmp_path, 'w') as fout:
                    for line in fin:
                        fout.write(line.replace(pre, aft))
            os.remove(new_path)
            os.rename(tmp_path, new_path)
        else:
            raise Exception('What the Hell it is? %s' % new_path)
