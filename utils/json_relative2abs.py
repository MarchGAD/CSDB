import os
import json
import argparse


class Params:

    def __init__(self, json_path=None):
        if json_path is not None:
            self.update(json_path)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        return self.__dict__


parser = argparse.ArgumentParser()
parser.add_argument('sour_path', type=str)
parser.add_argument('-d', '--dest_path', type=str, default=None)

args = parser.parse_args()

if args.dest_path is None:
    args.dest_path = args.sour_path

source = os.path.abspath(args.sour_path)
sou_dir = os.path.dirname(source)
destin = os.path.abspath(args.dest_path)
if not os.path.isfile(source):
    raise Exception('No such file %s.' % source)

params = Params(source)
for key in params.dict:
    path = os.path.join(sou_dir, str(params.dict[key]))
    if os.path.isfile(path):
        params.dict[key] = os.path.abspath(path)

params.save(destin)
