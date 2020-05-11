if __name__ == '__main__':
    from kaldiio import WriteHelper
    import kaldiio
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, nargs='+')
    args = parser.parse_args()
    for file in args.file:
        dirname = os.path.dirname(file)
        filename = os.path.basename(file)
        with WriteHelper('ark,scp:' + os.path.join(dirname, 't_' + filename) + ',' +
                         os.path.join(dirname, filename[:-4] + '.scp')) as writer:
            ark = kaldiio.load_ark(file)
            for key, na in ark:
                writer(key, na)
