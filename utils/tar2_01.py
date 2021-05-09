import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-sf', '--source_file', type=str)
    parser.add_argument('-tf', '--target_file', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = main()
    soufile = args.source_file
    tarfile = args.target_file
    with open(soufile, 'r') as source:
        with open(tarfile, 'w') as target:
            for line in source:
                tmp = line.split()
                target.write(tmp[0] + '\t' + tmp[1] + '\t' +
                             ('1' if (tmp[2] == 'target') else '0') +
                             ('\n' if (line[-1] == '\n') else '')
                             )
