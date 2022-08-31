import os
import sys


def main():
    port = sys.argv[1]
    lines = os.popen(f'lsof -i :{port}').readlines()[1:]
    for line in lines:
        pid = line.split(' ')[2]
        cmd = f'kill -9 {pid}'
        print(cmd)
        os.system(cmd)


if __name__ == '__main__':
    main()
