import sys


def extract_results(filename):
    with open(filename) as f:
        content = f.readlines()
    flag = False
    tot_time = 0
    tot_gate = 0
    gate_product = 1
    for line in content:
        if flag:
            pos = line.find(':')
            pos2 = line.find(',', pos)
            pos3 = line.find('s', pos2)
            print(line[pos:pos2])
            tot_gate += int(line[pos + 2:pos2])
            gate_product *= int(line[pos + 2:pos2])
            tot_time += float(line[pos2 + 2:pos3])
        if line.startswith('Optimization'):
            flag = True
            pos = line.find('.qasm')
            pos2 = line.rfind(' ', 0, pos)
            print(line[pos2 + 1:pos], end=' ')
        else:
            flag = False
    print('tot_gate =', tot_gate)
    print('gate_product =', gate_product)
    print('tot_time =', tot_time)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python extract_results.py [result file of ./run_*.sh]')
        exit()
    extract_results(sys.argv[1])
