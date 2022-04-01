import sys
from natsort import natsorted


def extract_results(filename):
    with open(filename) as f:
        content = f.readlines()
    flag = False
    tot_time = 0
    tot_gate = 0
    gate_product = 1
    key = ''
    result = {}
    for line in content:
        line = line.strip()
        data = line.split()
        if flag:
            pos = line.find(':')
            pos2 = line.find(',', pos)
            pos3 = line.find('s', pos2)
            val = line[pos + 2:pos2]
            result[key] = val
            tot_gate += int(line[pos + 2:pos2])
            gate_product *= int(line[pos + 2:pos2])
            tot_time += float(line[pos2 + 2:pos3])
        if line.startswith('Optimization'):
            flag = True
            pos = line.find('.qasm')
            pos2 = line.rfind(' ', 0, pos)
            key = line[pos2 + 1:pos]
        else:
            flag = False
        if len(data) >= 2 and data[1] == 'Timeout.':
            key = data[0].split('.')[0]
            val = data[-1]
            result[key] = val
            tot_gate += int(data[-1])
            gate_product *= int(data[-1])
            tot_time += 86400  # 1-day timeout
    for k, v in natsorted(result.items()):
        print(k.ljust(15), v)
    print('tot_gate =', tot_gate)
    print('num_circuits =', len(result))
    print('geomean_gatecount =', gate_product ** (1 / len(result)))
    print('tot_time =', tot_time)
    for k, v in natsorted(result.items()):
        print(v)  # easy paste to google doc


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python extract_results.py [result file of ./run_*.sh]')
        exit()
    extract_results(sys.argv[1])
