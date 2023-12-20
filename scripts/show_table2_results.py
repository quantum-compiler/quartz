import sys


def extract_results(filename):
    with open(filename) as f:
        content = f.readlines()
    records = {}
    for line in content:
        line = line.strip()
        if line.startswith('***'):
            data = line.split()
            if data[2] == 'ReGen:':
                table2_col = 1
            elif data[2] == 'ReGen':
                table2_col = 2 if data[4] == 'ECC' else 3
            else:
                table2_col = 0
            records[(data[1][:-2], int(data[1][-1]), table2_col)] = (data[-2], data[-1])
        elif line.startswith('###'):
            data = line.split()
            if data[2] == 'Rn':
                table3_col = 4
            elif data[2] == 'Verification':
                table3_col = 5
            else:
                table3_col = 6
            records[(data[1][:-2], int(data[1][-1]), table3_col)] = data[-1]
    message = [
        'Column "Original"',
        'Algorithm 1 with only singleton removal (not shown in the submission)',
        'Column "Representative"',
        'Column "Common Subcircuit"',
        '|Rn| (not shown in the submission)',
        'Verification time (s) (not shown in the submission)',
        'Column "Running Time (s)"',
    ]
    nam_n = set()
    for k, v in records.items():
        if k[0] == 'Nam':
            nam_n.add(k[1])
    ibm_n = set()
    for k, v in records.items():
        if k[0] == 'IBM':
            ibm_n.add(k[1])
    rigetti_n = set()
    for k, v in records.items():
        if k[0] == 'Rigetti':
            rigetti_n.add(k[1])
    if len(nam_n) > 0:
        print('- Nam Gate Set:')
        for n in sorted(nam_n):
            print(f'  - Row "n = {n}":')
            v0 = None
            for k, v in sorted(records.items()):
                if k[0] == 'Nam' and k[1] == n:
                    if k[2] < 4:
                        print(f'    - {message[k[2]]}: {v[0]} {v[1]}')
                    else:
                        print(f'    - {message[k[2]]}: {v}')
                    if k[2] == 0:
                        v0 = v
                    elif k[2] == 3 and v0:
                        print(
                            f'    - Column "Overall Reduction": {1 - int(v[0]) / int(v0[0]):.2%} ({1 - int(v[1][1:-1]) / int(v0[1][1:-1]):.2%})'
                        )
    if len(ibm_n) > 0:
        print('- IBM Gate Set:')
        for n in sorted(ibm_n):
            print(f'  - Row "n = {n}":')
            v0 = None
            for k, v in sorted(records.items()):
                if k[0] == 'IBM' and k[1] == n:
                    if k[2] < 4:
                        print(f'    - {message[k[2]]}: {v[0]} {v[1]}')
                    else:
                        print(f'    - {message[k[2]]}: {v}')
                    if k[2] == 0:
                        v0 = v
                    elif k[2] == 3 and v0:
                        print(
                            f'    - Column "Overall Reduction": {1 - int(v[0]) / int(v0[0]):.2%} ({1 - int(v[1][1:-1]) / int(v0[1][1:-1]):.2%})'
                        )
    if len(rigetti_n) > 0:
        print('- Rigetti Gate Set:')
        for n in sorted(rigetti_n):
            print(f'  - Row "n = {n}":')
            v0 = None
            for k, v in sorted(records.items()):
                if k[0] == 'Rigetti' and k[1] == n:
                    if k[2] < 4:
                        print(f'    - {message[k[2]]}: {v[0]} {v[1]}')
                    else:
                        print(f'    - {message[k[2]]}: {v}')
                    if k[2] == 0:
                        v0 = v
                    elif k[2] == 3 and v0:
                        print(
                            f'    - Column "Overall Reduction": {1 - int(v[0]) / int(v0[0]):.2%} ({1 - int(v[1][1:-1]) / int(v0[1][1:-1]):.2%})'
                        )


if __name__ == '__main__':
    fn = None
    if len(sys.argv) != 2:
        fn = 'table2.log'
    else:
        fn = sys.argv[1]
    extract_results(fn)
