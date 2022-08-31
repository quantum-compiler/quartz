import os
import re
import sys

path = r'./Arithmetic_and_Toffoli'
path2 = r'./qasm_output'


def gate_rename(string):
    if string == 'not':
        return 'x'
    else:
        return string.lower()


def process_line(line):
    dagger = False
    target = re.findall(r'QGate\["(.*?)"\]\((.*?)\)', line)
    if len(target) == 0:
        target = re.findall(r'QGate\["(.*?)"\]\*\((.*?)\)', line)
        dagger = True
    if 'controls=' in line:
        control = re.findall(r'controls=\[(.*?)\]', line)
        control_q = re.findall(r'\+(\d*)', control[0])
        control_split = control[0].split(',')
        if len(control_q) != len(control_split):
            print("warning: " + line)
        output_control = ','.join([' q[' + e + ']' for e in control_q])
        if target[0][0] == 'Z' and len(control_q) == 2:
            a, b, c = control_q[0], control_q[1], target[0][1]
            output = f'cx q[{b}], q[{c}];\ntdg q[{c}];\ncx q[{a}], q[{c}];\nt q[{c}];\n cx q[{b}], q[{c}];\ntdg q[{c}];\n cx q[{a}], q[{c}];\n cx q[{a}], q[{b}];\ntdg q[{b}];\n cx q[{a}], q[{b}];\nt q[{a}];\n t q[{b}];\n t q[{c}];'
        else:
            output = (
                'c' * len(control_q)
                + gate_rename(target[0][0])
                + output_control
                + ', q['
                + target[0][1]
                + '];'
            )
    else:
        output = (
            gate_rename(target[0][0])
            + ('dg' if dagger else '')
            + ' q['
            + target[0][1]
            + '];'
        )
    return output


for file in os.listdir(path):
    filename = os.path.join(path, file)
    outputname = os.path.join(path2, file)
    print(filename)
    if os.path.isfile(filename):
        input_f = open(filename, "r")
        output_f = open(outputname + '.qasm', "w")
        output_f.write("OPENQASM 2.0;\n")
        output_f.write('include "qelib1.inc";\n')

        lines = input_f.readlines()
        num_q = len(re.findall(r" (.*?):Qbit", lines[0]))
        output_f.write('qreg q[' + str(num_q) + '];\n')

        for line in lines[1:-1]:
            output_line = process_line(line)
            output_f.write(output_line + '\n')
        input_f.close()
        output_f.close()
