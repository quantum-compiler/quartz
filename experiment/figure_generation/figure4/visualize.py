import os

from qiskit import QuantumCircuit, transpile


def main():
    input_path = "./path_7"
    filename_list = os.listdir(input_path)
    for filename in filename_list:
        whole_filename = os.path.join(input_path, filename)
        barenco_tof_3_circ = QuantumCircuit.from_qasm_file(whole_filename)
        barenco_tof_3_circ.draw(output='mpl', filename=whole_filename + ".jpg")


if __name__ == '__main__':
    main()
