import os

from qiskit import QuantumCircuit, transpile


def main():
    # draw
    input_path = f"./path"
    output_path = f"./pic"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    filename_list = os.listdir(input_path)
    for filename in filename_list:
        whole_filename = os.path.join(input_path, filename)
        barenco_tof_3_circ = QuantumCircuit.from_qasm_file(whole_filename)
        barenco_tof_3_circ.draw(output='mpl', filename=os.path.join(output_path, filename) + ".jpg")


if __name__ == '__main__':
    main()
