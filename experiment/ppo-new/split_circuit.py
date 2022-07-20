import quartz
import random
from quartz import PyGraph


def split_qasm(qasm_str: str,
               split_line_cnt: int) -> tuple[str, str, str, str, int]:
    qasm_str_lines: list[str] = qasm_str.split("\n")
    qasm_head: str = "\n".join(qasm_str_lines[:3])
    front_piece: str = ""
    rear_piece: str = ""

    qasm_str_lines = qasm_str_lines[3:]
    if qasm_str_lines[-1] == "":
        qasm_str_lines = qasm_str_lines[:-1]
    line_cnt: int = len(qasm_str_lines)

    if line_cnt <= split_line_cnt:
        return qasm_str, qasm_head, front_piece, rear_piece, 0
    begin: int = random.randint(0, line_cnt - split_line_cnt)
    end: int = begin + split_line_cnt
    split_qasm_segment: str = "\n".join(qasm_str_lines[begin:end])
    split_qasm_str: str = f"{qasm_head}\n{split_qasm_segment}"
    front_piece = "\n".join(qasm_str_lines[:begin])
    rear_piece = "\n".join(qasm_str_lines[end:])
    rest_line_cnt: int = line_cnt - split_line_cnt
    return split_qasm_str, qasm_head, front_piece, rear_piece, rest_line_cnt


def merge_qasm(qasm_str: str, qasm_head: str, front_piece: str,
               rear_piece: str) -> str:
    qasm_str_lines = qasm_str.split("\n")
    mid_piece = "\n".join(qasm_str_lines[3:])
    merged_qasm_str: str = "\n".join(
        [qasm_head, front_piece, mid_piece, rear_piece])
    return merged_qasm_str.replace("\n\n", "\n")


if __name__ == "__main__":
    context = quartz.QuartzContext(gate_set=['h', 'cx', 't', 'tdg', 'x'],
                                   filename='../ecc_set/t_tdg.json.ecc',
                                   no_increase=False,
                                   include_nop=True)
    circ: PyGraph = PyGraph.from_qasm(
        context=context, filename="../t_tdg_circs/barenco_tof_3.qasm")
    qasm_str = circ.to_qasm_str()
    # print(qasm_str)

    qasm_str, head, front, rear, rest_cnt = split_qasm(qasm_str, 10)
    print(f"qasm str: {qasm_str}")
    print(f"head: {head}")
    print(f"front: {front}")
    print(f"rear: {rear}")
    print(rest_cnt)

    qasm_str = merge_qasm(qasm_str, head, front, rear)
    print(qasm_str)