import quartz
import random


def split_qasm(qasm_str: str,
               split_line_cnt: int) -> tuple[str, str, str, str]:
    qasm_str_lines: list[str] = qasm_str.split("\n")
    qasm_head: str = "\n".join(qasm_str_lines[:3])
    front_piece: str = ""
    rear_piece: str = ""

    qasm_str_lines = qasm_str_lines[3:]
    if qasm_str_lines[-1] == "":
        qasm_str_lines == qasm_str_lines[:-1]
    line_cnt: int = len(qasm_str_lines)

    if line_cnt <= split_line_cnt:
        return qasm_str, qasm_head, front_piece, rear_piece
    begin: int = random.randint(0, line_cnt - split_line_cnt)
    end: int = begin + split_line_cnt
    split_qasm_segment: str = "\n".join(qasm_str_lines[begin:end])
    split_qasm_str: str = f"{qasm_str}\n{split_qasm_segment}"
    front_piece = "\n".join(qasm_str_lines[:begin])
    rear_piece = "\n".join(qasm_str_lines[end:])
    return split_qasm_str, qasm_head, front_piece, rear_piece


def merge_qasm(qasm_str: str, qasm_head: str, front_piece: str,
               rear_piece: str) -> str:
    qasm_str_lines = qasm_str.split("\n")
    mid_piece = "\n".join(qasm_str_lines[3:])
    merged_qasm_str: str = "\n".join(
        [qasm_head, front_piece, mid_piece, rear_piece])
    return merged_qasm_str.replace("\n\n", "\n")