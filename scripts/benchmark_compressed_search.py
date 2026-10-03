"""Compare optimizer modes in fresh Linux processes with kernel peak RSS."""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path


def run(binary, mode, circuit, ecc, expansions, timeout):
    with tempfile.TemporaryFile(mode="w+") as output:
        child = subprocess.Popen(
            [str(binary), mode, str(circuit), str(ecc), str(expansions), str(timeout)],
            stdout=output,
            stderr=subprocess.STDOUT,
        )
        _, status, usage = os.wait4(child.pid, 0)
        child.returncode = os.waitstatus_to_exitcode(status)
        output.seek(0)
        log = output.read()
    if child.returncode:
        raise RuntimeError(log)
    summary = next(line for line in log.splitlines() if line.startswith("mode="))
    result = dict(field.split("=", 1) for field in summary.split())
    for key in result:
        if key != "mode":
            result[key] = (
                float(result[key])
                if key in ("seconds", "best_cost")
                else int(result[key])
            )
    result["peak_rss_kib"] = usage.ru_maxrss
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("circuit", type=Path)
    parser.add_argument("ecc", type=Path)
    parser.add_argument(
        "--binary", type=Path, default=Path("build/benchmark_compressed_search")
    )
    parser.add_argument(
        "--expansions",
        type=int,
        default=1000,
        help="0 uses only the wall-clock timeout",
    )
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1 or args.expansions < 0 or args.timeout < 0:
        parser.error("repeats must be positive; expansions and timeout nonnegative")
    results = []
    for repetition in range(args.repeats):
        # Alternate order to reduce systematic warm-cache/order effects.
        modes = ["reference", "compressed"]
        if repetition % 2:
            modes.reverse()
        pair = {
            mode: run(
                args.binary.resolve(),
                mode,
                args.circuit,
                args.ecc,
                args.expansions,
                args.timeout,
            )
            for mode in modes
        }
        if args.expansions:
            for result in pair.values():
                if result["expanded"] != args.expansions:
                    raise RuntimeError(
                        "Search exhausted or timed out before the work budget"
                    )
            for key in (
                "expanded",
                "accepted",
                "peak_candidates",
                "shrinks",
                "popped_hash_digest",
                "best_cost",
                "result_hash",
            ):
                if pair["reference"][key] != pair["compressed"][key]:
                    raise RuntimeError(f"Search modes disagree on {key}: {pair}")
        results.append(pair)
        print(json.dumps({"repetition": repetition + 1, **pair}), flush=True)
    return results


if __name__ == "__main__":
    main()
