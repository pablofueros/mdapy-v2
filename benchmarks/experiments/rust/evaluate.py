"""Evaluate a sequential Rust script against NumPy and Numba candidates."""

import argparse
import json
import os
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from time import perf_counter, time_ns
from typing import Any

import numpy as np

from benchmarks.experiments.compact_results import compact_campaign
from benchmarks.experiments.numba import evaluate as numba_evaluate
from benchmarks.scale import RANDOM_SEED, SUITES, WORKLOADS, make_columns

ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[2]
BENCHMARKS = ROOT.parents[1]
RUST_SOURCE = ROOT / "rust_candidate.rs"
ALGORITHMS = (
    "v2_2",
    "numba_candidate",
    "numba_parallel_candidate",
    "rust_candidate",
)
RUST_MAGIC = b"MDAPYRS1"
HEADER = struct.Struct("<8sB7xQQQQ")
DEFAULT_NUMBA_THREADS = 4


def command_version(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        cwd=REPOSITORY,
    )
    return completed.stdout.strip()


def write_fixture(
    prepared: Any, selections: int, seed_index: int, destination: Path
) -> float:
    """Write prepared 2.2 data to the versioned Rust interchange format."""
    started = perf_counter()
    data = np.ascontiguousarray(prepared.data)
    if data.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise TypeError(f"Unsupported Rust fixture dtype: {data.dtype}")
    dtype_marker = data.dtype.itemsize
    directional = np.zeros(data.shape[1], dtype=np.uint8)
    directional[prepared.directional] = 1
    deviations = np.asarray(prepared.deviations, dtype="<f8")
    little_endian = data.astype(data.dtype.newbyteorder("<"), copy=False)
    with destination.open("wb") as stream:
        stream.write(
            HEADER.pack(
                RUST_MAGIC,
                dtype_marker,
                data.shape[0],
                data.shape[1],
                selections,
                seed_index,
            )
        )
        stream.write(directional.tobytes())
        stream.write(deviations.tobytes())
        stream.write(little_endian.tobytes(order="C"))
    return perf_counter() - started


def run_rust(
    fixture: Path,
    repeats: int,
    timeout: float,
    *,
    source: Path = RUST_SOURCE,
    extra_arguments: tuple[str, ...] = (),
) -> tuple[dict[str, Any], float]:
    command = ["rust-script"]
    command.extend([str(source), str(fixture), str(repeats)])
    command.extend(extra_arguments)
    started = perf_counter()
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=REPOSITORY,
    )
    elapsed = perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"rust-script failed with code {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    return json.loads(completed.stdout), elapsed


def rust_worker(
    workload_name: str,
    timeout: float,
    *,
    algorithm_name: str = "rust_candidate",
    source: Path = RUST_SOURCE,
    extra_arguments: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Measure one Rust CLI process while retaining its internal kernel time."""
    started = perf_counter()
    reference = numba_evaluate.load_algorithm("v2_2")
    module_load_s = perf_counter() - started
    specification = WORKLOADS[workload_name]
    columns, directional, seed_index, data_load_s = make_columns(specification)

    started = perf_counter()
    prepared = reference.prepare(columns, directional)
    preparation_s = perf_counter() - started
    selections = int(specification["selections"])
    with tempfile.TemporaryDirectory(prefix="mdapy-rust-input-") as temporary:
        fixture = Path(temporary) / "input.bin"
        serialization_s = write_fixture(prepared, selections, seed_index, fixture)
        result, rust_cli_s = run_rust(
            fixture,
            2,
            timeout,
            source=source,
            extra_arguments=extra_arguments,
        )

    timings = [float(value) for value in result["selection_s"]]
    if len(timings) != 2:
        raise AssertionError("Rust worker did not return two selection timings")
    indices = [int(value) for value in result["indices"]]
    input_read_s = float(result["input_read_s"])
    rust_overhead_s = max(0.0, rust_cli_s - input_read_s - sum(timings))
    return {
        "algorithm": algorithm_name,
        "source": f"experimental {algorithm_name} based on mdapy 2.2",
        "workload": workload_name,
        "rows": len(columns[0]),
        "columns": len(columns),
        "directional": directional,
        "input_dtype": [str(column.dtype) for column in columns],
        "input_mib": sum(column.nbytes for column in columns) / 2**20,
        "work_dtype": str(prepared.data.dtype),
        "output_dtype": str(prepared.data.dtype),
        "random_seed": None if specification.get("dataset") else RANDOM_SEED,
        "seed_index": seed_index,
        "selections": selections,
        "indices": indices,
        "module_load_s": module_load_s,
        "data_load_s": data_load_s,
        "preparation_s": preparation_s,
        "serialization_s": serialization_s,
        "input_read_s": input_read_s,
        "first_selection_s": timings[0],
        "hot_selection_s": timings[1],
        "rust_cli_s": rust_cli_s,
        "rust_overhead_s": rust_overhead_s,
    }


def run_rust_worker(workload_name: str, timeout: float) -> tuple[dict[str, Any], float]:
    started = perf_counter()
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.experiments.rust.evaluate",
            "--worker",
            "rust_candidate",
            "--workload",
            workload_name,
            "--timeout",
            str(timeout),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=REPOSITORY,
    )
    return json.loads(completed.stdout), perf_counter() - started


def validate_rust(timeout: float) -> None:
    """Require exact Rust indices for every preserved 2.2 reference case."""
    cases = json.loads((BENCHMARKS / "cases.json").read_text())
    reference = numba_evaluate.load_algorithm("v2_2")
    with tempfile.TemporaryDirectory(prefix="mdapy-rust-validation-") as temporary:
        fixture = Path(temporary) / "input.bin"
        for dtype in (np.float32, np.float64):
            for case in cases:
                columns = tuple(
                    np.asarray(column, dtype=dtype) for column in case["columns"]
                )
                prepared = reference.prepare(columns, case["directional"])
                write_fixture(
                    prepared,
                    int(case["selections"]),
                    int(case["seed_index"]),
                    fixture,
                )
                result, _ = run_rust(fixture, 1, timeout)
                actual = [int(value) for value in result["indices"]]
                expected = case["expected"]["v2_2"]
                if actual != expected:
                    raise AssertionError(
                        f"{case['name']}/{dtype.__name__}: "
                        f"rust={actual}, expected={expected}"
                    )


def validate(timeout: float) -> None:
    numba_evaluate.validate(("v2_2", "numba_candidate", "numba_parallel_candidate"))
    validate_rust(timeout)


def compilation_probe(timeout: float) -> dict[str, Any]:
    cases = json.loads((BENCHMARKS / "cases.json").read_text())
    case = cases[0]
    reference = numba_evaluate.load_algorithm("v2_2")
    columns = tuple(np.asarray(column, dtype=np.float64) for column in case["columns"])
    prepared = reference.prepare(columns, case["directional"])
    with tempfile.TemporaryDirectory(prefix="mdapy-rust-compilation-") as temporary:
        temporary_path = Path(temporary)
        fixture = temporary_path / "input.bin"
        cold_source = temporary_path / "rust_candidate_probe.rs"
        cold_source.write_text(
            RUST_SOURCE.read_text() + f"\n// cold compilation probe {time_ns()}\n"
        )
        write_fixture(
            prepared,
            int(case["selections"]),
            int(case["seed_index"]),
            fixture,
        )
        cold, cold_process_s = run_rust(fixture, 1, timeout, source=cold_source)
        cached, cached_process_s = run_rust(fixture, 1, timeout, source=cold_source)
    if cold["indices"] != cached["indices"]:
        raise AssertionError("Rust compilation probe changed selected indices")
    return {
        "workload": f"reference-{case['name']}-float64",
        "cold_process_s": cold_process_s,
        "cold_selection_s": cold["selection_s"][0],
        "cached_process_s": cached_process_s,
        "cached_selection_s": cached["selection_s"][0],
    }


def timing_summary(records: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "preparation_median_s": median(
            float(record["preparation_s"]) for record in records
        ),
        "first_selection_median_s": median(
            float(record["first_selection_s"]) for record in records
        ),
        "hot_selection_median_s": median(
            float(record["hot_selection_s"]) for record in records
        ),
        "process_median_s": median(float(record["process_s"]) for record in records),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    workloads = {}
    for workload_name in dict.fromkeys(str(record["workload"]) for record in records):
        entries = {}
        for algorithm_name in ALGORITHMS:
            selected = [
                record
                for record in records
                if record["workload"] == workload_name
                and record["algorithm"] == algorithm_name
            ]
            entries[algorithm_name] = timing_summary(selected)
        rust_records = [
            record
            for record in records
            if record["workload"] == workload_name
            and record["algorithm"] == "rust_candidate"
        ]
        entries["rust_candidate"].update(
            {
                "serialization_median_s": median(
                    float(record["serialization_s"]) for record in rust_records
                ),
                "input_read_median_s": median(
                    float(record["input_read_s"]) for record in rust_records
                ),
                "rust_cli_median_s": median(
                    float(record["rust_cli_s"]) for record in rust_records
                ),
                "rust_overhead_median_s": median(
                    float(record["rust_overhead_s"]) for record in rust_records
                ),
                "single_cli_estimate_median_s": median(
                    float(record["rust_cli_s"] - record["hot_selection_s"])
                    for record in rust_records
                ),
                "bridge_call_estimate_median_s": median(
                    float(
                        record["preparation_s"]
                        + record["serialization_s"]
                        + record["rust_cli_s"]
                        - record["hot_selection_s"]
                    )
                    for record in rust_records
                ),
            }
        )
        rust_s = entries["rust_candidate"]["hot_selection_median_s"]
        entries["rust_speedup_vs_numpy"] = (
            entries["v2_2"]["hot_selection_median_s"] / rust_s
        )
        entries["rust_speedup_vs_numba_sequential"] = (
            entries["numba_candidate"]["hot_selection_median_s"] / rust_s
        )
        entries["rust_speedup_vs_numba_parallel"] = (
            entries["numba_parallel_candidate"]["hot_selection_median_s"] / rust_s
        )
        workloads[workload_name] = entries
    return {"workloads": workloads}


def write_metadata(
    output: Path, args: argparse.Namespace, workloads: list[str]
) -> None:
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "rust_script": command_version(["rust-script", "--version"]),
        "rustc": command_version(["rustc", "--version"]),
        "cargo": command_version(["cargo", "--version"]),
        "numba_cpu_name": os.environ.get("NUMBA_CPU_NAME"),
        "numba_parallel_threads": args.threads,
        "random_seed": RANDOM_SEED,
        "suite": args.suite,
        "workloads": {name: WORKLOADS[name] for name in workloads},
        "runs": args.runs,
        "timeout_s": args.timeout,
        "algorithms": {
            "v2_2": {
                "file": str(
                    numba_evaluate.ALGORITHM_PATHS["v2_2"].relative_to(REPOSITORY)
                ),
                "sha256": numba_evaluate.sha256(numba_evaluate.ALGORITHM_PATHS["v2_2"]),
            },
            "numba_candidate": {
                "file": str(
                    numba_evaluate.ALGORITHM_PATHS["numba_candidate"].relative_to(
                        REPOSITORY
                    )
                ),
                "sha256": numba_evaluate.sha256(
                    numba_evaluate.ALGORITHM_PATHS["numba_candidate"]
                ),
            },
            "numba_parallel_candidate": {
                "file": str(
                    numba_evaluate.ALGORITHM_PATHS[
                        "numba_parallel_candidate"
                    ].relative_to(REPOSITORY)
                ),
                "sha256": numba_evaluate.sha256(
                    numba_evaluate.ALGORITHM_PATHS["numba_parallel_candidate"]
                ),
            },
            "rust_candidate": {
                "file": str(RUST_SOURCE.relative_to(REPOSITORY)),
                "sha256": numba_evaluate.sha256(RUST_SOURCE),
            },
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def benchmark(args: argparse.Namespace) -> Path:
    validate(args.timeout)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Choose an empty output directory")
    suite_workloads = SUITES[args.suite]
    workloads = args.workloads or list(suite_workloads)
    if any(name not in suite_workloads for name in workloads):
        raise ValueError(f"Choose workloads from the {args.suite!r} suite")
    write_metadata(output, args, workloads)
    (output / "compilation.json").write_text(
        json.dumps(compilation_probe(args.timeout), indent=2) + "\n"
    )

    records = []
    with tempfile.TemporaryDirectory(prefix="mdapy-rust-numba-cache-") as temporary:
        cache_dir = Path(temporary)
        for workload_name in workloads:
            expected = None
            for run_number in range(1, args.runs + 1):
                ordered = ALGORITHMS if run_number % 2 else tuple(reversed(ALGORITHMS))
                for algorithm_name in ordered:
                    if algorithm_name == "rust_candidate":
                        record, process_s = run_rust_worker(workload_name, args.timeout)
                    else:
                        extra_env = (
                            {"NUMBA_NUM_THREADS": str(args.threads)}
                            if algorithm_name == "numba_parallel_candidate"
                            else None
                        )
                        record, process_s = numba_evaluate.run_worker(
                            algorithm_name,
                            workload_name,
                            args.timeout,
                            cache_dir,
                            extra_env=extra_env,
                        )
                    record["run"] = run_number
                    record["process_s"] = process_s
                    if expected is None:
                        expected = record["indices"]
                    elif record["indices"] != expected:
                        raise ValueError(
                            f"Selection discrepancy: {workload_name}/{algorithm_name}"
                        )
                    records.append(record)
                    result = output / f"{workload_name}-{algorithm_name}.jsonl"
                    with result.open("a") as stream:
                        stream.write(json.dumps(record) + "\n")

    (output / "summary.json").write_text(
        json.dumps(summarize(records), indent=2) + "\n"
    )
    compact_campaign(output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--suite", choices=SUITES, default="grounded")
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--threads", type=int, default=DEFAULT_NUMBA_THREADS)
    parser.add_argument("--workloads", nargs="+", choices=tuple(WORKLOADS))
    parser.add_argument(
        "--output",
        default=str(ROOT / "results" / datetime.now(UTC).strftime("%Y%m%d-%H%M%S")),
    )
    parser.add_argument("--worker", choices=("rust_candidate",), help=argparse.SUPPRESS)
    parser.add_argument("--workload", choices=WORKLOADS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    cpu_count = os.cpu_count() or 1
    if args.runs < 2 or args.timeout <= 0:
        parser.error("runs must be at least 2 and timeout must be positive")
    if not 1 <= args.threads <= cpu_count:
        parser.error(f"threads must be between 1 and {cpu_count}")
    if bool(args.worker) != bool(args.workload):
        parser.error("worker and workload must be used together")
    if shutil.which("rust-script") is None:
        parser.error("rust-script is required")
    return args


def main() -> None:
    args = parse_args()
    if args.worker:
        print(json.dumps(rust_worker(args.workload, args.timeout)))
    elif args.check:
        validate(args.timeout)
        print("Rust candidate reproduced all mdapy 2.2 reference cases.")
    else:
        output = benchmark(args)
        print(f"Compact results and summary written to {output}")


if __name__ == "__main__":
    main()
