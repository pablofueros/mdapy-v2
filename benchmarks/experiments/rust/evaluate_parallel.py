"""Tune and compare the row-parallel Rust candidate."""

import argparse
import json
import os
import platform
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
from benchmarks.experiments.rust import evaluate as base
from benchmarks.scale import RANDOM_SEED, SUITES, WORKLOADS

ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[2]
BENCHMARKS = ROOT.parents[1]
PARALLEL_SOURCE = ROOT / "rust_parallel_candidate.rs"
RAYON_REQUIREMENT = ">=1.10, <2"
RAYON_VERSION = "1.12.0"
RAYON_CORE_VERSION = "1.13.0"
ALGORITHMS = (
    "v2_2",
    "numba_candidate",
    "numba_parallel_candidate",
    "rust_candidate",
    "rust_parallel_candidate",
)
DEFAULT_THREAD_COUNTS = (1, 2, 4, 8, 12)
TUNING_WORKLOADS = (
    "grounded-datawave-k20",
    "columns-d30",
    "grounded-castro-k5000",
    "grounded-ncsu-k1250",
    "grounded-directional-k600",
)


def run_parallel_worker(
    workload_name: str, timeout: float, threads: int
) -> tuple[dict[str, Any], float]:
    started = perf_counter()
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.experiments.rust.evaluate_parallel",
            "--worker",
            "rust_parallel_candidate",
            "--workload",
            workload_name,
            "--threads",
            str(threads),
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


def validate_parallel(timeout: float, thread_counts: list[int]) -> None:
    cases = json.loads((BENCHMARKS / "cases.json").read_text())
    reference = numba_evaluate.load_algorithm("v2_2")
    with tempfile.TemporaryDirectory(prefix="mdapy-rust-parallel-validation-") as temp:
        fixture = Path(temp) / "input.bin"
        for dtype in (np.float32, np.float64):
            for case in cases:
                columns = tuple(
                    np.asarray(column, dtype=dtype) for column in case["columns"]
                )
                prepared = reference.prepare(columns, case["directional"])
                base.write_fixture(
                    prepared,
                    int(case["selections"]),
                    int(case["seed_index"]),
                    fixture,
                )
                expected = case["expected"]["v2_2"]
                for threads in thread_counts:
                    result, _ = base.run_rust(
                        fixture,
                        1,
                        timeout,
                        source=PARALLEL_SOURCE,
                        extra_arguments=(str(threads),),
                    )
                    actual = [int(value) for value in result["indices"]]
                    if actual != expected:
                        raise AssertionError(
                            f"{case['name']}/{dtype.__name__}/threads={threads}: "
                            f"rust={actual}, expected={expected}"
                        )


def validate(timeout: float, thread_counts: list[int]) -> None:
    base.validate(timeout)
    validate_parallel(timeout, thread_counts)


def compilation_probe(timeout: float) -> dict[str, Any]:
    cases = json.loads((BENCHMARKS / "cases.json").read_text())
    case = cases[0]
    reference = numba_evaluate.load_algorithm("v2_2")
    columns = tuple(np.asarray(column, dtype=np.float64) for column in case["columns"])
    prepared = reference.prepare(columns, case["directional"])
    with tempfile.TemporaryDirectory(prefix="mdapy-rust-parallel-compile-") as temp:
        temporary = Path(temp)
        fixture = temporary / "input.bin"
        cold_source = temporary / "rust_parallel_candidate_probe.rs"
        cold_source.write_text(
            PARALLEL_SOURCE.read_text() + f"\n// cold compilation probe {time_ns()}\n"
        )
        base.write_fixture(
            prepared,
            int(case["selections"]),
            int(case["seed_index"]),
            fixture,
        )
        cold, cold_process_s = base.run_rust(
            fixture,
            1,
            timeout,
            source=cold_source,
            extra_arguments=("1",),
        )
        cached, cached_process_s = base.run_rust(
            fixture,
            1,
            timeout,
            source=cold_source,
            extra_arguments=("1",),
        )
    if cold["indices"] != cached["indices"]:
        raise AssertionError("Parallel Rust compilation changed selected indices")
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


def summarize_tuning(
    records: list[dict[str, Any]], thread_counts: list[int]
) -> dict[str, Any]:
    workloads = {}
    aggregate: dict[int, list[float]] = {
        thread_count: [] for thread_count in thread_counts
    }
    for workload_name in dict.fromkeys(str(record["workload"]) for record in records):
        sequential_records = [
            record
            for record in records
            if record["workload"] == workload_name
            and record["algorithm"] == "rust_candidate"
        ]
        sequential = timing_summary(sequential_records)
        parallel = {}
        for thread_count in thread_counts:
            selected = [
                record
                for record in records
                if record["workload"] == workload_name
                and record["algorithm"] == "rust_parallel_candidate"
                and record["threads"] == thread_count
            ]
            entry = timing_summary(selected)
            entry["speedup_vs_sequential"] = (
                sequential["hot_selection_median_s"] / entry["hot_selection_median_s"]
            )
            aggregate[thread_count].append(entry["speedup_vs_sequential"])
            parallel[str(thread_count)] = entry
        workloads[workload_name] = {
            "rust_sequential": sequential,
            "rust_parallel": parallel,
        }

    thread_summary = {
        str(thread_count): {
            "median_speedup_vs_sequential": median(speedups),
            "minimum_speedup_vs_sequential": min(speedups),
            "maximum_speedup_vs_sequential": max(speedups),
            "faster_workload_count": sum(speedup > 1 for speedup in speedups),
            "workload_count": len(speedups),
        }
        for thread_count, speedups in aggregate.items()
    }
    recommended = max(
        thread_counts,
        key=lambda thread_count: thread_summary[str(thread_count)][
            "median_speedup_vs_sequential"
        ],
    )
    return {
        "workloads": workloads,
        "threads": thread_summary,
        "recommended_threads_for_this_environment": recommended,
    }


def summarize_comparison(records: list[dict[str, Any]]) -> dict[str, Any]:
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
        rust_parallel_s = entries["rust_parallel_candidate"]["hot_selection_median_s"]
        for algorithm_name in ALGORITHMS[:-1]:
            entries[f"rust_parallel_speedup_vs_{algorithm_name}"] = (
                entries[algorithm_name]["hot_selection_median_s"] / rust_parallel_s
            )
        workloads[workload_name] = entries
    return {"workloads": workloads}


def algorithm_metadata() -> dict[str, Any]:
    paths = {
        "v2_2": numba_evaluate.ALGORITHM_PATHS["v2_2"],
        "numba_candidate": numba_evaluate.ALGORITHM_PATHS["numba_candidate"],
        "numba_parallel_candidate": numba_evaluate.ALGORITHM_PATHS[
            "numba_parallel_candidate"
        ],
        "rust_candidate": base.RUST_SOURCE,
        "rust_parallel_candidate": PARALLEL_SOURCE,
    }
    return {
        name: {
            "file": str(path.relative_to(REPOSITORY)),
            "sha256": numba_evaluate.sha256(path),
        }
        for name, path in paths.items()
    }


def write_metadata(
    output: Path,
    args: argparse.Namespace,
    workloads: list[str],
    algorithm_names: tuple[str, ...],
) -> None:
    algorithms = algorithm_metadata()
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "rust_script": base.command_version(["rust-script", "--version"]),
        "rustc": base.command_version(["rustc", "--version"]),
        "cargo": base.command_version(["cargo", "--version"]),
        "rayon_requirement": RAYON_REQUIREMENT,
        "rayon": RAYON_VERSION,
        "rayon_core": RAYON_CORE_VERSION,
        "numba_cpu_name": os.environ.get("NUMBA_CPU_NAME"),
        "numba_parallel_threads": args.numba_threads,
        "random_seed": RANDOM_SEED,
        "mode": args.mode,
        "suite": args.suite if args.mode == "comparison" else None,
        "thread_counts": args.thread_counts,
        "comparison_threads": args.threads if args.mode == "comparison" else None,
        "workloads": {name: WORKLOADS[name] for name in workloads},
        "runs": args.runs,
        "timeout_s": args.timeout,
        "algorithms": {name: algorithms[name] for name in algorithm_names},
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def prepare_output(path: str) -> Path:
    output = Path(path).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Choose an empty output directory")
    return output


def tune(args: argparse.Namespace) -> Path:
    output = prepare_output(args.output)
    workloads = args.workloads or list(TUNING_WORKLOADS)
    write_metadata(
        output,
        args,
        workloads,
        ("rust_candidate", "rust_parallel_candidate"),
    )
    (output / "compilation.json").write_text(
        json.dumps(compilation_probe(args.timeout), indent=2) + "\n"
    )
    records = []
    for workload_name in workloads:
        expected = None
        for run_number in range(1, args.runs + 1):
            variants: list[tuple[str, int | None]] = [
                ("rust_candidate", None),
                *[
                    ("rust_parallel_candidate", thread_count)
                    for thread_count in args.thread_counts
                ],
            ]
            if run_number % 2 == 0:
                variants.reverse()
            for algorithm_name, thread_count in variants:
                if algorithm_name == "rust_candidate":
                    record, process_s = base.run_rust_worker(
                        workload_name, args.timeout
                    )
                else:
                    if thread_count is None:
                        raise AssertionError("Parallel Rust requires a thread count")
                    record, process_s = run_parallel_worker(
                        workload_name, args.timeout, thread_count
                    )
                record["run"] = run_number
                record["threads"] = thread_count
                record["process_s"] = process_s
                if expected is None:
                    expected = record["indices"]
                elif record["indices"] != expected:
                    raise ValueError(
                        f"Selection discrepancy: {workload_name}/"
                        f"{algorithm_name}/threads={thread_count}"
                    )
                records.append(record)
                suffix = (
                    "sequential"
                    if thread_count is None
                    else f"parallel-t{thread_count}"
                )
                result = output / f"{workload_name}-{suffix}.jsonl"
                with result.open("a") as stream:
                    stream.write(json.dumps(record) + "\n")

    (output / "summary.json").write_text(
        json.dumps(summarize_tuning(records, args.thread_counts), indent=2) + "\n"
    )
    compact_campaign(output)
    return output


def compare(args: argparse.Namespace) -> Path:
    output = prepare_output(args.output)
    suite_workloads = SUITES[args.suite]
    workloads = args.workloads or list(suite_workloads)
    if any(name not in suite_workloads for name in workloads):
        raise ValueError(f"Choose workloads from the {args.suite!r} suite")
    write_metadata(output, args, workloads, ALGORITHMS)
    (output / "compilation.json").write_text(
        json.dumps(compilation_probe(args.timeout), indent=2) + "\n"
    )

    records = []
    with tempfile.TemporaryDirectory(prefix="mdapy-rust-parallel-numba-") as temp:
        cache_dir = Path(temp)
        for workload_name in workloads:
            expected = None
            for run_number in range(1, args.runs + 1):
                ordered = ALGORITHMS if run_number % 2 else tuple(reversed(ALGORITHMS))
                for algorithm_name in ordered:
                    if algorithm_name == "rust_candidate":
                        record, process_s = base.run_rust_worker(
                            workload_name, args.timeout
                        )
                    elif algorithm_name == "rust_parallel_candidate":
                        record, process_s = run_parallel_worker(
                            workload_name, args.timeout, args.threads
                        )
                    else:
                        extra_env = (
                            {"NUMBA_NUM_THREADS": str(args.numba_threads)}
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
                    record["threads"] = (
                        args.threads
                        if algorithm_name == "rust_parallel_candidate"
                        else None
                    )
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
        json.dumps(summarize_comparison(records), indent=2) + "\n"
    )
    compact_campaign(output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--mode", choices=("tuning", "comparison"))
    parser.add_argument("--suite", choices=SUITES, default="grounded")
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--numba-threads", type=int, default=4)
    parser.add_argument("--thread-counts", nargs="+", type=int)
    parser.add_argument("--workloads", nargs="+", choices=tuple(WORKLOADS))
    parser.add_argument("--output")
    parser.add_argument(
        "--worker", choices=("rust_parallel_candidate",), help=argparse.SUPPRESS
    )
    parser.add_argument("--workload", choices=WORKLOADS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    cpu_count = os.cpu_count() or 1
    if args.thread_counts is None:
        args.thread_counts = [
            thread_count
            for thread_count in DEFAULT_THREAD_COUNTS
            if thread_count <= cpu_count
        ]
    if args.threads is None:
        args.threads = min(4, cpu_count)
    if args.runs < 2 or args.timeout <= 0:
        parser.error("runs must be at least 2 and timeout must be positive")
    if (
        not args.thread_counts
        or args.thread_counts != sorted(set(args.thread_counts))
        or min(args.thread_counts) < 1
        or max(args.thread_counts) > cpu_count
    ):
        parser.error(f"thread counts must be unique, increasing, and <= {cpu_count}")
    if not 1 <= args.threads <= cpu_count:
        parser.error(f"threads must be between 1 and {cpu_count}")
    if not 1 <= args.numba_threads <= cpu_count:
        parser.error(f"numba threads must be between 1 and {cpu_count}")
    if bool(args.worker) != bool(args.workload):
        parser.error("worker and workload must be used together")
    if (
        not args.worker
        and not args.check
        and (args.mode is None or args.output is None)
    ):
        parser.error("mode and output are required unless check or worker is used")
    return args


def main() -> None:
    args = parse_args()
    if args.worker:
        print(
            json.dumps(
                base.rust_worker(
                    args.workload,
                    args.timeout,
                    algorithm_name="rust_parallel_candidate",
                    source=PARALLEL_SOURCE,
                    extra_arguments=(str(args.threads),),
                )
            )
        )
    elif args.check:
        validate(args.timeout, args.thread_counts)
        print("Parallel Rust reproduced all mdapy 2.2 reference cases.")
    else:
        validate(args.timeout, args.thread_counts)
        output = tune(args) if args.mode == "tuning" else compare(args)
        print(f"Compact results and summary written to {output}")


if __name__ == "__main__":
    main()
