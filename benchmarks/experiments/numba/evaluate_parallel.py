"""Tune and compare the row-parallel Numba candidate."""

import argparse
import json
import os
import platform
import tempfile
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import scipy

from benchmarks.experiments.compact_results import compact_campaign
from benchmarks.experiments.numba import evaluate as base

ROOT = Path(__file__).resolve().parent
ALGORITHMS = ("v2_2", "numba_candidate", "numba_parallel_candidate")
NUMBA_ALGORITHMS = ("numba_candidate", "numba_parallel_candidate")
DEFAULT_THREAD_COUNTS = (1, 2, 4, 8, 12)
TUNING_WORKLOADS = (
    "grounded-datawave-k20",
    "columns-d30",
    "grounded-castro-k5000",
    "grounded-ncsu-k1250",
    "grounded-directional-k600",
)


def algorithm_metadata(algorithm_names: tuple[str, ...]) -> dict[str, Any]:
    return {
        name: {
            "file": base.ALGORITHM_PATHS[name].name,
            "sha256": base.sha256(base.ALGORITHM_PATHS[name]),
        }
        for name in algorithm_names
    }


def write_metadata(
    output: Path,
    args: argparse.Namespace,
    workloads: list[str],
    algorithm_names: tuple[str, ...],
) -> None:
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "numba": version("numba"),
        "llvmlite": version("llvmlite"),
        "numba_cpu_name": os.environ.get("NUMBA_CPU_NAME"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "random_seed": base.RANDOM_SEED,
        "mode": args.mode,
        "suite": args.suite if args.mode == "comparison" else None,
        "thread_counts": args.thread_counts,
        "comparison_threads": args.threads if args.mode == "comparison" else None,
        "workloads": {name: base.WORKLOADS[name] for name in workloads},
        "runs": args.runs,
        "timeout_s": args.timeout,
        "algorithms": algorithm_metadata(algorithm_names),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def compilation_probes(timeout: float, cache_dir: Path) -> dict[str, Any]:
    """Measure uncached and cached first calls for both Numba candidates."""
    probes = {}
    for algorithm_name in NUMBA_ALGORITHMS:
        entries = {}
        for dtype, workload_name in base.COMPILE_PROBES.items():
            extra_env = (
                {"NUMBA_NUM_THREADS": "1"}
                if algorithm_name == "numba_parallel_candidate"
                else None
            )
            uncached, uncached_process_s = base.run_worker(
                algorithm_name,
                workload_name,
                timeout,
                cache_dir,
                extra_env=extra_env,
            )
            cached, cached_process_s = base.run_worker(
                algorithm_name,
                workload_name,
                timeout,
                cache_dir,
                extra_env=extra_env,
            )
            if uncached["indices"] != cached["indices"]:
                raise AssertionError(
                    f"Compilation probe changed {algorithm_name}/{dtype} indices"
                )
            entries[dtype] = {
                "workload": workload_name,
                "uncached_first_selection_s": uncached["first_selection_s"],
                "uncached_hot_selection_s": uncached["hot_selection_s"],
                "uncached_process_s": uncached_process_s,
                "cached_first_selection_s": cached["first_selection_s"],
                "cached_hot_selection_s": cached["hot_selection_s"],
                "cached_process_s": cached_process_s,
                "numba_num_threads": cached["numba_num_threads"],
                "numba_threading_layer": cached["numba_threading_layer"],
            }
        probes[algorithm_name] = entries
    return probes


def timing_summary(records: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "module_load_median_s": median(
            float(record["module_load_s"]) for record in records
        ),
        "preparation_median_s": median(
            float(record["preparation_s"]) for record in records
        ),
        "first_selection_median_s": median(
            float(record["first_selection_s"]) for record in records
        ),
        "hot_selection_median_s": median(
            float(record["hot_selection_s"]) for record in records
        ),
        "peak_median_mib": median(float(record["peak_mib"]) for record in records),
    }


def summarize_tuning(
    records: list[dict[str, Any]], threads: list[int]
) -> dict[str, Any]:
    workloads = {}
    aggregate: dict[int, list[float]] = {thread_count: [] for thread_count in threads}
    for workload_name in dict.fromkeys(str(record["workload"]) for record in records):
        sequential_records = [
            record
            for record in records
            if record["workload"] == workload_name
            and record["algorithm"] == "numba_candidate"
        ]
        sequential = timing_summary(sequential_records)
        parallel = {}
        for thread_count in threads:
            selected = [
                record
                for record in records
                if record["workload"] == workload_name
                and record["algorithm"] == "numba_parallel_candidate"
                and record["threads"] == thread_count
            ]
            entry = timing_summary(selected)
            entry["speedup_vs_sequential"] = (
                sequential["hot_selection_median_s"] / entry["hot_selection_median_s"]
            )
            aggregate[thread_count].append(entry["speedup_vs_sequential"])
            parallel[str(thread_count)] = entry
        workloads[workload_name] = {
            "numba_sequential": sequential,
            "numba_parallel": parallel,
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
        threads,
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
        numpy_s = entries["v2_2"]["hot_selection_median_s"]
        sequential_s = entries["numba_candidate"]["hot_selection_median_s"]
        parallel_s = entries["numba_parallel_candidate"]["hot_selection_median_s"]
        entries["sequential_speedup_vs_numpy"] = numpy_s / sequential_s
        entries["parallel_speedup_vs_numpy"] = numpy_s / parallel_s
        entries["parallel_speedup_vs_sequential"] = sequential_s / parallel_s
        entries["parallel_saved_vs_sequential_s"] = sequential_s - parallel_s
        workloads[workload_name] = entries
    return {"workloads": workloads}


def prepare_output(path: str) -> Path:
    output = Path(path).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Choose an empty output directory")
    return output


def tune(args: argparse.Namespace) -> Path:
    output = prepare_output(args.output)
    workloads = args.workloads or list(TUNING_WORKLOADS)
    write_metadata(output, args, workloads, NUMBA_ALGORITHMS)
    records = []
    with tempfile.TemporaryDirectory(prefix="mdapy-numba-parallel-") as temporary:
        cache_dir = Path(temporary)
        probes = compilation_probes(args.timeout, cache_dir)
        (output / "compilation.json").write_text(json.dumps(probes, indent=2) + "\n")

        for workload_name in workloads:
            expected = None
            for run_number in range(1, args.runs + 1):
                variants: list[tuple[str, int | None]] = [
                    ("numba_candidate", None),
                    *[
                        ("numba_parallel_candidate", thread_count)
                        for thread_count in args.thread_counts
                    ],
                ]
                if run_number % 2 == 0:
                    variants.reverse()
                for algorithm_name, thread_count in variants:
                    extra_env = (
                        {"NUMBA_NUM_THREADS": str(thread_count)}
                        if thread_count is not None
                        else None
                    )
                    record, process_s = base.run_worker(
                        algorithm_name,
                        workload_name,
                        args.timeout,
                        cache_dir,
                        extra_env=extra_env,
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

    summary = summarize_tuning(records, args.thread_counts)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    compact_campaign(output)
    return output


def compare(args: argparse.Namespace) -> Path:
    output = prepare_output(args.output)
    suite_workloads = base.SUITES[args.suite]
    workloads = args.workloads or list(suite_workloads)
    if any(name not in suite_workloads for name in workloads):
        raise ValueError(f"Choose workloads from the {args.suite!r} suite")
    write_metadata(output, args, workloads, ALGORITHMS)
    records = []
    with tempfile.TemporaryDirectory(prefix="mdapy-numba-parallel-") as temporary:
        cache_dir = Path(temporary)
        probes = compilation_probes(args.timeout, cache_dir)
        (output / "compilation.json").write_text(json.dumps(probes, indent=2) + "\n")

        for workload_name in workloads:
            expected = None
            for run_number in range(1, args.runs + 1):
                ordered = ALGORITHMS if run_number % 2 else tuple(reversed(ALGORITHMS))
                for algorithm_name in ordered:
                    extra_env = (
                        {"NUMBA_NUM_THREADS": str(args.threads)}
                        if algorithm_name == "numba_parallel_candidate"
                        else None
                    )
                    record, process_s = base.run_worker(
                        algorithm_name,
                        workload_name,
                        args.timeout,
                        cache_dir,
                        extra_env=extra_env,
                    )
                    record["run"] = run_number
                    record["threads"] = (
                        args.threads
                        if algorithm_name == "numba_parallel_candidate"
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
    parser.add_argument("--mode", choices=("tuning", "comparison"), required=True)
    parser.add_argument("--suite", choices=base.SUITES, default="grounded")
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--thread-counts", nargs="+", type=int)
    parser.add_argument("--workloads", nargs="+", choices=tuple(base.WORKLOADS))
    parser.add_argument("--output", required=True)
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
    return args


def main() -> None:
    args = parse_args()
    base.validate(ALGORITHMS)
    output = tune(args) if args.mode == "tuning" else compare(args)
    print(f"Compact results and summary written to {output}")


if __name__ == "__main__":
    main()
