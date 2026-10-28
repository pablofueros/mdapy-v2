"""Evaluate the sequential Numba candidate against mdapy 2.2."""

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from statistics import median
from time import perf_counter
from types import ModuleType
from typing import Any

import numpy as np
import scipy

from benchmarks.scale import (
    GROUNDED_WORKLOADS,
    RANDOM_SEED,
    SCALING_WORKLOADS,
    WORKLOADS,
    make_columns,
)

ROOT = Path(__file__).resolve().parent
BENCHMARKS = ROOT.parents[1]
REPOSITORY = ROOT.parents[2]
ALGORITHMS = ("v2_2", "numba_candidate")
ALGORITHM_PATHS = {
    "v2_2": BENCHMARKS / "algorithms/v2_2.py",
    "numba_candidate": ROOT / "numba_candidate.py",
    "numba_parallel_candidate": ROOT / "numba_parallel_candidate.py",
}
PRIMARY_WORKLOADS = (
    "grounded-franklin-k600",
    "grounded-castro-k1000",
    "grounded-castro-k5000",
    "grounded-ncsu-k1250",
    "grounded-directional-k600",
)
SUITES = {
    "grounded": GROUNDED_WORKLOADS,
    "scaling": SCALING_WORKLOADS,
}
COMPILE_PROBES = {
    "float64": "selections-k5",
    "float32": "dtype-float32",
}


def load_algorithm(name: str) -> ModuleType:
    path = ALGORITHM_PATHS[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def peak_mib() -> float:
    divisor = 2**20 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor


def validate(algorithm_names: tuple[str, ...] = ALGORITHMS) -> None:
    """Require candidates to reproduce every preserved 2.2 reference case."""
    cases = json.loads((BENCHMARKS / "cases.json").read_text())
    baseline = load_algorithm("v2_2")
    candidates = [load_algorithm(name) for name in algorithm_names if name != "v2_2"]
    for dtype in (np.float32, np.float64):
        for case in cases:
            columns = tuple(
                np.asarray(column, dtype=dtype) for column in case["columns"]
            )
            baseline_points, baseline_indices = baseline.select(
                baseline.prepare(columns, case["directional"]),
                case["selections"],
                case["seed_index"],
            )
            expected = case["expected"]["v2_2"]
            if baseline_indices != expected:
                raise AssertionError(
                    f"{case['name']}/{dtype.__name__}: "
                    f"baseline={baseline_indices}, expected={expected}"
                )
            for candidate in candidates:
                candidate_points, candidate_indices = candidate.select(
                    candidate.prepare(columns, case["directional"]),
                    case["selections"],
                    case["seed_index"],
                )
                if candidate_indices != expected:
                    raise AssertionError(
                        f"{case['name']}/{dtype.__name__}/{candidate.SOURCE}: "
                        f"candidate={candidate_indices}, expected={expected}"
                    )
                np.testing.assert_array_equal(candidate_points, baseline_points)


def worker(algorithm_name: str, workload_name: str) -> dict[str, Any]:
    """Measure one isolated process, including a second call for Numba."""
    started = perf_counter()
    algorithm = load_algorithm(algorithm_name)
    module_load_s = perf_counter() - started
    specification = WORKLOADS[workload_name]
    columns, directional, seed_index, data_load_s = make_columns(specification)

    started = perf_counter()
    prepared = algorithm.prepare(columns, directional)
    preparation_s = perf_counter() - started
    selections = int(specification["selections"])

    started = perf_counter()
    points, indices = algorithm.select(prepared, selections, seed_index)
    first_selection_s = perf_counter() - started
    hot_selection_s = first_selection_s
    if algorithm_name != "v2_2":
        started = perf_counter()
        hot_points, hot_indices = algorithm.select(prepared, selections, seed_index)
        hot_selection_s = perf_counter() - started
        if hot_indices != indices:
            raise AssertionError("Numba first and hot calls selected different indices")
        np.testing.assert_array_equal(hot_points, points)

    numba_num_threads = None
    numba_threading_layer = None
    if algorithm_name != "v2_2":
        numba_module: Any = import_module("numba")
        numba_num_threads = numba_module.get_num_threads()
        try:
            numba_threading_layer = numba_module.threading_layer()
        except ValueError:
            pass

    return {
        "algorithm": algorithm_name,
        "source": algorithm.SOURCE,
        "workload": workload_name,
        "rows": len(columns[0]),
        "columns": len(columns),
        "directional": directional,
        "input_dtype": [str(column.dtype) for column in columns],
        "input_mib": sum(column.nbytes for column in columns) / 2**20,
        "work_dtype": str(prepared.data.dtype),
        "output_dtype": str(points.dtype),
        "random_seed": None if specification.get("dataset") else RANDOM_SEED,
        "seed_index": seed_index,
        "selections": selections,
        "indices": indices,
        "module_load_s": module_load_s,
        "data_load_s": data_load_s,
        "preparation_s": preparation_s,
        "first_selection_s": first_selection_s,
        "hot_selection_s": hot_selection_s,
        "first_call_s": preparation_s + first_selection_s,
        "hot_call_s": preparation_s + hot_selection_s,
        "peak_mib": peak_mib(),
        "numba_num_threads": numba_num_threads,
        "numba_threading_layer": numba_threading_layer,
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_worker(
    algorithm_name: str,
    workload_name: str,
    timeout: float,
    cache_dir: Path,
    extra_env: dict[str, str] | None = None,
) -> tuple[dict[str, Any], float]:
    env = os.environ.copy()
    env["NUMBA_CACHE_DIR"] = str(cache_dir)
    env.update(extra_env or {})
    started = perf_counter()
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.experiments.numba.evaluate",
            "--worker",
            algorithm_name,
            "--workload",
            workload_name,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        cwd=REPOSITORY,
    )
    return json.loads(completed.stdout), perf_counter() - started


def compilation_probes(timeout: float, cache_dir: Path) -> dict[str, Any]:
    """Measure uncached compilation and cached first-call overhead by dtype."""
    probes = {}
    for dtype, workload_name in COMPILE_PROBES.items():
        uncached, uncached_process_s = run_worker(
            "numba_candidate", workload_name, timeout, cache_dir
        )
        cached, cached_process_s = run_worker(
            "numba_candidate", workload_name, timeout, cache_dir
        )
        if uncached["indices"] != cached["indices"]:
            raise AssertionError(f"Compilation probe changed {dtype} indices")
        probes[dtype] = {
            "workload": workload_name,
            "uncached_first_selection_s": uncached["first_selection_s"],
            "uncached_hot_selection_s": uncached["hot_selection_s"],
            "uncached_process_s": uncached_process_s,
            "cached_first_selection_s": cached["first_selection_s"],
            "cached_hot_selection_s": cached["hot_selection_s"],
            "cached_process_s": cached_process_s,
        }
    return probes


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
            entries[algorithm_name] = {
                "module_load_median_s": median(
                    float(record["module_load_s"]) for record in selected
                ),
                "preparation_median_s": median(
                    float(record["preparation_s"]) for record in selected
                ),
                "first_selection_median_s": median(
                    float(record["first_selection_s"]) for record in selected
                ),
                "hot_selection_median_s": median(
                    float(record["hot_selection_s"]) for record in selected
                ),
                "peak_median_mib": median(
                    float(record["peak_mib"]) for record in selected
                ),
            }
        baseline_s = entries["v2_2"]["hot_selection_median_s"]
        candidate_s = entries["numba_candidate"]["hot_selection_median_s"]
        entries["hot_speedup"] = baseline_s / candidate_s
        entries["hot_saved_s"] = baseline_s - candidate_s
        entries["candidate_peak_change_percent"] = (
            entries["numba_candidate"]["peak_median_mib"]
            / entries["v2_2"]["peak_median_mib"]
            - 1
        ) * 100
        workloads[workload_name] = entries

    measured_primary = [name for name in PRIMARY_WORKLOADS if name in workloads]
    thresholds = None
    if len(measured_primary) == len(PRIMARY_WORKLOADS):
        speedup_count = sum(
            workloads[name]["hot_speedup"] >= 1.5 for name in measured_primary
        )
        absolute_saving_count = sum(
            workloads[name]["hot_saved_s"] >= 0.25 for name in measured_primary
        )
        excessive_regressions = [
            name
            for name in measured_primary
            if workloads[name]["numba_candidate"]["hot_selection_median_s"]
            > workloads[name]["v2_2"]["hot_selection_median_s"] * 1.1
        ]
        thresholds = {
            "primary_workloads": measured_primary,
            "speedup_at_least_1_5_count": speedup_count,
            "speedup_required_count": 4,
            "saving_at_least_0_25_s_count": absolute_saving_count,
            "saving_required_count": 3,
            "regressions_over_10_percent": excessive_regressions,
            "performance_accepted": speedup_count >= 4
            and absolute_saving_count >= 3
            and not excessive_regressions,
        }
    return {"workloads": workloads, "thresholds": thresholds}


def benchmark(args: argparse.Namespace) -> Path:
    validate()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Choose an empty output directory")
    suite_workloads = SUITES[args.suite]
    workloads = args.workloads or list(suite_workloads)
    if any(name not in suite_workloads for name in workloads):
        raise ValueError(f"Choose workloads from the {args.suite!r} suite")

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
        "random_seed": RANDOM_SEED,
        "suite": args.suite,
        "workloads": {name: WORKLOADS[name] for name in workloads},
        "runs": args.runs,
        "timeout_s": args.timeout,
        "algorithms": {
            name: {
                "file": ALGORITHM_PATHS[name].name,
                "sha256": sha256(ALGORITHM_PATHS[name]),
            }
            for name in ALGORITHMS
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    records = []
    with tempfile.TemporaryDirectory(prefix="mdapy-numba-cache-") as temporary:
        cache_dir = Path(temporary)
        probes = compilation_probes(args.timeout, cache_dir)
        (output / "compilation.json").write_text(json.dumps(probes, indent=2) + "\n")

        for workload_name in workloads:
            expected = None
            for run_number in range(1, args.runs + 1):
                ordered = ALGORITHMS if run_number % 2 else tuple(reversed(ALGORITHMS))
                for algorithm_name in ordered:
                    record, process_s = run_worker(
                        algorithm_name, workload_name, args.timeout, cache_dir
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
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--suite", choices=SUITES, default="grounded")
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--workloads", nargs="+", choices=tuple(WORKLOADS))
    parser.add_argument(
        "--output",
        default=str(ROOT / "results" / datetime.now(UTC).strftime("%Y%m%d-%H%M%S")),
    )
    parser.add_argument(
        "--worker", choices=tuple(ALGORITHM_PATHS), help=argparse.SUPPRESS
    )
    parser.add_argument("--workload", choices=WORKLOADS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.runs < 2 or args.timeout <= 0:
        parser.error("runs must be at least 2 and timeout must be positive")
    if bool(args.worker) != bool(args.workload):
        parser.error("worker and workload must be used together")
    return args


def main() -> None:
    args = parse_args()
    if args.worker:
        print(json.dumps(worker(args.worker, args.workload)))
    elif args.check:
        validate()
        print("Numba candidate reproduced all mdapy 2.2 reference cases.")
    else:
        output = benchmark(args)
        print(f"Raw results and summary written to {output}")


if __name__ == "__main__":
    main()
