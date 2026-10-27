"""Measure mdapy 2.1 and 2.2 across scaling workloads."""

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import resource
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from time import perf_counter
from types import ModuleType
from typing import Any

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "examples/data/datawave.npz"
ALGORITHMS = ("v2_1", "v2_2")
RANDOM_SEED = 20261020

SCALING_WORKLOADS: dict[str, dict[str, Any]] = {
    "datawave-k5": {"dataset": "datawave", "selections": 5},
    "datawave-k20": {"dataset": "datawave", "selections": 20},
    "datawave-k100": {"dataset": "datawave", "selections": 100},
    "rows-n10000": {"rows": 10_000, "columns": 3, "selections": 20},
    "rows-n100000": {"rows": 100_000, "columns": 3, "selections": 20},
    "rows-n1000000": {"rows": 1_000_000, "columns": 3, "selections": 20},
    "columns-d1": {"rows": 100_000, "columns": 1, "selections": 20},
    "columns-d3": {"rows": 100_000, "columns": 3, "selections": 20},
    "columns-d10": {"rows": 100_000, "columns": 10, "selections": 20},
    "columns-d30": {"rows": 100_000, "columns": 30, "selections": 20},
    "selections-k5": {"rows": 100_000, "columns": 3, "selections": 5},
    "selections-k20": {"rows": 100_000, "columns": 3, "selections": 20},
    "selections-k100": {"rows": 100_000, "columns": 3, "selections": 100},
    "selections-k500": {"rows": 100_000, "columns": 3, "selections": 500},
    "dtype-float32": {
        "rows": 100_000,
        "columns": 3,
        "selections": 20,
        "dtype": "float32",
    },
    "dtype-float64": {
        "rows": 100_000,
        "columns": 3,
        "selections": 20,
        "dtype": "float64",
    },
    "directional-none": {"rows": 100_000, "columns": 3, "selections": 20},
    "directional-one": {
        "rows": 100_000,
        "columns": 3,
        "selections": 20,
        "directional": 1,
    },
    "directional-all": {
        "rows": 100_000,
        "columns": 3,
        "selections": 20,
        "directional": 3,
    },
}

GROUNDED_WORKLOADS: dict[str, dict[str, Any]] = {
    "grounded-datawave-k20": {
        "dataset": "datawave",
        "selections": 20,
    },
    "grounded-datawave-k100": {
        "dataset": "datawave",
        "selections": 100,
    },
    "grounded-franklin-k600": {
        "rows": 87_664,
        "columns": 3,
        "selections": 600,
    },
    "grounded-castro-k100": {
        "rows": 121_968,
        "columns": 3,
        "selections": 100,
    },
    "grounded-castro-k1000": {
        "rows": 121_968,
        "columns": 3,
        "selections": 1_000,
    },
    "grounded-castro-k5000": {
        "rows": 121_968,
        "columns": 3,
        "selections": 5_000,
    },
    "grounded-ncsu-k1250": {
        "rows": 1_000_000,
        "columns": 3,
        "selections": 1_250,
    },
    "grounded-tudelft-k1000": {
        "rows": 10_000,
        "columns": 3,
        "selections": 1_000,
    },
    "grounded-directional-k600": {
        "rows": 87_664,
        "columns": 3,
        "selections": 600,
        "directional": 1,
    },
}

WORKLOADS = SCALING_WORKLOADS | GROUNDED_WORKLOADS
SUITES = {
    "scaling": SCALING_WORKLOADS,
    "grounded": GROUNDED_WORKLOADS,
}


def load_algorithm(name: str) -> ModuleType:
    path = ROOT / "algorithms" / f"{name}.py"
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


def make_columns(specification: dict[str, Any]):
    started = perf_counter()
    if specification.get("dataset") == "datawave":
        with np.load(DATA) as dataset:
            columns = tuple(dataset[name] for name in ("hs", "tp", "dir"))
        directional = [2]
        seed_index = int(np.argmax(columns[1]))
    else:
        rows = int(specification["rows"])
        column_count = int(specification["columns"])
        dtype = np.dtype(str(specification.get("dtype", "float64")))
        directional_count = int(specification.get("directional", 0))
        rng = np.random.default_rng(RANDOM_SEED)
        columns = tuple(
            (
                rng.uniform(0, 360, rows).astype(dtype)
                if index >= column_count - directional_count
                else rng.normal(size=rows).astype(dtype)
            )
            for index in range(column_count)
        )
        directional = list(range(column_count - directional_count, column_count))
        seed_index = 0
    return columns, directional, seed_index, perf_counter() - started


def worker(algorithm_name: str, workload_name: str) -> dict[str, Any]:
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
    selection_s = perf_counter() - started

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
        "selection_s": selection_s,
        "call_s": preparation_s + selection_s,
        "peak_mib": peak_mib(),
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {}
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
                "preparation_median_s": median(
                    float(record["preparation_s"]) for record in selected
                ),
                "selection_median_s": median(
                    float(record["selection_s"]) for record in selected
                ),
                "call_median_s": median(float(record["call_s"]) for record in selected),
                "peak_median_mib": median(
                    float(record["peak_mib"]) for record in selected
                ),
                "selection_min_s": min(
                    float(record["selection_s"]) for record in selected
                ),
                "selection_max_s": max(
                    float(record["selection_s"]) for record in selected
                ),
            }
        baseline = entries["v2_1"]
        version_2_2 = entries["v2_2"]
        entries["selection_speedup"] = (
            baseline["selection_median_s"] / version_2_2["selection_median_s"]
        )
        entries["selection_saved_s"] = (
            baseline["selection_median_s"] - version_2_2["selection_median_s"]
        )
        entries["call_speedup"] = (
            baseline["call_median_s"] / version_2_2["call_median_s"]
        )
        entries["peak_change_percent"] = (
            (version_2_2["peak_median_mib"] / baseline["peak_median_mib"]) - 1
        ) * 100
        summary[workload_name] = entries
    return summary


def benchmark(args: argparse.Namespace) -> Path:
    from run import validate

    validate()
    suite_workloads = SUITES[args.suite]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Choose an empty output directory")
    workloads = args.workloads or list(suite_workloads)
    if any(name not in suite_workloads for name in workloads):
        raise ValueError(f"Choose workloads from the {args.suite!r} suite")
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
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
                "file": f"{name}.py",
                "sha256": sha256(ROOT / "algorithms" / f"{name}.py"),
            }
            for name in ALGORITHMS
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    records = []
    script = Path(__file__).resolve()
    for workload_name in workloads:
        expected = None
        for run_number in range(1, args.runs + 1):
            ordered = ALGORITHMS if run_number % 2 else tuple(reversed(ALGORITHMS))
            for algorithm_name in ordered:
                started = perf_counter()
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(script),
                        "--worker",
                        algorithm_name,
                        "--workload",
                        workload_name,
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=args.timeout,
                )
                record = json.loads(completed.stdout)
                record["run"] = run_number
                record["process_s"] = perf_counter() - started
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
    parser.add_argument("--suite", choices=SUITES, default="scaling")
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--workloads", nargs="+", choices=WORKLOADS)
    parser.add_argument(
        "--output",
        default=str(
            ROOT / "results" / datetime.now(UTC).strftime("scale-%Y%m%d-%H%M%S")
        ),
    )
    parser.add_argument("--worker", choices=ALGORITHMS, help=argparse.SUPPRESS)
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
    else:
        output = benchmark(args)
        print(f"Raw results and summary written to {output}")


if __name__ == "__main__":
    main()
