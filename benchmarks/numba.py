"""Confirm the optional Numba engine through the current public API."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, Literal

import numpy as np
import scipy

from benchmarks.experiments.compact_results import compact_campaign
from benchmarks.scale import RANDOM_SEED, WORKLOADS, make_columns

try:
    import resource
except ImportError:  # pragma: no cover - resource is unavailable on Windows.
    resource = None

ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parent
type Engine = Literal["numpy", "numba"]
ENGINES: tuple[Engine, ...] = ("numpy", "numba")
DEFAULT_WORKLOADS = (
    "columns-d1",
    "grounded-datawave-k100",
    "grounded-castro-k5000",
)
COMPILE_PROBES = {
    "float64": "selections-k5",
    "float32": "dtype-float32",
}
SOURCE_PATHS = (
    REPOSITORY / "src/mdapy/api.py",
    REPOSITORY / "src/mdapy/core.py",
    REPOSITORY / "src/mdapy/numba.py",
)


def peak_mib() -> float | None:
    """Return peak process RSS where the standard resource module is available."""
    if resource is None:
        return None
    divisor = 2**20 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_digest(points: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(points.dtype).encode())
    digest.update(str(points.shape).encode())
    digest.update(points.tobytes())
    return digest.hexdigest()


def public_columns(raw_columns, directional):
    """Adapt shared benchmark inputs to the public column API."""
    from mdapy import DataColumn

    directional_set = set(directional)
    return tuple(
        DataColumn(column, directional=index in directional_set)
        for index, column in enumerate(raw_columns)
    )


def validate() -> None:
    """Require both public engines to reproduce the preserved 2.2 cases."""
    from mdapy import DataColumn, max_diss_alg

    cases = json.loads((ROOT / "cases.json").read_text())
    for dtype in (np.float32, np.float64):
        for case in cases:
            directional = set(case["directional"])
            columns = tuple(
                DataColumn(
                    np.asarray(column, dtype=dtype),
                    directional=index in directional,
                )
                for index, column in enumerate(case["columns"])
            )
            expected = case["expected"]["v2_2"]
            expected_points = None
            for engine in ENGINES:
                points, indices = max_diss_alg(
                    *columns,
                    n_clusters=case["selections"],
                    seed_index=case["seed_index"],
                    engine=engine,
                )
                if indices != expected:
                    raise AssertionError(
                        f"{case['name']}/{dtype.__name__}/{engine}: "
                        f"indices={indices}, expected={expected}"
                    )
                if expected_points is None:
                    expected_points = points
                else:
                    np.testing.assert_array_equal(points, expected_points)


def worker(engine: Engine, workload_name: str) -> dict[str, Any]:
    """Measure two public calls in one isolated process."""
    started = perf_counter()
    from mdapy import max_diss_alg

    module_load_s = perf_counter() - started
    specification = WORKLOADS[workload_name]
    raw_columns, directional, seed_index, data_load_s = make_columns(specification)

    started = perf_counter()
    columns = public_columns(raw_columns, directional)
    data_column_s = perf_counter() - started
    selections = int(specification["selections"])

    started = perf_counter()
    points, indices = max_diss_alg(
        *columns,
        n_clusters=selections,
        seed_index=seed_index,
        engine=engine,
    )
    first_call_s = perf_counter() - started

    started = perf_counter()
    hot_points, hot_indices = max_diss_alg(
        *columns,
        n_clusters=selections,
        seed_index=seed_index,
        engine=engine,
    )
    hot_call_s = perf_counter() - started
    if hot_indices != indices:
        raise AssertionError(f"{engine} first and hot calls selected different indices")
    np.testing.assert_array_equal(hot_points, points)

    numba_num_threads = None
    numba_threading_layer = None
    if engine == "numba":
        numba_module: Any = import_module("numba")
        numba_num_threads = numba_module.get_num_threads()
        numba_threading_layer = numba_module.threading_layer()

    return {
        "engine": engine,
        "source": "mdapy.max_diss_alg public API",
        "workload": workload_name,
        "rows": len(raw_columns[0]),
        "columns": len(raw_columns),
        "directional": directional,
        "input_dtype": [str(column.dtype) for column in raw_columns],
        "input_mib": sum(column.nbytes for column in raw_columns) / 2**20,
        "output_dtype": str(points.dtype),
        "random_seed": None if specification.get("dataset") else RANDOM_SEED,
        "seed_index": seed_index,
        "selections": selections,
        "indices": indices,
        "selected_sha256": selected_digest(points),
        "module_load_s": module_load_s,
        "data_load_s": data_load_s,
        "data_column_s": data_column_s,
        "first_call_s": first_call_s,
        "hot_call_s": hot_call_s,
        "peak_mib": peak_mib(),
        "numba_num_threads": numba_num_threads,
        "numba_threading_layer": numba_threading_layer,
    }


def run_worker(
    engine: Engine,
    workload_name: str,
    timeout: float,
    cache_dir: Path,
) -> tuple[dict[str, Any], float]:
    env = os.environ.copy()
    env["NUMBA_CACHE_DIR"] = str(cache_dir)
    started = perf_counter()
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.numba",
            "--worker",
            engine,
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
    """Measure uncached and cached public Numba calls for both input dtypes."""
    probes = {}
    for dtype, workload_name in COMPILE_PROBES.items():
        uncached, uncached_process_s = run_worker(
            "numba", workload_name, timeout, cache_dir
        )
        cached, cached_process_s = run_worker(
            "numba", workload_name, timeout, cache_dir
        )
        if (
            uncached["indices"] != cached["indices"]
            or uncached["selected_sha256"] != cached["selected_sha256"]
        ):
            raise AssertionError(f"Compilation probe changed {dtype} output")
        probes[dtype] = {
            "workload": workload_name,
            "uncached_first_call_s": uncached["first_call_s"],
            "uncached_hot_call_s": uncached["hot_call_s"],
            "uncached_process_s": uncached_process_s,
            "cached_first_call_s": cached["first_call_s"],
            "cached_hot_call_s": cached["hot_call_s"],
            "cached_process_s": cached_process_s,
            "numba_num_threads": cached["numba_num_threads"],
            "numba_threading_layer": cached["numba_threading_layer"],
        }
    return probes


def timing_summary(records: list[dict[str, Any]]) -> dict[str, float | None]:
    peaks = [float(record["peak_mib"]) for record in records if record["peak_mib"]]
    return {
        "module_load_median_s": median(
            float(record["module_load_s"]) for record in records
        ),
        "data_column_median_s": median(
            float(record["data_column_s"]) for record in records
        ),
        "first_call_median_s": median(
            float(record["first_call_s"]) for record in records
        ),
        "hot_call_median_s": median(float(record["hot_call_s"]) for record in records),
        "process_median_s": median(float(record["process_s"]) for record in records),
        "peak_median_mib": median(peaks) if peaks else None,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    workloads = {}
    for workload_name in dict.fromkeys(str(record["workload"]) for record in records):
        entries = {}
        for engine in ENGINES:
            selected = [
                record
                for record in records
                if record["workload"] == workload_name and record["engine"] == engine
            ]
            entries[engine] = timing_summary(selected)

        numpy_first = entries["numpy"]["first_call_median_s"]
        numba_first = entries["numba"]["first_call_median_s"]
        numpy_hot = entries["numpy"]["hot_call_median_s"]
        numba_hot = entries["numba"]["hot_call_median_s"]
        if (
            not isinstance(numpy_first, float)
            or not isinstance(numba_first, float)
            or not isinstance(numpy_hot, float)
            or not isinstance(numba_hot, float)
        ):
            raise TypeError("Missing public call timing")
        numpy_peak = entries["numpy"]["peak_median_mib"]
        numba_peak = entries["numba"]["peak_median_mib"]
        workloads[workload_name] = {
            **entries,
            "first_call_speedup": numpy_first / numba_first,
            "first_call_saved_s": numpy_first - numba_first,
            "hot_call_speedup": numpy_hot / numba_hot,
            "hot_call_saved_s": numpy_hot - numba_hot,
            "numba_peak_change_percent": (
                (numba_peak / numpy_peak - 1) * 100
                if isinstance(numpy_peak, float) and isinstance(numba_peak, float)
                else None
            ),
        }
    return {"workloads": workloads}


def prepare_output(path: str) -> Path:
    output = Path(path).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Choose an empty output directory")
    return output


def benchmark(args: argparse.Namespace) -> Path:
    validate()
    output = prepare_output(args.output)
    workloads = args.workloads or list(DEFAULT_WORKLOADS)
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "numba": version("numba"),
        "llvmlite": version("llvmlite"),
        "numba_cpu_name": os.environ.get("NUMBA_CPU_NAME"),
        "numba_num_threads_env": os.environ.get("NUMBA_NUM_THREADS"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "random_seed": RANDOM_SEED,
        "workloads": {name: WORKLOADS[name] for name in workloads},
        "runs": args.runs,
        "timeout_s": args.timeout,
        "engines": list(ENGINES),
        "public_api": "mdapy.max_diss_alg",
        "sources": {
            str(path.relative_to(REPOSITORY)): sha256(path) for path in SOURCE_PATHS
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    records = []
    with tempfile.TemporaryDirectory(prefix="mdapy-public-numba-") as temporary:
        cache_dir = Path(temporary)
        probes = compilation_probes(args.timeout, cache_dir)
        (output / "compilation.json").write_text(json.dumps(probes, indent=2) + "\n")

        for workload_name in workloads:
            expected_indices = None
            expected_digest = None
            for run_number in range(1, args.runs + 1):
                ordered = ENGINES if run_number % 2 else tuple(reversed(ENGINES))
                for engine in ordered:
                    record, process_s = run_worker(
                        engine, workload_name, args.timeout, cache_dir
                    )
                    record["run"] = run_number
                    record["process_s"] = process_s
                    if expected_indices is None:
                        expected_indices = record["indices"]
                        expected_digest = record["selected_sha256"]
                    elif (
                        record["indices"] != expected_indices
                        or record["selected_sha256"] != expected_digest
                    ):
                        raise ValueError(
                            f"Selection discrepancy: {workload_name}/{engine}"
                        )
                    records.append(record)
                    result = output / f"{workload_name}-{engine}.jsonl"
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
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--workloads", nargs="+", choices=tuple(WORKLOADS))
    parser.add_argument(
        "--output",
        default=str(ROOT / "results/v2_3-numba-public"),
    )
    parser.add_argument("--worker", choices=ENGINES, help=argparse.SUPPRESS)
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
        print("Public NumPy and Numba engines reproduced all mdapy 2.2 cases.")
    else:
        output = benchmark(args)
        print(f"Compact public-engine results written to {output}")


if __name__ == "__main__":
    main()
