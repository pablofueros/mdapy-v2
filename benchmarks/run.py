"""Validate and benchmark the available algorithm references."""

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
from time import perf_counter
from types import ModuleType

import numpy as np
import scipy

from benchmarks.experiments.compact_results import compact_campaign

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "examples/data/datawave.npz"
ALGORITHM_NAMES = ("v2_0", "v2_1", "v2_2")
ALGORITHMS = tuple(ROOT / "algorithms" / f"{name}.py" for name in ALGORITHM_NAMES)


def load(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate() -> None:
    cases = json.loads((ROOT / "cases.json").read_text())
    for path in ALGORITHMS:
        algorithm = load(path)
        for case in cases:
            columns = tuple(
                np.asarray(column, dtype=np.float64) for column in case["columns"]
            )
            prepared = algorithm.prepare(columns, case["directional"])
            points, indices = algorithm.select(
                prepared, case["selections"], case["seed_index"]
            )
            expected = case["expected"][path.stem]
            if indices != expected:
                raise AssertionError(
                    f"{path.stem}/{case['name']}: {indices} != {expected}"
                )
            source = np.column_stack(list(columns)).astype(points.dtype)
            np.testing.assert_array_equal(points, source[indices])


def peak_mib() -> float:
    divisor = 2**20 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor


def worker(algorithm_name: str, selections: int) -> dict[str, object]:
    started = perf_counter()
    algorithm = load(ROOT / "algorithms" / f"{algorithm_name}.py")
    module_load_s = perf_counter() - started

    started = perf_counter()
    with np.load(DATA) as dataset:
        columns = tuple(dataset[name] for name in ("hs", "tp", "dir"))
    seed_index = int(np.argmax(columns[1]))
    data_load_s = perf_counter() - started

    started = perf_counter()
    prepared = algorithm.prepare(columns, [2])
    preparation_s = perf_counter() - started

    started = perf_counter()
    points, indices = algorithm.select(prepared, selections, seed_index)
    selection_s = perf_counter() - started

    return {
        "algorithm": algorithm_name,
        "source": algorithm.SOURCE,
        "rows": len(columns[0]),
        "columns": len(columns),
        "input_dtype": [str(column.dtype) for column in columns],
        "work_dtype": str(prepared.data.dtype),
        "output_dtype": str(points.dtype),
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


def benchmark(args: argparse.Namespace) -> Path:
    validate()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Choose an empty output directory")

    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "dataset": str(DATA.relative_to(ROOT.parent)),
        "dataset_sha256": sha256(DATA),
        "algorithms": {
            path.stem: {"file": path.name, "sha256": sha256(path)}
            for path in ALGORITHMS
        },
        "counts": args.counts,
        "runs": args.runs,
        "timeout_s": args.timeout,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    for selections in args.counts:
        for run_number in range(1, args.runs + 1):
            paths = ALGORITHMS if run_number % 2 else tuple(reversed(ALGORITHMS))
            for path in paths:
                command = [
                    sys.executable,
                    "-m",
                    "benchmarks.run",
                    "--worker",
                    path.stem,
                    "--selections",
                    str(selections),
                ]
                started = perf_counter()
                completed = subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=args.timeout,
                    cwd=ROOT.parent,
                )
                process_s = perf_counter() - started
                record = json.loads(completed.stdout)
                record["run"] = run_number
                record["process_s"] = process_s
                result = output / f"{path.stem}-k{selections}.jsonl"
                with result.open("a") as stream:
                    stream.write(json.dumps(record) + "\n")

    compact_campaign(output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--counts", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument(
        "--output",
        default=str(ROOT / "results" / datetime.now(UTC).strftime("%Y%m%d-%H%M%S")),
    )
    parser.add_argument("--worker", choices=[path.stem for path in ALGORITHMS])
    parser.add_argument("--selections", type=int, default=2)
    args = parser.parse_args()
    if not ALGORITHMS:
        parser.error("No algorithm references found")
    if args.runs < 2 or args.timeout <= 0:
        parser.error("runs must be at least 2 and timeout must be positive")
    if (
        not args.counts
        or min(args.counts) < 1
        or args.counts != sorted(set(args.counts))
    ):
        parser.error("counts must be positive and strictly increasing")
    return args


def main() -> None:
    args = parse_args()
    if args.worker:
        print(json.dumps(worker(args.worker, args.selections)))
    elif args.check:
        validate()
        print("All reference cases passed.")
    else:
        output = benchmark(args)
        print(f"Compact results written to {output}")


if __name__ == "__main__":
    main()
