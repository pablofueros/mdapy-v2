"""Compact verbose experimental campaign results without losing information."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
COMPACT_FILENAMES = frozenset(
    {"campaign.json", "measurements.jsonl", "validations.jsonl"}
)
CAMPAIGN_ARTIFACTS = ("metadata.json", "compilation.json", "summary.json")


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _indices_digest(indices: list[int]) -> str:
    payload = json.dumps(indices, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _raw_result_paths(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.glob("*.jsonl") if path.name not in COMPACT_FILENAMES
    )


def _build_measurements(
    paths: list[Path],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, list[dict[str, Any]]],
]:
    measurements = []
    validations: dict[str, list[int]] = {}
    originals = {}
    for path in paths:
        records = _read_jsonl(path)
        if not records:
            raise ValueError(f"Empty experimental result: {path}")
        originals[path.name] = records

        shared_keys = set(records[0])
        for record in records[1:]:
            shared_keys.intersection_update(record)
        invariant_keys = sorted(
            key
            for key in shared_keys
            if all(record[key] == records[0][key] for record in records[1:])
        )
        context = {key: records[0][key] for key in invariant_keys if key != "indices"}
        indices = records[0].get("indices")
        if not isinstance(indices, list) or not all(
            isinstance(index, int) for index in indices
        ):
            raise ValueError(f"Missing integer indices in {path}")
        if any(record.get("indices") != indices for record in records):
            raise ValueError(f"Selected indices vary between samples in {path}")

        digest = _indices_digest(indices)
        if digest in validations and validations[digest] != indices:
            raise ValueError(f"SHA-256 collision while compacting {path}")
        validations[digest] = indices
        measurements.append(
            {
                "source_file": path.name,
                "context": context,
                "indices_sha256": digest,
                "samples": [
                    {
                        key: value
                        for key, value in record.items()
                        if key not in context and key != "indices"
                    }
                    for record in records
                ],
            }
        )

    validation_rows = [
        {"sha256": digest, "indices": indices}
        for digest, indices in sorted(validations.items())
    ]
    return measurements, validation_rows, originals


def _verify_reconstruction(
    measurements: list[dict[str, Any]],
    validations: list[dict[str, Any]],
    originals: dict[str, list[dict[str, Any]]],
) -> None:
    indices_by_digest = {
        validation["sha256"]: validation["indices"] for validation in validations
    }
    reconstructed = {}
    for measurement in measurements:
        records = []
        for sample in measurement["samples"]:
            record = dict(measurement["context"])
            record.update(sample)
            record["indices"] = indices_by_digest[measurement["indices_sha256"]]
            records.append(record)
        reconstructed[measurement["source_file"]] = records
    if reconstructed != originals:
        raise AssertionError("Compacted results do not reconstruct the source records")


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        "".join(
            json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
            for value in values
        )
    )
    temporary.replace(path)


def compact_campaign(directory: Path) -> dict[str, int]:
    """Replace verbose results in one campaign directory with schema v1."""
    paths = _raw_result_paths(directory)
    if not paths:
        raise ValueError(f"No verbose experimental results found in {directory}")
    existing = [directory / name for name in COMPACT_FILENAMES]
    if any(path.exists() for path in existing):
        raise ValueError(f"Compact results already exist in {directory}")

    measurements, validations, originals = _build_measurements(paths)
    _verify_reconstruction(measurements, validations, originals)
    artifacts = {
        filename: _read_json(directory / filename)
        for filename in CAMPAIGN_ARTIFACTS
        if (directory / filename).exists()
    }
    campaign = {
        "schema_version": SCHEMA_VERSION,
        "artifacts": artifacts,
    }

    _write_json(directory / "campaign.json", campaign)
    _write_jsonl(directory / "measurements.jsonl", measurements)
    _write_jsonl(directory / "validations.jsonl", validations)

    stored_measurements = _read_jsonl(directory / "measurements.jsonl")
    stored_validations = _read_jsonl(directory / "validations.jsonl")
    _verify_reconstruction(stored_measurements, stored_validations, originals)
    if _read_json(directory / "campaign.json") != campaign:
        raise AssertionError("Stored campaign metadata differs from its source")

    for path in paths:
        path.unlink()
    for filename in CAMPAIGN_ARTIFACTS:
        artifact = directory / filename
        if artifact.exists():
            artifact.unlink()
    return {
        "measurements": len(measurements),
        "samples": sum(len(records) for records in originals.values()),
        "validations": len(validations),
    }


def validate_campaign(directory: Path) -> dict[str, int]:
    """Validate hashes and references in one compact campaign directory."""
    campaign = _read_json(directory / "campaign.json")
    if campaign.get("schema_version") != SCHEMA_VERSION or not isinstance(
        campaign.get("artifacts"), dict
    ):
        raise ValueError(f"Invalid campaign document in {directory}")

    measurements = _read_jsonl(directory / "measurements.jsonl")
    validations = _read_jsonl(directory / "validations.jsonl")
    indices_by_digest = {}
    for validation in validations:
        digest = validation.get("sha256")
        indices = validation.get("indices")
        if not isinstance(digest, str) or not isinstance(indices, list):
            raise TypeError(f"Invalid validation record in {directory}")
        if digest != _indices_digest(indices):
            raise ValueError(f"Invalid indices hash in {directory}: {digest}")
        if digest in indices_by_digest:
            raise ValueError(f"Duplicate indices hash in {directory}: {digest}")
        indices_by_digest[digest] = indices

    source_files = set()
    sample_count = 0
    for measurement in measurements:
        source_file = measurement.get("source_file")
        context = measurement.get("context")
        digest = measurement.get("indices_sha256")
        samples = measurement.get("samples")
        if (
            not isinstance(source_file, str)
            or not isinstance(context, dict)
            or not isinstance(digest, str)
            or not isinstance(samples, list)
            or not samples
        ):
            raise ValueError(f"Invalid measurement record in {directory}")
        if source_file in source_files:
            raise ValueError(f"Duplicate source file in {directory}: {source_file}")
        if digest not in indices_by_digest:
            raise ValueError(f"Unknown indices hash in {directory}: {digest}")
        if "indices" in context or any(
            not isinstance(sample, dict) or "indices" in sample for sample in samples
        ):
            raise ValueError(f"Duplicated indices in measurements from {directory}")
        source_files.add(source_file)
        sample_count += len(samples)

    return {
        "measurements": len(measurements),
        "samples": sample_count,
        "validations": len(validations),
    }


def compact_tree(root: Path) -> dict[str, dict[str, int]]:
    """Compact every verbose campaign below an experiment results directory."""
    directories = sorted({path.parent for path in root.rglob("*.jsonl")})
    compacted = {}
    for directory in directories:
        if not _raw_result_paths(directory):
            continue
        compacted[str(directory.relative_to(root))] = compact_campaign(directory)
    if not compacted:
        raise ValueError(f"No verbose campaigns found below {root}")
    return compacted


def validate_tree(root: Path) -> dict[str, dict[str, int]]:
    """Validate every compact campaign below an experiment results directory."""
    directories = sorted(path.parent for path in root.rglob("campaign.json"))
    validated = {
        str(directory.relative_to(root)): validate_campaign(directory)
        for directory in directories
    }
    if not validated:
        raise ValueError(f"No compact campaigns found below {root}")
    return validated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("roots", nargs="+", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for root in args.roots:
        resolved = root.resolve()
        if args.check:
            validated = validate_tree(resolved)
            print(f"{root}: validated {len(validated)} compact campaigns")
        else:
            compacted = compact_tree(resolved)
            print(f"{root}: compacted {len(compacted)} campaigns")


if __name__ == "__main__":
    main()
