from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List

from dataset_prep.discover import discover_database_units

FAILURE_STATUSES = {"syntax_error", "runtime_error", "unsupported", "load_error"}


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    records = list(iter_enriched_records(output_root))
    failure_records = [
        record
        for record in records
        if record.get("oracle_validation_status") in FAILURE_STATUSES
    ]

    report = build_report(
        failure_records,
        output_root=output_root,
        dataset_root=Path(args.dataset_root) if args.dataset_root else None,
        splits=args.splits,
        sample_limit=args.sample_limit,
    )
    write_json(output_root / "failure_analysis.json", report)
    write_markdown(output_root / "failure_analysis.md", report)
    print(f"Wrote {output_root / 'failure_analysis.json'}")
    print(f"Wrote {output_root / 'failure_analysis.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Group dataset_prep translation failures by root-cause signature."
    )
    parser.add_argument("--output-root", default="output/dataset_prep")
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--sample-limit", type=int, default=5)
    return parser.parse_args()


def iter_enriched_records(output_root: Path) -> Iterable[Dict[str, Any]]:
    for path in sorted(output_root.glob("*/*/oracle_sqlpgq_enriched.jsonl")):
        with open(path, encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                meta = record.setdefault("oracle_dataset_meta", {})
                meta.setdefault("output_file", str(path))
                meta.setdefault("output_line", line_number)
                yield record


def build_report(
    records: List[Dict[str, Any]],
    output_root: Path,
    dataset_root: Path | None,
    splits: List[str],
    sample_limit: int,
) -> Dict[str, Any]:
    status_counts = Counter(record.get("oracle_validation_status") for record in records)
    signature_counts: Counter[tuple[str, str]] = Counter()
    by_signature: dict[tuple[str, str], list[Dict[str, Any]]] = defaultdict(list)
    by_database: Counter[str] = Counter()

    for record in records:
        status = record.get("oracle_validation_status", "unknown")
        signature = failure_signature(record)
        signature_counts[(status, signature)] += 1
        by_signature[(status, signature)].append(record)
        meta = record.get("oracle_dataset_meta", {})
        by_database[f"{meta.get('split')}/{meta.get('database')}"] += 1

    missing_outputs = []
    if dataset_root is not None and dataset_root.exists():
        completed = {
            str(path.relative_to(output_root)).split("/summary.json", 1)[0]
            for path in output_root.glob("*/*/summary.json")
        }
        expected = {
            f"{unit.split}/{unit.database}"
            for unit in discover_database_units(dataset_root, splits)
        }
        missing_outputs = sorted(expected - completed)

    top_signatures = []
    for (status, signature), count in signature_counts.most_common():
        top_signatures.append(
            {
                "status": status,
                "signature": signature,
                "count": count,
                "likely_next_action": likely_next_action(status, signature),
                "samples": [
                    sample_record(record)
                    for record in by_signature[(status, signature)][:sample_limit]
                ],
            }
        )

    return {
        "output_root": str(output_root),
        "total_failures": len(records),
        "status_counts": dict(status_counts),
        "top_databases_by_failures": dict(by_database.most_common(30)),
        "databases_without_completed_output": missing_outputs,
        "top_signatures": top_signatures,
    }


def failure_signature(record: Dict[str, Any]) -> str:
    status = record.get("oracle_validation_status", "")
    if status == "unsupported":
        features = record.get("oracle_unsupported_features") or []
        return ",".join(features) or record.get("oracle_translation_category") or "unsupported"

    error = str(record.get("oracle_validation_error") or "").strip()
    if not error:
        return "empty_error"
    ora_match = re.search(r"\bORA-\d{5}\b", error)
    if ora_match:
        first_line = error.splitlines()[0]
        return f"{ora_match.group(0)} {first_line.split(':', 1)[-1].strip()}"
    return error.splitlines()[0][:240]


def likely_next_action(status: str, signature: str) -> str:
    if status == "unsupported":
        return (
            "Decide whether this Cypher feature maps to documented Oracle SQL/PGQ; "
            "otherwise keep classified as unsupported."
        )
    if "ORA-40996" in signature:
        return (
            "Check whether a string literal was emitted as a double-quoted identifier "
            "or a variable was used without a MATCH declaration."
        )
    if "ORA-00936" in signature or "ORA-00904" in signature:
        return (
            "Inspect generated SELECT/ORDER BY aliases; project any outer SQL "
            "references through GRAPH_TABLE COLUMNS."
        )
    if "ORA-42414" in signature:
        return "Fix graph DDL property typing or omit mixed-type properties from colliding labels."
    if "ORA-02291" in signature:
        return (
            "Investigate CSV referential integrity, or add an unenforced import mode "
            "for benchmark data."
        )
    if "ORA-12899" in signature:
        return (
            "Use CLOB or wider text columns for long STRING/TEXT properties during "
            "dataset loading."
        )
    return (
        "Inspect sample source query and generated SQL, then add a focused translator "
        "or loader test."
    )


def sample_record(record: Dict[str, Any]) -> Dict[str, Any]:
    meta = record.get("oracle_dataset_meta", {})
    return {
        "split": meta.get("split"),
        "database": meta.get("database"),
        "record_index": meta.get("record_index"),
        "id": record.get("id"),
        "source_query": record.get("oracle_source_query"),
        "oracle_sqlpgq": record.get("oracle_sqlpgq"),
        "error": record.get("oracle_validation_error"),
        "output_file": meta.get("output_file"),
        "output_line": meta.get("output_line"),
    }


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_markdown(path: Path, report: Dict[str, Any]) -> None:
    lines = [
        "# Dataset Prep Failure Analysis",
        "",
        f"- Output root: `{report['output_root']}`",
        f"- Total query failures: `{report['total_failures']}`",
        f"- Status counts: `{json.dumps(report['status_counts'], sort_keys=True)}`",
        "",
        "## Top Databases",
        "",
    ]
    for database, count in report["top_databases_by_failures"].items():
        lines.append(f"- `{database}`: `{count}`")

    if report["databases_without_completed_output"]:
        lines += ["", "## Databases Without Completed Output", ""]
        for database in report["databases_without_completed_output"]:
            lines.append(f"- `{database}`")

    lines += ["", "## Top Failure Signatures", ""]
    for item in report["top_signatures"]:
        lines.append(
            f"### {item['status']} / {item['signature']} / {item['count']}"
        )
        lines.append("")
        lines.append(item["likely_next_action"])
        lines.append("")
        for sample in item["samples"]:
            lines.append(
                f"- `{sample['split']}/{sample['database']}` "
                f"record `{sample['record_index']}` id `{sample['id']}`"
            )
            lines.append(f"  - Source: `{sample['source_query']}`")
            lines.append(f"  - SQL: `{sample['oracle_sqlpgq']}`")
            if sample.get("error"):
                lines.append(f"  - Error: `{sample['error'].splitlines()[0]}`")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
