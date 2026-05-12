from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List

from app.core.validator.db_client import QueryStatus
from app.impl.oracle_sqlpgq.db_client.oracle_db_client import OracleDBClient
from app.impl.oracle_sqlpgq.utils.sqlpgq import OracleNameSanitizer
from dataset_prep.discover import DatabaseUnit, discover_database_units, source_query
from dataset_prep.oracle_loader import DatasetOracleLoader
from dataset_prep.preflight import run_preflight
from dataset_prep.reporting import merge_global_summaries, summarize_records, write_json
from examples.cypher2oracle_sqlpgq import cypher2oracle_sqlpgq

UNSUPPORTED_PATTERNS = {
    "shortest_path": re.compile(r"\b(ANY|ALL)\s+SHORTEST\b|\bSHORTEST\b", re.IGNORECASE),
    "cost": re.compile(r"\b(TOTAL_COST|COST)\b", re.IGNORECASE),
    "inline_subquery": re.compile(r"\bCALL\s*\{|\bEXISTS\s*\{", re.IGNORECASE),
    "lateral": re.compile(r"\bLATERAL\b", re.IGNORECASE),
    "optional_match": re.compile(r"\bOPTIONAL\s+MATCH\b", re.IGNORECASE),
    "relative_duration": re.compile(r"\bdate\s*\(\s*\)\s*[-+]\s*duration\s*\(", re.IGNORECASE),
    "unwind": re.compile(r"\bUNWIND\b", re.IGNORECASE),
    "case_label_predicate": re.compile(
        r"\bCASE\b(?:(?!\bEND\b).)*\b[A-Za-z_][A-Za-z0-9_]*\s*:",
        re.IGNORECASE | re.DOTALL,
    ),
}


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    preflight = run_preflight(require_oracle_env=not args.skip_live_validation)
    for warning in preflight.warnings:
        print(f"[preflight warning] {warning}")
    if not preflight.ok:
        for error in preflight.errors:
            print(f"[preflight error] {error}")
        raise SystemExit(2)

    units = discover_database_units(dataset_root, args.splits)
    if args.databases:
        requested = {name.lower() for name in args.databases}
        units = [unit for unit in units if unit.database.lower() in requested]
    if args.limit_databases:
        units = units[: args.limit_databases]

    global_summaries: List[Dict[str, Any]] = []
    unsupported_samples: List[Dict[str, Any]] = []
    for unit in units:
        summary_path = output_root / unit.split / unit.database / "summary.json"
        if args.resume and summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("complete"):
                global_summaries.append(summary)
                print(f"[skip] {unit.split}/{unit.database}")
                continue
        try:
            print(f"[start] {unit.split}/{unit.database}", flush=True)
            summary, samples = process_unit(unit, output_root, args)
            global_summaries.append(summary)
            unsupported_samples.extend(samples)
            print(f"[done] {unit.split}/{unit.database}: {summary['validation_statuses']}")
        except Exception as exc:
            print(f"[error] {unit.split}/{unit.database}: {exc}")
            if args.fail_fast:
                raise

    write_json(output_root / "global_summary.json", merge_global_summaries(global_summaries))
    if unsupported_samples:
        write_jsonl(output_root / "unsupported_samples.jsonl", unsupported_samples)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate Text2GQL dataset queries to Oracle SQL/PGQ."
    )
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--output-root", default="output/dataset_prep")
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--databases", nargs="*", default=[])
    parser.add_argument("--graph-prefix", default="T2GQL")
    parser.add_argument("--limit-databases", type=int, default=0)
    parser.add_argument("--limit-queries", type=int, default=0)
    parser.add_argument("--keep-db-on-failure", action="store_true")
    parser.add_argument("--skip-live-validation", action="store_true")
    parser.add_argument(
        "--oracle-validation-timeout-ms",
        type=int,
        default=int(os.environ.get("ORACLE_VALIDATION_TIMEOUT_MS", "60000")),
        help="Oracle timeout for each translated query validation call. Use 0 to disable.",
    )
    parser.add_argument(
        "--oracle-validation-fetch-limit",
        type=int,
        default=int(os.environ.get("ORACLE_VALIDATION_FETCH_LIMIT", "1")),
        help="Rows fetched for each translated query validation call. Use 0 to fetch all.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def process_unit(
    unit: DatabaseUnit,
    output_root: Path,
    args: argparse.Namespace,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    graph_name = graph_name_for(unit, args.graph_prefix)
    out_dir = output_root / unit.split / unit.database
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "oracle_sqlpgq_enriched.jsonl"
    records = load_records(unit.query_path)
    if args.limit_queries:
        records = records[: args.limit_queries]

    client = None
    loader = None
    load_counts: Dict[str, int] = {}
    node_label_map: Dict[str, List[str]] = {}
    edge_label_map: Dict[str, List[str]] = {}
    property_type_map: Dict[str, Dict[str, str]] = {}
    node_primary_key_map: Dict[str, str] = {}
    edge_primary_key_map: Dict[str, str] = {}

    enriched: List[Dict[str, Any]] = []
    unsupported_samples: List[Dict[str, Any]] = []
    try:
        if not args.skip_live_validation:
            client = OracleDBClient(
                {
                    "dsn": os.environ["ORACLE_DSN"],
                    "user": os.environ["ORACLE_USER"],
                    "password": os.environ["ORACLE_PASSWORD"],
                }
            )
            loader = DatasetOracleLoader(client, unit.import_config_path, unit.csv_root, graph_name)
            load_counts = loader.setup()
            node_label_map = loader.node_label_map()
            edge_label_map = loader.edge_label_map()
            property_type_map = loader.property_type_map()
            node_primary_key_map = loader.node_primary_key_map()
            edge_primary_key_map = loader.edge_primary_key_map()

        for index, record in enumerate(records):
            enriched_record = translate_record(
                record,
                graph_name,
                client,
                node_label_map=node_label_map,
                edge_label_map=edge_label_map,
                property_type_map=property_type_map,
                node_primary_key_map=node_primary_key_map,
                edge_primary_key_map=edge_primary_key_map,
                validation_timeout_ms=args.oracle_validation_timeout_ms,
                validation_fetch_limit=args.oracle_validation_fetch_limit,
            )
            enriched_record["oracle_dataset_meta"] = {
                "split": unit.split,
                "database": unit.database,
                "query_file": str(unit.query_path),
                "import_config": str(unit.import_config_path),
                "graph_name": graph_name,
                "record_index": index,
            }
            enriched.append(enriched_record)
            if enriched_record["oracle_validation_status"] == "unsupported":
                unsupported_samples.append(
                    {
                        "split": unit.split,
                        "database": unit.database,
                        "id": record.get("id"),
                        "unsupported_features": enriched_record["oracle_unsupported_features"],
                        "query": enriched_record.get("oracle_source_query"),
                    }
                )
        write_jsonl(output_path, enriched)
        summary = summarize_records(enriched)
        summary.update(
            {
                "split": unit.split,
                "database": unit.database,
                "graph_name": graph_name,
                "query_file": str(unit.query_path),
                "import_config": str(unit.import_config_path),
                "load_counts": load_counts,
            }
        )
        write_json(out_dir / "summary.json", summary)
        return summary, unsupported_samples
    except Exception:
        if not args.keep_db_on_failure and loader is not None:
            loader.cleanup(ignore_errors=True)
        raise
    finally:
        if loader is not None and not args.keep_db_on_failure:
            loader.cleanup(ignore_errors=True)
        if client is not None:
            client.close()


def load_records(query_path: Path) -> List[Dict[str, Any]]:
    data = json.loads(query_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return [dict(value, id=key) if isinstance(value, dict) else {"id": key, "value": value}
                for key, value in data.items()]
    return list(data)


def translate_record(
    record: Dict[str, Any],
    graph_name: str,
    client: OracleDBClient | None,
    node_label_map: Dict[str, List[str]] | None = None,
    edge_label_map: Dict[str, List[str]] | None = None,
    property_type_map: Dict[str, Dict[str, str]] | None = None,
    node_primary_key_map: Dict[str, str] | None = None,
    edge_primary_key_map: Dict[str, str] | None = None,
    validation_timeout_ms: int = 0,
    validation_fetch_limit: int = 0,
) -> Dict[str, Any]:
    output = dict(record)
    query_field, query = source_query(record)
    output["oracle_source_query_field"] = query_field
    output["oracle_source_query"] = query
    output["oracle_unsupported_features"] = detect_unsupported_features(query)
    if not query:
        output.update(
            _status(None, "missing_source_query", "unsupported", "No source query found.")
        )
        return output
    if output["oracle_unsupported_features"]:
        output.update(
            _status(
                None,
                "Graph-IL Not Support",
                "unsupported",
                "Query uses constructs intentionally not emitted for Oracle SQL/PGQ.",
            )
        )
        return output

    translated, category = cypher2oracle_sqlpgq(
        query,
        graph_name=graph_name,
        node_label_map=node_label_map,
        edge_label_map=edge_label_map,
        property_type_map=property_type_map,
        node_primary_key_map=node_primary_key_map,
        edge_primary_key_map=edge_primary_key_map,
        strict_property_validation=bool(property_type_map),
    )
    if category != "Graph-IL Translatable":
        output.update(_status(None, category, "unsupported", translated))
        return output

    validation_status = "syntax_ok"
    error = ""
    if client is not None:
        result = client.execute_query(
            translated,
            fetch_limit=validation_fetch_limit,
            call_timeout_ms=validation_timeout_ms,
        )
        if result.status_code == QueryStatus.SUCCESS:
            validation_status = "success"
        elif result.status_code == QueryStatus.NO_RECORD:
            validation_status = "no_record"
        elif result.status_code == QueryStatus.CLIENT_ERROR:
            validation_status = "syntax_error"
            error = result.error or ""
        else:
            validation_status = "runtime_error"
            error = result.error or ""

    output.update(_status(translated, category, validation_status, error))
    return output


def detect_unsupported_features(query: str) -> List[str]:
    return [
        name
        for name, pattern in UNSUPPORTED_PATTERNS.items()
        if query and pattern.search(query)
    ]


def _status(
    sqlpgq: str | None,
    category: str,
    validation_status: str,
    error: str,
) -> Dict[str, Any]:
    return {
        "oracle_sqlpgq": sqlpgq,
        "oracle_translation_category": category,
        "oracle_validation_status": validation_status,
        "oracle_validation_error": error,
    }


def graph_name_for(unit: DatabaseUnit, prefix: str) -> str:
    return OracleNameSanitizer.clean(f"{prefix}_{unit.split}_{unit.database}", fallback="GRAPH")


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
