from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import date, datetime, timedelta
from decimal import Decimal
import json
import logging
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, Iterable, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from neo4j import GraphDatabase, Query
except ImportError:  # pragma: no cover - depends on local environment.
    GraphDatabase = None
    Query = None

from app.core.validator.db_client import QueryStatus
from app.impl.oracle_sqlpgq.db_client.oracle_db_client import OracleDBClient
from app.impl.oracle_sqlpgq.utils.sqlpgq import OracleNameSanitizer
from dataset_prep.discover import DatabaseUnit, discover_database_units
from dataset_prep.oracle_loader import DatasetOracleLoader
from dataset_prep.reporting import append_jsonl, write_json


DEFAULT_VALID_ORACLE_STATUSES = {"success", "no_record"}
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)


class DatasetNeo4jLoader:
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str,
        import_config_path: Path,
        csv_root: Path,
        batch_size: int = 1000,
    ):
        if GraphDatabase is None:
            raise RuntimeError("Install the 'neo4j' package before running this script.")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.import_config_path = import_config_path
        self.csv_root = csv_root
        self.batch_size = batch_size
        self.clear_batch_size = max(batch_size, 1000)
        self.config = json.loads(import_config_path.read_text(encoding="utf-8"))
        self.schema = list(self.config.get("schema", []))
        self.files = list(self.config.get("files", []))
        self.vertices = [item for item in self.schema if item.get("type") == "VERTEX"]
        self.edges = [item for item in self.schema if item.get("type") == "EDGE"]
        self.vertex_by_label = {item["label"]: item for item in self.vertices}
        self.primary_by_label = {
            item["label"]: item.get("primary", "_id") for item in self.vertices
        }
        self.property_types_by_label = {
            item["label"]: {
                prop["name"]: prop.get("type", "STRING")
                for prop in item.get("properties", [])
                if prop.get("name")
            }
            for item in self.schema
        }

    def close(self) -> None:
        self.driver.close()

    def setup(self, clear: bool = True) -> Dict[str, int]:
        with self.driver.session(database=self.database) as session:
            if clear:
                self.clear(session)
            self._create_constraints(session)
        counts = self._load_vertices()
        counts.update(self._load_edges())
        return counts

    def clear(self, session: Any | None = None) -> None:
        if session is not None:
            self._clear_with_session(session)
            return
        with self.driver.session(database=self.database) as owned_session:
            self._clear_with_session(owned_session)

    def _clear_with_session(self, session: Any) -> None:
        rel_delete = (
            "MATCH ()-[r]-() "
            "WITH r LIMIT $limit "
            "DELETE r "
            "RETURN count(r) AS deleted"
        )
        node_delete = (
            "MATCH (n) "
            "WITH n LIMIT $limit "
            "DELETE n "
            "RETURN count(n) AS deleted"
        )
        self._delete_until_empty(session, rel_delete)
        self._delete_until_empty(session, node_delete)

    def _delete_until_empty(self, session: Any, query: str) -> None:
        while True:
            record = session.run(query, limit=self.clear_batch_size).single()
            deleted = record["deleted"] if record else 0
            if deleted == 0:
                return

    def execute(self, query: str, timeout_s: float | None = None) -> tuple[str, list[dict], str]:
        query = self.prepare_query(query)
        try:
            with self.driver.session(database=self.database) as session:
                executable = (
                    Query(query, timeout=timeout_s)
                    if Query is not None and timeout_s
                    else query
                )
                result = session.run(executable)
                return "success", [dict(record) for record in result], ""
        except Exception as exc:
            error = str(exc)
            status = "client_error" if "syntax" in error.lower() else "server_error"
            return status, [], error

    def prepare_query(self, query: str) -> str:
        query = self._coerce_string_backed_boolean_literals(query)
        query = self._coerce_string_backed_numeric_comparisons(query)
        query = self._coerce_string_backed_date_comparisons(query)
        return query

    def _coerce_string_backed_boolean_literals(self, query: str) -> str:
        variables = self._query_variable_labels(query)

        def replace_not_property(match: re.Match) -> str:
            variable = match.group("variable")
            property_name = match.group("property")
            if not self._is_string_property(variables.get(variable, ""), property_name):
                return match.group(0)
            return f"{variable}.{property_name} = 'false'"

        query = re.sub(
            r"\bNOT\s+(?P<variable>[A-Za-z_][A-Za-z0-9_]*)\."
            r"(?P<property>[A-Za-z_][A-Za-z0-9_$#]*)\b",
            replace_not_property,
            query,
            flags=re.IGNORECASE,
        )

        def replace_comparison(match: re.Match) -> str:
            variable = match.group("variable")
            property_name = match.group("property")
            value = match.group("value").lower()
            if not self._is_string_property(variables.get(variable, ""), property_name):
                return match.group(0)
            return f"{variable}.{property_name} {match.group('operator')} '{value}'"

        query = re.sub(
            r"\b(?P<variable>[A-Za-z_][A-Za-z0-9_]*)\."
            r"(?P<property>[A-Za-z_][A-Za-z0-9_$#]*)\s*"
            r"(?P<operator>=|<>)\s*(?P<value>true|false)\b",
            replace_comparison,
            query,
            flags=re.IGNORECASE,
        )

        def replace_map_literal(match: re.Match) -> str:
            property_name = match.group("property")
            value = match.group("value").lower()
            if not self._has_string_property(property_name):
                return match.group(0)
            return f"{property_name}: '{value}'"

        return re.sub(
            r"\b(?P<property>[A-Za-z_][A-Za-z0-9_$#]*)\s*:\s*"
            r"(?P<value>true|false)\b",
            replace_map_literal,
            query,
            flags=re.IGNORECASE,
        )

    def _coerce_string_backed_numeric_comparisons(self, query: str) -> str:
        variables = self._query_variable_labels(query)

        def replace_left(match: re.Match) -> str:
            variable = match.group("variable")
            property_name = match.group("property")
            if not self._is_string_property(variables.get(variable, ""), property_name):
                return match.group(0)
            return (
                f"{variable}.{property_name} {match.group('operator')} "
                f"'{match.group('value')}'"
            )

        query = re.sub(
            r"\b(?P<variable>[A-Za-z_][A-Za-z0-9_]*)\."
            r"(?P<property>[A-Za-z_][A-Za-z0-9_$#]*)\s*"
            r"(?P<operator><=|>=|<>|=|<|>)\s*"
            r"(?P<value>-?\d+(?:\.\d+)?)\b",
            replace_left,
            query,
        )

        def replace_right(match: re.Match) -> str:
            variable = match.group("variable")
            property_name = match.group("property")
            if not self._is_string_property(variables.get(variable, ""), property_name):
                return match.group(0)
            return (
                f"'{match.group('value')}' {match.group('operator')} "
                f"{variable}.{property_name}"
            )

        query = re.sub(
            r"\b(?P<value>-?\d+(?:\.\d+)?)\s*"
            r"(?P<operator><=|>=|<>|=|<|>)\s*"
            r"(?P<variable>[A-Za-z_][A-Za-z0-9_]*)\."
            r"(?P<property>[A-Za-z_][A-Za-z0-9_$#]*)\b",
            replace_right,
            query,
        )

        def replace_map_literal(match: re.Match) -> str:
            property_name = match.group("property")
            value = match.group("value")
            if not self._has_string_property(property_name):
                return match.group(0)
            return f"{property_name}: '{value}'"

        return re.sub(
            r"\b(?P<property>[A-Za-z_][A-Za-z0-9_$#]*)\s*:\s*"
            r"(?P<value>-?\d+(?:\.\d+)?)\b",
            replace_map_literal,
            query,
        )

    def _coerce_string_backed_date_comparisons(self, query: str) -> str:
        variables = self._query_variable_labels(query)

        query = self._coerce_string_backed_date_accessors(query, variables)

        def replace_left(match: re.Match) -> str:
            variable = match.group("variable")
            property_name = match.group("property")
            left = match.group("left")
            if not self._is_string_property(variables.get(variable, ""), property_name):
                return match.group(0)
            if re.fullmatch(r"\s*date\s*\(", left, flags=re.IGNORECASE):
                return match.group(0)
            return (
                f"date({variable}.{property_name}) "
                f"{match.group('operator')} {match.group('right')}"
            )

        query = re.sub(
            r"(?P<left>\b(?P<variable>[A-Za-z_][A-Za-z0-9_]*)\."
            r"(?P<property>[A-Za-z_][A-Za-z0-9_$#]*))\s*"
            r"(?P<operator><=|>=|<>|=|<|>)\s*"
            r"(?P<right>date\s*\([^)]+\))",
            replace_left,
            query,
            flags=re.IGNORECASE,
        )

        def replace_right(match: re.Match) -> str:
            variable = match.group("variable")
            property_name = match.group("property")
            if not self._is_string_property(variables.get(variable, ""), property_name):
                return match.group(0)
            return (
                f"{match.group('left')} {match.group('operator')} "
                f"date({variable}.{property_name})"
            )

        return re.sub(
            r"(?P<left>date\s*\([^)]+\))\s*"
            r"(?P<operator><=|>=|<>|=|<|>)\s*"
            r"(?P<variable>[A-Za-z_][A-Za-z0-9_]*)\."
            r"(?P<property>[A-Za-z_][A-Za-z0-9_$#]*)\b",
            replace_right,
            query,
            flags=re.IGNORECASE,
        )

    def _coerce_string_backed_date_accessors(
        self,
        query: str,
        variables: Dict[str, str],
    ) -> str:
        def accessor_expression(variable: str, property_name: str, accessor: str) -> str:
            base = (
                f"datetime({variable}.{property_name})"
                if self._looks_like_datetime_string_property(variable, property_name)
                else f"date({variable}.{property_name})"
            )
            return f"{base}.{accessor}"

        def replace_date_call(match: re.Match) -> str:
            variable = match.group("variable")
            property_name = match.group("property")
            if not self._is_string_property(variables.get(variable, ""), property_name):
                return match.group(0)
            return accessor_expression(variable, property_name, match.group("accessor"))

        query = re.sub(
            r"\bdate\s*\(\s*(?P<variable>[A-Za-z_][A-Za-z0-9_]*)\."
            r"(?P<property>[A-Za-z_][A-Za-z0-9_$#]*)\s*\)\."
            r"(?P<accessor>year|month|day|weekday|dayOfWeek)\b",
            replace_date_call,
            query,
            flags=re.IGNORECASE,
        )

        def replace_direct_accessor(match: re.Match) -> str:
            variable = match.group("variable")
            property_name = match.group("property")
            if not self._is_string_property(variables.get(variable, ""), property_name):
                return match.group(0)
            return accessor_expression(variable, property_name, match.group("accessor"))

        return re.sub(
            r"\b(?P<variable>[A-Za-z_][A-Za-z0-9_]*)\."
            r"(?P<property>[A-Za-z_][A-Za-z0-9_$#]*)\."
            r"(?P<accessor>year|month|day|weekday|dayOfWeek)\b",
            replace_direct_accessor,
            query,
            flags=re.IGNORECASE,
        )

    def _looks_like_datetime_string_property(self, variable: str, property_name: str) -> bool:
        return property_name.lower().endswith(("at", "time", "timestamp", "datetime"))

    def _query_variable_labels(self, query: str) -> Dict[str, str]:
        labels: Dict[str, str] = {}
        for match in re.finditer(
            r"\(\s*(?P<variable>[A-Za-z_][A-Za-z0-9_]*)\s*:\s*"
            r"(?:`(?P<quoted_label>[^`]+)`|(?P<label>[A-Za-z_][A-Za-z0-9_$#-]*))",
            query,
        ):
            labels[match.group("variable")] = match.group("quoted_label") or match.group("label")
        return labels

    def _is_string_property(self, label: str, property_name: str) -> bool:
        if not label:
            return False
        label_types = self.property_types_by_label.get(label, {})
        return label_types.get(property_name, "").upper() == "STRING"

    def _has_string_property(self, property_name: str) -> bool:
        return any(
            properties.get(property_name, "").upper() == "STRING"
            for properties in self.property_types_by_label.values()
        )

    def _create_constraints(self, session: Any) -> None:
        for vertex in self.vertices:
            label = vertex["label"]
            primary = vertex.get("primary")
            if not primary:
                continue
            name = _safe_identifier(f"constraint_{label}_{primary}")
            session.run(
                f"CREATE CONSTRAINT `{name}` IF NOT EXISTS "
                f"FOR (n:`{_escape_backticks(label)}`) "
                f"REQUIRE n.`{_escape_backticks(primary)}` IS UNIQUE"
            ).consume()

    def _load_vertices(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        files_by_label = {item["label"]: item for item in self.files if "SRC_ID" not in item}
        for vertex in self.vertices:
            file_item = files_by_label.get(vertex["label"])
            if not file_item:
                continue
            rows = self._read_file(vertex, file_item, is_edge=False)
            self._write_vertex_batches(vertex, rows)
            counts[f"vertex:{vertex['label']}"] = len(rows)
        return counts

    def _load_edges(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for file_item in [item for item in self.files if "SRC_ID" in item and "DST_ID" in item]:
            edge = self._find_edge_schema(file_item)
            if not edge:
                continue
            rows = self._read_file(edge, file_item, is_edge=True)
            self._write_edge_batches(edge, file_item, rows)
            key = f"edge:{file_item['SRC_ID']}-[{edge['label']}]->{file_item['DST_ID']}"
            counts[key] = len(rows)
        return counts

    def _find_edge_schema(self, file_item: Dict[str, Any]) -> Dict[str, Any] | None:
        label = file_item.get("label")
        src = file_item.get("SRC_ID")
        dst = file_item.get("DST_ID")
        for edge in self.edges:
            if edge.get("label") != label:
                continue
            if [src, dst] in edge.get("constraints", []):
                return edge
        for edge in self.edges:
            if edge.get("label") == label:
                return edge
        return None

    def _read_file(
        self,
        schema_item: Dict[str, Any],
        file_item: Dict[str, Any],
        is_edge: bool,
    ) -> List[Dict[str, Any]]:
        path = self.csv_root / file_item["path"]
        source_columns = list(file_item.get("columns", []))
        schema_types = {
            prop["name"]: prop.get("type", "STRING")
            for prop in schema_item.get("properties", [])
        }
        header_rows = int(file_item.get("header", 0))
        rows: List[Dict[str, Any]] = []
        with open(path, newline="", encoding="utf-8-sig") as file:
            reader = csv.reader(file)
            for index, raw in enumerate(reader):
                if index < header_rows:
                    continue
                row = {
                    column: raw[position] if position < len(raw) else ""
                    for position, column in enumerate(source_columns)
                }
                converted = {}
                for column, value in row.items():
                    if is_edge and column == "SRC_ID":
                        converted[column] = _convert_value(
                            value,
                            self._vertex_primary_type(file_item["SRC_ID"]),
                        )
                    elif is_edge and column == "DST_ID":
                        converted[column] = _convert_value(
                            value,
                            self._vertex_primary_type(file_item["DST_ID"]),
                        )
                    else:
                        converted[column] = _convert_value(
                            value,
                            schema_types.get(column, "STRING"),
                        )
                rows.append(converted)
        return rows

    def _vertex_primary_type(self, label: str) -> str:
        vertex = self.vertex_by_label.get(label, {})
        primary = vertex.get("primary", "_id")
        for prop in vertex.get("properties", []):
            if prop.get("name") == primary:
                return prop.get("type", "STRING")
        return "STRING"

    def _write_vertex_batches(self, vertex: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        label = _escape_backticks(vertex["label"])
        primary = _escape_backticks(vertex.get("primary", "_id"))
        query = (
            f"UNWIND $batch AS row "
            f"MERGE (n:`{label}` {{`{primary}`: row.`{primary}`}}) "
            f"SET n += row"
        )
        self._run_batches(query, rows)

    def _write_edge_batches(
        self,
        edge: Dict[str, Any],
        file_item: Dict[str, Any],
        rows: List[Dict[str, Any]],
    ) -> None:
        if not rows:
            return
        src_label = file_item["SRC_ID"]
        dst_label = file_item["DST_ID"]
        src_pk = self.primary_by_label.get(src_label, "_id")
        dst_pk = self.primary_by_label.get(dst_label, "_id")
        rel_type = _escape_backticks(edge["label"])
        query = (
            f"UNWIND $batch AS row "
            f"MATCH (src:`{_escape_backticks(src_label)}` "
            f"{{`{_escape_backticks(src_pk)}`: row.SRC_ID}}) "
            f"MATCH (dst:`{_escape_backticks(dst_label)}` "
            f"{{`{_escape_backticks(dst_pk)}`: row.DST_ID}}) "
            f"CREATE (src)-[r:`{rel_type}`]->(dst) "
            f"SET r += row.props"
        )
        batch_rows = [
            {
                "SRC_ID": row["SRC_ID"],
                "DST_ID": row["DST_ID"],
                "props": {
                    key: value
                    for key, value in row.items()
                    if key not in ("SRC_ID", "DST_ID")
                },
            }
            for row in rows
        ]
        self._run_batches(query, batch_rows)

    def _run_batches(self, query: str, rows: List[Dict[str, Any]]) -> None:
        with self.driver.session(database=self.database) as session:
            for start in range(0, len(rows), self.batch_size):
                session.run(query, batch=rows[start : start + self.batch_size]).consume()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Oracle SQL/PGQ query results with Neo4j Cypher results."
    )
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--dataset-output-root", default="output/dataset_prep")
    parser.add_argument("--output-root", default="output/oracle_neo4j_compare")
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--databases", nargs="*", default=[])
    parser.add_argument("--limit-databases", type=int, default=0)
    parser.add_argument("--limit-queries", type=int, default=0)
    parser.add_argument(
        "--query-offset",
        type=int,
        default=0,
        help="Skip this many enriched query records before applying --limit-queries.",
    )
    parser.add_argument("--graph-prefix", default="T2GQL")
    parser.add_argument(
        "--oracle-statuses",
        nargs="+",
        default=sorted(DEFAULT_VALID_ORACLE_STATUSES),
    )
    parser.add_argument("--include-all-translatable", action="store_true")
    parser.add_argument("--oracle-timeout-ms", type=int, default=60000)
    parser.add_argument("--neo4j-timeout-s", type=float, default=60.0)
    parser.add_argument("--neo4j-uri", default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--neo4j-user", default=os.environ.get("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.environ.get("NEO4J_PASSWORD", "password"))
    parser.add_argument("--neo4j-database", default=os.environ.get("NEO4J_DATABASE", "neo4j"))
    parser.add_argument("--neo4j-batch-size", type=int, default=1000)
    parser.add_argument("--keep-loaded", action="store_true")
    parser.add_argument(
        "--reuse-loaded",
        action="store_true",
        help="Skip Oracle/Neo4j load setup and validate against already-loaded graphs.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="Print query progress every N selected records.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    failures_path = output_root / "mismatched_or_failed_queries.jsonl"
    failures_path.write_text("", encoding="utf-8")
    units = discover_database_units(Path(args.dataset_root), args.splits)
    if args.databases:
        requested = {name.lower() for name in args.databases}
        units = [unit for unit in units if unit.database.lower() in requested]
    if args.limit_databases:
        units = units[: args.limit_databases]

    oracle_client = OracleDBClient(
        {
            "dsn": os.environ["ORACLE_DSN"],
            "user": os.environ["ORACLE_USER"],
            "password": os.environ["ORACLE_PASSWORD"],
        }
    )
    all_summaries: List[Dict[str, Any]] = []
    try:
        for unit in units:
            print(f"[start] {unit.split}/{unit.database}", flush=True)
            summary = compare_unit(unit, oracle_client, args, failures_path)
            all_summaries.append(summary)
            matched = summary["matched"]
            failed = summary["failed"]
            skipped = summary["skipped"]
            print(
                f"[done] {unit.split}/{unit.database}: "
                f"matched={matched} failed={failed} skipped={skipped}",
                flush=True,
            )
    finally:
        oracle_client.close()

    write_json(output_root / "summary.json", merge_compare_summaries(all_summaries))


def compare_unit(
    unit: DatabaseUnit,
    oracle_client: OracleDBClient,
    args: argparse.Namespace,
    failures_path: Path,
) -> Dict[str, Any]:
    graph_name = graph_name_for(unit, args.graph_prefix)
    oracle_loader = DatasetOracleLoader(
        oracle_client,
        unit.import_config_path,
        unit.csv_root,
        graph_name,
    )
    neo4j_loader = DatasetNeo4jLoader(
        args.neo4j_uri,
        args.neo4j_user,
        args.neo4j_password,
        args.neo4j_database,
        unit.import_config_path,
        unit.csv_root,
        args.neo4j_batch_size,
    )
    summary: Dict[str, Any] = {
        "split": unit.split,
        "database": unit.database,
        "query_file": str(unit.query_path),
        "import_config": str(unit.import_config_path),
        "graph_name": graph_name,
        "loaded": {},
        "considered": 0,
        "matched": 0,
        "failed": 0,
        "skipped": 0,
        "skip_reasons": {},
        "failure_reasons": {},
    }
    failures: List[Dict[str, Any]] = []
    try:
        if args.reuse_loaded:
            print(f"[load] {unit.split}/{unit.database}: reusing loaded graphs", flush=True)
            summary["loaded"] = {"reused": True}
        else:
            print(f"[load] {unit.split}/{unit.database}: oracle", flush=True)
            oracle_counts = oracle_loader.setup()
            print(f"[load] {unit.split}/{unit.database}: neo4j", flush=True)
            neo4j_counts = neo4j_loader.setup(clear=True)
            summary["loaded"] = {"oracle": oracle_counts, "neo4j": neo4j_counts}
            print(f"[load] {unit.split}/{unit.database}: done", flush=True)
        all_records = load_enriched_records(unit, Path(args.dataset_output_root))
        records = select_records_for_range(all_records, args.query_offset, args.limit_queries)
        summary["total_records"] = len(all_records)
        summary["query_offset"] = args.query_offset
        summary["selected_records"] = len(records)
        if args.limit_queries:
            summary["limit_queries"] = args.limit_queries
        valid_statuses = set(args.oracle_statuses)
        for selected_index, record in enumerate(records, start=1):
            if args.progress_every and selected_index % args.progress_every == 0:
                print(
                    f"[progress] {unit.split}/{unit.database}: "
                    f"{selected_index}/{len(records)} selected records",
                    flush=True,
                )
            skip_reason = skip_reason_for_record(
                record,
                valid_statuses=valid_statuses,
                include_all_translatable=args.include_all_translatable,
            )
            if skip_reason:
                summary["skipped"] += 1
                increment(summary["skip_reasons"], skip_reason)
                continue
            cypher = (
                record.get("oracle_source_query")
                or record.get("initial_cypher")
                or record.get("cypher")
                or ""
            )
            if is_nondeterministic_limit_without_order(cypher):
                summary["skipped"] += 1
                increment(summary["skip_reasons"], "nondeterministic_limit_without_order")
                continue
            summary["considered"] += 1
            comparison = compare_record(record, oracle_client, neo4j_loader, args)
            if comparison["matched"]:
                summary["matched"] += 1
                continue
            if comparison["reason"] == "suspected_order_by_limit_tie":
                summary["skipped"] += 1
                increment(summary["skip_reasons"], comparison["reason"])
                continue
            summary["failed"] += 1
            increment(summary["failure_reasons"], comparison["reason"])
            failures.append(
                {
                    "split": unit.split,
                    "database": unit.database,
                    "record_id": record.get("id"),
                    "record_index": record.get("oracle_dataset_meta", {}).get("record_index"),
                    "reason": comparison["reason"],
                    "cypher": comparison["cypher"],
                    "oracle_sqlpgq": comparison["oracle_sqlpgq"],
                    "oracle_status": comparison["oracle_status"],
                    "neo4j_status": comparison["neo4j_status"],
                    "oracle_error": comparison["oracle_error"],
                    "neo4j_error": comparison["neo4j_error"],
                    "oracle_rows_sample": comparison["oracle_rows_sample"],
                    "neo4j_rows_sample": comparison["neo4j_rows_sample"],
                }
            )
            if len(failures) >= 100:
                append_jsonl(failures_path, failures)
                failures = []
        if failures:
            append_jsonl(failures_path, failures)
        write_json(
            Path(args.output_root) / unit.split / unit.database / "summary.json",
            summary,
        )
        return summary
    finally:
        if not args.keep_loaded:
            if not args.reuse_loaded:
                oracle_loader.cleanup(ignore_errors=True)
                try:
                    neo4j_loader.clear()
                except Exception:
                    pass
        neo4j_loader.close()


def compare_record(
    record: Dict[str, Any],
    oracle_client: OracleDBClient,
    neo4j_loader: DatasetNeo4jLoader,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    oracle_sqlpgq = record.get("oracle_sqlpgq") or ""
    cypher = (
        record.get("oracle_source_query")
        or record.get("initial_cypher")
        or record.get("cypher")
        or ""
    )
    oracle_result = oracle_client.execute_query(
        oracle_sqlpgq,
        call_timeout_ms=args.oracle_timeout_ms,
    )
    neo4j_status, neo4j_rows, neo4j_error = neo4j_loader.execute(cypher, args.neo4j_timeout_s)
    oracle_status = query_status_name(oracle_result.status_code)
    oracle_rows = oracle_result.data if isinstance(oracle_result.data, list) else []
    if oracle_status not in ("success", "no_record") or neo4j_status != "success":
        return comparison_result(
            False,
            "execution_error",
            cypher,
            oracle_sqlpgq,
            oracle_status,
            neo4j_status,
            oracle_result.error or "",
            neo4j_error,
            oracle_rows,
            neo4j_rows,
            neo4j_loader.primary_by_label,
        )
    oracle_counter = normalized_counter(oracle_rows, neo4j_loader.primary_by_label)
    neo4j_counter = normalized_counter(neo4j_rows, neo4j_loader.primary_by_label)
    matched = oracle_counter == neo4j_counter
    reason = "result_mismatch" if not matched else ""
    if not matched and is_order_by_limit_query(cypher) and oracle_rows and neo4j_rows:
        reason = "suspected_order_by_limit_tie"
    return comparison_result(
        matched,
        reason,
        cypher,
        oracle_sqlpgq,
        oracle_status,
        neo4j_status,
        "",
        "",
        oracle_rows,
        neo4j_rows,
        neo4j_loader.primary_by_label,
    )


def comparison_result(
    matched: bool,
    reason: str,
    cypher: str,
    oracle_sqlpgq: str,
    oracle_status: str,
    neo4j_status: str,
    oracle_error: str,
    neo4j_error: str,
    oracle_rows: Sequence[Dict[str, Any]],
    neo4j_rows: Sequence[Dict[str, Any]],
    primary_by_label: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    return {
        "matched": matched,
        "reason": reason,
        "cypher": cypher,
        "oracle_sqlpgq": oracle_sqlpgq,
        "oracle_status": oracle_status,
        "neo4j_status": neo4j_status,
        "oracle_error": oracle_error,
        "neo4j_error": neo4j_error,
        "oracle_rows_sample": normalize_rows(oracle_rows[:5], primary_by_label),
        "neo4j_rows_sample": normalize_rows(neo4j_rows[:5], primary_by_label),
    }


def load_enriched_records(unit: DatabaseUnit, output_root: Path) -> List[Dict[str, Any]]:
    path = output_root / unit.split / unit.database / "oracle_sqlpgq_enriched.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing enriched Oracle SQL/PGQ records for {unit.split}/{unit.database}: {path}"
        )
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def select_records_for_range(
    records: Sequence[Dict[str, Any]],
    query_offset: int = 0,
    limit_queries: int = 0,
) -> List[Dict[str, Any]]:
    start = max(query_offset, 0)
    if limit_queries > 0:
        return list(records[start : start + limit_queries])
    return list(records[start:])


def skip_reason_for_record(
    record: Dict[str, Any],
    valid_statuses: set[str],
    include_all_translatable: bool,
) -> str:
    if record.get("oracle_translation_category") != "Graph-IL Translatable":
        return "not_translatable"
    if not record.get("oracle_sqlpgq"):
        return "missing_oracle_sqlpgq"
    if not (
        record.get("oracle_source_query")
        or record.get("initial_cypher")
        or record.get("cypher")
    ):
        return "missing_cypher"
    if include_all_translatable:
        return ""
    status = record.get("oracle_validation_status")
    if status not in valid_statuses:
        return f"oracle_status:{status}"
    return ""


def is_nondeterministic_limit_without_order(query: str) -> bool:
    normalized = _strip_string_literals(query)
    return bool(
        re.search(r"\bLIMIT\b", normalized, flags=re.IGNORECASE)
        and not re.search(r"\bORDER\s+BY\b", normalized, flags=re.IGNORECASE)
    )


def is_order_by_limit_query(query: str) -> bool:
    normalized = _strip_string_literals(query)
    return bool(
        re.search(r"\bORDER\s+BY\b", normalized, flags=re.IGNORECASE)
        and re.search(r"\bLIMIT\b", normalized, flags=re.IGNORECASE)
    )


def _strip_string_literals(query: str) -> str:
    return re.sub(r"'(?:''|\\'|[^'])*'|\"(?:\\\"|[^\"])*\"", "''", query or "")


def normalized_counter(
    rows: Sequence[Dict[str, Any]],
    primary_by_label: Dict[str, str] | None = None,
) -> Counter[str]:
    return Counter(
        json.dumps(row, sort_keys=True, ensure_ascii=False)
        for row in normalize_rows(rows, primary_by_label)
    )


def normalize_rows(
    rows: Sequence[Dict[str, Any]],
    primary_by_label: Dict[str, str] | None = None,
) -> List[Any]:
    return [normalize_row(row, primary_by_label) for row in rows]


def normalize_row(
    row: Dict[str, Any],
    primary_by_label: Dict[str, str] | None = None,
) -> Any:
    # Compare return values rather than aliases: Oracle aliases often differ from Cypher aliases.
    values = list(row.values())
    if len(values) == 1 and _looks_like_path(values[0]):
        return _normalize_path(values[0], primary_by_label)
    return [_normalize_value(value, primary_by_label) for value in values]


def _normalize_value(value: Any, primary_by_label: Dict[str, str] | None = None) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else round(float(value), 6)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        rounded = round(value, 6)
        return int(rounded) if rounded.is_integer() else rounded
    if isinstance(value, timedelta):
        return value.total_seconds()
    if isinstance(value, datetime):
        if value.hour == value.minute == value.second == value.microsecond == 0:
            return value.date().isoformat()
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        return _normalize_temporal_string(value)
    if _looks_like_path(value):
        return _normalize_path(value, primary_by_label)
    if isinstance(value, (list, tuple)):
        return [_normalize_value(item, primary_by_label) for item in value]
    if isinstance(value, dict):
        oracle_identity = _normalize_oracle_graph_identity(value, primary_by_label)
        if oracle_identity is not None:
            return oracle_identity
        return {
            str(key): _normalize_value(item, primary_by_label)
            for key, item in sorted(value.items())
        }
    if hasattr(value, "items") and hasattr(value, "labels"):
        return _normalize_neo4j_node(value, primary_by_label)
    if hasattr(value, "items") and hasattr(value, "type"):
        return _normalize_neo4j_relationship(value, primary_by_label)
    if hasattr(value, "iso_format"):
        return value.iso_format()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _normalize_temporal_string(value: str) -> str:
    date_midnight = re.fullmatch(r"(\d{4}-\d{2}-\d{2})T00:00:00(?:\.0{6,9})?(?:Z)?", value)
    if date_midnight:
        return date_midnight.group(1)
    match = re.fullmatch(
        r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.0{6,9})?(?:Z)?",
        value,
    )
    if match:
        return match.group(1)
    return value


def _normalize_oracle_graph_identity(
    value: Dict[str, Any],
    primary_by_label: Dict[str, str] | None = None,
) -> Dict[str, Any] | None:
    if "ELEM_TABLE" not in value or "KEY_VALUE" not in value:
        return None
    label = str(value["ELEM_TABLE"])
    normalized = {
        "element": OracleNameSanitizer.clean(label, fallback=label),
    }
    key = _normalize_value(value["KEY_VALUE"], primary_by_label)
    if key:
        normalized["key"] = key
    return normalized


def _normalize_neo4j_node(
    value: Any,
    primary_by_label: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    labels = sorted(str(label) for label in value.labels)
    label = labels[0] if labels else ""
    properties = dict(value.items())
    key = _node_key(label, properties, primary_by_label)
    if key:
        return {
            "element": OracleNameSanitizer.clean(label, fallback=label),
            "key": _normalize_value(key, primary_by_label),
        }
    return {
        "element": OracleNameSanitizer.clean(label, fallback=label),
        "properties": _normalize_value(properties, primary_by_label),
    }


def _normalize_neo4j_relationship(
    value: Any,
    primary_by_label: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    rel_type = str(value.type)
    normalized: Dict[str, Any] = {
        "element": OracleNameSanitizer.clean(rel_type, fallback=rel_type),
    }
    properties = dict(value.items())
    if properties:
        normalized["properties"] = _normalize_value(properties, primary_by_label)
    return normalized


def _node_key(
    label: str,
    properties: Dict[str, Any],
    primary_by_label: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    candidates = []
    if primary_by_label:
        candidates.extend(
            [
                primary_by_label.get(label),
                primary_by_label.get(OracleNameSanitizer.clean(label, fallback=label)),
            ]
        )
    candidates.extend(["_id", "vid", "id", f"{label}_id"])
    candidates.extend(sorted(key for key in properties if key.lower().endswith("_id")))
    for candidate in candidates:
        if candidate and candidate in properties:
            return {candidate: properties[candidate]}
    return {}


def _looks_like_path(value: Any) -> bool:
    return hasattr(value, "nodes") and hasattr(value, "relationships")


def _normalize_path(
    value: Any,
    primary_by_label: Dict[str, str] | None = None,
) -> List[Any]:
    nodes = list(value.nodes)
    relationships = list(value.relationships)
    normalized = []
    for index, node in enumerate(nodes):
        normalized.append(_normalize_value(node, primary_by_label))
        if index < len(relationships):
            normalized.append(_normalize_value(relationships[index], primary_by_label))
    return normalized


def _convert_value(value: str, type_name: str) -> Any:
    if value == "":
        return None
    normalized = type_name.upper()
    if normalized in ("INT8", "INT16", "INT32", "INT64", "INTEGER"):
        return int(float(value))
    if normalized in ("FLOAT", "DOUBLE", "FLOAT32", "FLOAT64"):
        return float(value)
    if normalized in ("BOOL", "BOOLEAN"):
        return value.strip().lower() in ("true", "1", "yes", "y")
    if normalized == "DATE":
        parsed = _parse_datetime(value)
        return parsed.date() if isinstance(parsed, datetime) else parsed
    if normalized in ("DATETIME", "TIMESTAMP"):
        parsed = _parse_datetime(value)
        if isinstance(parsed, date) and not isinstance(parsed, datetime):
            return datetime.combine(parsed, datetime.min.time())
        return parsed
    return value


def _parse_datetime(value: str) -> date | datetime | str:
    normalized = value.strip()
    try:
        return datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError:
        pass
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d",
    ):
        try:
            parsed = datetime.strptime(normalized, fmt)
            if fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d"):
                return parsed.date()
            return parsed
        except ValueError:
            continue
    return normalized


def graph_name_for(unit: DatabaseUnit, prefix: str) -> str:
    return OracleNameSanitizer.clean(
        f"{prefix}_{unit.split}_{unit.database}",
        fallback="T2GQL_GRAPH",
    )


def query_status_name(status: QueryStatus) -> str:
    if status == QueryStatus.SUCCESS:
        return "success"
    if status == QueryStatus.NO_RECORD:
        return "no_record"
    if status == QueryStatus.CLIENT_ERROR:
        return "client_error"
    if status == QueryStatus.SERVER_ERROR:
        return "server_error"
    return str(status)


def increment(mapping: Dict[str, int], key: str) -> None:
    mapping[key] = mapping.get(key, 0) + 1


def merge_compare_summaries(summaries: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {
        "databases": 0,
        "considered": 0,
        "matched": 0,
        "failed": 0,
        "skipped": 0,
        "skip_reasons": {},
        "failure_reasons": {},
        "units": [],
    }
    for summary in summaries:
        merged["databases"] += 1
        for key in ("considered", "matched", "failed", "skipped"):
            merged[key] += int(summary.get(key, 0))
        for key, value in summary.get("skip_reasons", {}).items():
            merged["skip_reasons"][key] = merged["skip_reasons"].get(key, 0) + int(value)
        for key, value in summary.get("failure_reasons", {}).items():
            merged["failure_reasons"][key] = merged["failure_reasons"].get(key, 0) + int(value)
        merged["units"].append(summary)
    return merged


def _escape_backticks(value: str) -> str:
    return value.replace("`", "``")


def _safe_identifier(value: str) -> str:
    cleaned = "".join(char if char.isalnum() else "_" for char in value)
    return cleaned[:120] or "constraint_name"


if __name__ == "__main__":
    main()
