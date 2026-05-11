import argparse
import contextlib
import io
import json
from pathlib import Path
import re

from app.impl.oracle_sqlpgq.translator.oracle_sqlpgq_query_translator import (
    OracleSqlPgqQueryTranslator,
)
from app.impl.tugraph_cypher.ast_visitor.tugraph_cypher_ast_visitor import (
    TugraphCypherAstVisitor,
)
from app.impl.tugraph_cypher.translator.tugraph_cypher_query_translator import (
    TugraphCypherQueryTranslator as CypherTranslator,
)

SUPPORTED_SCOPE = (
    "Graph-IL subset: basic MATCH paths, simple property comparisons, RETURN items, "
    "aliases, aggregates, ORDER BY, SKIP, LIMIT, DISTINCT, and relationship hop ranges."
)


def cypher2oracle_sqlpgq(
    query: str,
    graph_name: str = "GRAPH",
    node_label_map: dict[str, list[str]] | None = None,
    edge_label_map: dict[str, list[str]] | None = None,
    property_type_map: dict[str, dict[str, str]] | None = None,
) -> tuple[str, str]:
    """Translate a supported Cypher query into Oracle SQL/PGQ GRAPH_TABLE syntax."""

    union_query = _translate_union_query(
        query,
        graph_name=graph_name,
        node_label_map=node_label_map,
        edge_label_map=edge_label_map,
        property_type_map=property_type_map,
    )
    if union_query is not None:
        return union_query

    query_visitor = TugraphCypherAstVisitor()
    cypher_translator = CypherTranslator()
    oracle_translator = OracleSqlPgqQueryTranslator(
        graph_name=graph_name,
        node_label_map=node_label_map,
        edge_label_map=edge_label_map,
        property_type_map=property_type_map,
    )

    if not cypher_translator.grammar_check(query):
        return "Unable to Translate to Oracle SQL/PGQ", "Not Comply with OpenCypher"

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        success, query_pattern = query_visitor.get_query_pattern(query)
    if not success:
        return "Unable to Translate to Oracle SQL/PGQ", "Graph-IL Not Support"

    try:
        sqlpgq_query = oracle_translator.translate(query_pattern)
    except Exception:
        return "Unable to Translate to Oracle SQL/PGQ", "Graph-IL Not Support"

    if oracle_translator.grammar_check(sqlpgq_query):
        return sqlpgq_query, "Graph-IL Translatable"

    return "Unable to Translate to Oracle SQL/PGQ", "No Related Oracle SQL/PGQ Standard"


def _translate_union_query(
    query: str,
    graph_name: str,
    node_label_map: dict[str, list[str]] | None,
    edge_label_map: dict[str, list[str]] | None,
    property_type_map: dict[str, dict[str, str]] | None,
) -> tuple[str, str] | None:
    branches, operators = _split_top_level_unions(query)
    if len(branches) == 1:
        return None

    translated_branches: list[str] = []
    branch_aliases: list[list[str]] = []
    for branch in branches:
        translated, category = cypher2oracle_sqlpgq(
            branch,
            graph_name=graph_name,
            node_label_map=node_label_map,
            edge_label_map=edge_label_map,
            property_type_map=property_type_map,
        )
        if category != "Graph-IL Translatable":
            return "Unable to Translate to Oracle SQL/PGQ", category
        aliases = _graph_table_column_aliases(translated)
        if not aliases:
            return "Unable to Translate to Oracle SQL/PGQ", "Graph-IL Not Support"
        translated_branches.append(translated)
        branch_aliases.append(aliases)

    output_aliases: list[str] = []
    for aliases in branch_aliases:
        for alias in aliases:
            if alias not in output_aliases:
                output_aliases.append(alias)

    sql_branches = [
        _wrap_union_branch(translated, aliases, output_aliases, index)
        for index, (translated, aliases) in enumerate(
            zip(translated_branches, branch_aliases, strict=True),
            start=1,
        )
    ]
    union_sql = sql_branches[0]
    for operator, branch_sql in zip(operators, sql_branches[1:], strict=True):
        union_sql += f"\n{operator}\n{branch_sql}"

    if OracleSqlPgqQueryTranslator(graph_name=graph_name).grammar_check(union_sql):
        return union_sql, "Graph-IL Translatable"
    return "Unable to Translate to Oracle SQL/PGQ", "No Related Oracle SQL/PGQ Standard"


def _wrap_union_branch(
    translated: str,
    branch_aliases: list[str],
    output_aliases: list[str],
    index: int,
) -> str:
    select_items = [
        alias if alias in branch_aliases else f"NULL AS {alias}"
        for alias in output_aliases
    ]
    return (
        "SELECT "
        + ", ".join(select_items)
        + f"\nFROM (\n{_indent_sql(translated)}\n) union_branch_{index}"
    )


def _indent_sql(sql: str) -> str:
    return "\n".join("  " + line for line in sql.splitlines())


def _graph_table_column_aliases(sql: str) -> list[str]:
    start = re.search(r"\bCOLUMNS\s*\(", sql, flags=re.IGNORECASE)
    if not start:
        return []
    index = start.end()
    depth = 1
    in_single = False
    in_double = False
    body_start = index
    while index < len(sql):
        char = sql[index]
        if char == "'" and not in_double:
            if index + 1 < len(sql) and sql[index + 1] == "'":
                index += 2
                continue
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif not in_single and not in_double:
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    return _projection_aliases(sql[body_start:index])
        index += 1
    return []


def _projection_aliases(columns_body: str) -> list[str]:
    projections = _split_top_level_commas(columns_body)
    aliases = []
    for projection in projections:
        match = re.search(
            r"\bAS\s+([A-Za-z_][A-Za-z0-9_$#]*)\s*$",
            projection.strip(),
            flags=re.IGNORECASE,
        )
        if not match:
            return []
        aliases.append(match.group(1))
    return aliases


def _split_top_level_commas(text: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    in_single = False
    in_double = False
    index = 0
    while index < len(text):
        char = text[index]
        if char == "'" and not in_double:
            if index + 1 < len(text) and text[index + 1] == "'":
                index += 2
                continue
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif not in_single and not in_double:
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            elif char == "," and depth == 0:
                parts.append(text[start:index].strip())
                start = index + 1
        index += 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _split_top_level_unions(query: str) -> tuple[list[str], list[str]]:
    branches: list[str] = []
    operators: list[str] = []
    start = 0
    index = 0
    depth = 0
    in_single = False
    in_double = False
    in_backtick = False
    while index < len(query):
        char = query[index]
        if char == "`" and not in_single and not in_double:
            in_backtick = not in_backtick
            index += 1
            continue
        if char == "'" and not in_double and not in_backtick:
            if index + 1 < len(query) and query[index + 1] == "'":
                index += 2
                continue
            in_single = not in_single
            index += 1
            continue
        if char == '"' and not in_single and not in_backtick:
            in_double = not in_double
            index += 1
            continue
        if in_single or in_double or in_backtick:
            index += 1
            continue
        if char in "([{":
            depth += 1
            index += 1
            continue
        if char in ")]}":
            depth -= 1
            index += 1
            continue
        if depth == 0 and _keyword_at(query, index, "UNION"):
            operator, end = _union_operator_at(query, index)
            branches.append(query[start:index].strip())
            operators.append(operator)
            start = end
            index = end
            continue
        index += 1
    if not operators:
        return [query], []
    branches.append(query[start:].strip())
    if any(not branch for branch in branches):
        return [query], []
    return branches, operators


def _union_operator_at(query: str, index: int) -> tuple[str, int]:
    end = index + len("UNION")
    while end < len(query) and query[end].isspace():
        end += 1
    if _keyword_at(query, end, "ALL"):
        return "UNION ALL", end + len("ALL")
    return "UNION", end


def _keyword_at(text: str, index: int, keyword: str) -> bool:
    end = index + len(keyword)
    if text[index:end].upper() != keyword:
        return False
    before = text[index - 1] if index > 0 else ""
    after = text[end] if end < len(text) else ""
    return not (_is_identifier_char(before) or _is_identifier_char(after))


def _is_identifier_char(char: str) -> bool:
    return bool(char and (char.isalnum() or char in "_$#"))


def _translate_queries(queries: list[str], graph_name: str) -> list[dict[str, str]]:
    output_query_list = []
    for query in queries:
        translated_query, category = cypher2oracle_sqlpgq(query, graph_name=graph_name)
        output_query_list.append(
            {
                "cypher": query,
                "oracle_sqlpgq": translated_query,
                "category": category,
            }
        )
    return output_query_list


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Translate supported Cypher queries into Oracle SQL/PGQ GRAPH_TABLE syntax. "
            f"Supported scope: {SUPPORTED_SCOPE}"
        )
    )
    parser.add_argument("--query", help="Single Cypher query to translate.")
    parser.add_argument("--input", type=Path, help="JSON file containing a list of Cypher queries.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("test_oracle_sqlpgq_query.json"),
        help="Output JSON file when --input is used.",
    )
    parser.add_argument("--graph-name", default="GRAPH", help="Oracle property graph name.")
    args = parser.parse_args()

    if args.query:
        translated_query, category = cypher2oracle_sqlpgq(args.query, graph_name=args.graph_name)
        print(json.dumps({"query": translated_query, "category": category}, indent=2))
        return

    if not args.input:
        parser.error("Provide either --query or --input.")

    with open(args.input, encoding="utf-8") as file:
        query_list = json.load(file)

    output_query_list = _translate_queries(query_list, graph_name=args.graph_name)
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(output_query_list, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
