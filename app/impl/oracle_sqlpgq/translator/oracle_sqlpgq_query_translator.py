from collections import Counter
import re
from typing import Dict, Iterable, List, Tuple

from app.core.clauses.clause import Clause
from app.core.clauses.match_clause import EdgePattern, MatchClause, NodePattern, PathPattern
from app.core.clauses.return_clause import ReturnBody, ReturnClause, ReturnItem, SortItem
from app.core.clauses.where_clause import CompareExpression, WhereClause
from app.core.clauses.with_clause import WithClause
from app.core.translator.query_translator import QueryTranslator
from app.impl.oracle_sqlpgq.utils.sqlpgq import (
    OracleNameSanitizer,
    validate_graph_table_query,
    validate_property_graph_ddl,
)


class OracleSqlPgqQueryTranslator(QueryTranslator):
    """Translate the framework's graph-query IR into Oracle SQL/PGQ."""

    AGGREGATE_FUNCTIONS = {"COUNT", "AVG", "SUM", "MIN", "MAX"}

    def __init__(
        self,
        graph_name: str = "GRAPH",
        node_label_map: Dict[str, List[str]] | None = None,
        edge_label_map: Dict[str, List[str]] | None = None,
        property_type_map: Dict[str, Dict[str, str]] | None = None,
    ):
        self.graph_name = graph_name
        self.node_label_map = node_label_map or {}
        self.edge_label_map = edge_label_map or {}
        self.property_type_map = property_type_map or {}
        self._var_kinds: Dict[str, str] = {}
        self._var_sql_names: Dict[str, str] = {}
        self._var_labels: Dict[str, str] = {}
        self._path_variables: Dict[str, List[Tuple[str, str]]] = {}
        self._path_variable_has_quantifier: Dict[str, bool] = {}
        self._pattern_where_expressions: List[str] = []
        self._auto_node_index = 0
        self._auto_edge_index = 0

    def grammar_check(self, query: str) -> bool:
        normalized = " ".join(str(query or "").strip().split()).upper()
        if "CREATE" in normalized and "PROPERTY GRAPH" in normalized:
            return validate_property_graph_ddl(query)
        return validate_graph_table_query(query)

    def translate(self, query_pattern: List[Clause]) -> str:
        self._reset()
        if any(isinstance(clause, WithClause) for clause in query_pattern):
            return self._translate_supported_with(query_pattern)
        match_clauses: List[MatchClause] = []
        where_expressions: List[CompareExpression] = []
        return_body: ReturnBody | None = None
        distinct = False

        for clause in query_pattern:
            if isinstance(clause, MatchClause):
                match_clauses.append(clause)
            elif isinstance(clause, WhereClause):
                where_expressions.extend(self._as_compare_list(clause.compare_expression_list))
            elif isinstance(clause, ReturnClause):
                return_body = clause.return_body
                distinct = clause.distinct
            elif isinstance(clause, WithClause):
                if return_body is None:
                    return_body = clause.return_body
                where_expressions.extend(self._as_compare_list(clause.compare_expression_list))

        if not match_clauses:
            raise ValueError("Oracle SQL/PGQ translation requires at least one MatchClause.")

        match_parts = [self._translate_match_clause(clause) for clause in match_clauses]
        graph_table_parts = [
            OracleNameSanitizer.quote(self.graph_name, fallback="GRAPH"),
            "MATCH " + ", ".join(match_parts),
        ]

        where_parts = self._where_parts(where_expressions)
        if where_parts:
            graph_table_parts.append(
                "WHERE " + " AND ".join(where_parts)
            )

        aggregate_query = self._has_aggregate(return_body)
        hidden_sort_aliases = self._hidden_sort_aliases(return_body, aggregate_query)
        graph_table_parts.append(
            f"COLUMNS ({self._translate_columns(return_body, aggregate_query)})"
        )

        select_keyword = self._outer_select(
            return_body,
            distinct,
            aggregate_query,
            hidden_sort_aliases,
        )
        query = (
            f"{select_keyword}\n"
            f"FROM GRAPH_TABLE (\n  {' '.join(graph_table_parts)}\n) gt"
        )

        if return_body is not None:
            query += self._outer_group_order_and_paging(return_body, aggregate_query)

        return query

    def _translate_supported_with(self, query_pattern: List[Clause]) -> str:
        with_indexes = [
            index for index, clause in enumerate(query_pattern) if isinstance(clause, WithClause)
        ]
        if len(with_indexes) != 1:
            raise ValueError("Only one-stage WITH pipelines are supported.")

        with_index = with_indexes[0]
        before_with = query_pattern[:with_index]
        after_with = query_pattern[with_index + 1 :]
        with_clause = query_pattern[with_index]
        assert isinstance(with_clause, WithClause)

        if any(isinstance(clause, (MatchClause, WhereClause)) for clause in after_with):
            raise ValueError("WITH followed by MATCH requires staged SQL CTE support.")
        return_clauses = [clause for clause in after_with if isinstance(clause, ReturnClause)]
        if len(return_clauses) != 1:
            raise ValueError("WITH pipeline requires one final RETURN clause.")
        if (
            with_clause.return_body.sort_item_list
            or with_clause.return_body.skip != -1
            or with_clause.return_body.limit != -1
        ):
            raise ValueError("WITH ORDER BY/SKIP/LIMIT is not supported yet.")

        match_clauses = [clause for clause in before_with if isinstance(clause, MatchClause)]
        where_expressions: List[CompareExpression] = []
        for clause in before_with:
            if isinstance(clause, WhereClause):
                where_expressions.extend(self._as_compare_list(clause.compare_expression_list))
        if not match_clauses:
            raise ValueError("WITH translation requires a preceding MATCH.")

        self._reset()
        match_parts = [self._translate_match_clause(clause) for clause in match_clauses]
        graph_table_parts = [
            OracleNameSanitizer.quote(self.graph_name, fallback="GRAPH"),
            "MATCH " + ", ".join(match_parts),
        ]
        where_parts = self._where_parts(where_expressions)
        if where_parts:
            graph_table_parts.append(
                "WHERE " + " AND ".join(where_parts)
            )
        graph_table_parts.append(
            f"COLUMNS ({self._translate_with_projection_columns(with_clause)})"
        )

        return_clause = return_clauses[0]
        select_keyword = self._outer_select_for_with(
            return_clause.return_body,
            return_clause.distinct,
        )
        query = (
            f"{select_keyword}\n"
            f"FROM GRAPH_TABLE (\n  {' '.join(graph_table_parts)}\n) gt"
        )
        query += self._outer_order_and_paging_for_with(return_clause.return_body)
        return query

    def _translate_with_projection_columns(self, with_clause: WithClause) -> str:
        projections = []
        for item in with_clause.return_body.return_item_list:
            if not item.alias:
                raise ValueError("WITH projection requires aliases for Oracle SQL output.")
            if item.function_name:
                raise ValueError("Aggregate WITH projection requires staged SQL CTE support.")
            projections.append(
                f"{self._return_expression(item)} AS {OracleNameSanitizer.alias(item.alias)}"
            )
        if not projections:
            raise ValueError("WITH projection cannot be empty.")
        return ", ".join(projections)

    def _outer_select_for_with(self, return_body: ReturnBody, distinct: bool) -> str:
        select_items = []
        for item in return_body.return_item_list:
            expression = OracleNameSanitizer.alias(item.symbolic_name)
            if item.function_name:
                expression = f"{item.function_name.upper()}({expression})"
            alias = item.alias or item.property or item.symbolic_name
            select_items.append(f"{expression} AS {OracleNameSanitizer.alias(alias)}")
        keyword = "SELECT DISTINCT" if distinct else "SELECT"
        return f"{keyword} " + ", ".join(select_items)

    def _outer_order_and_paging_for_with(self, return_body: ReturnBody) -> str:
        suffix = ""
        if return_body.sort_item_list:
            suffix += "\nORDER BY " + ", ".join(
                self._translate_sort_item(item, return_body) for item in return_body.sort_item_list
            )
        if return_body.skip != -1:
            suffix += f"\nOFFSET {return_body.skip} ROWS"
        if return_body.limit != -1:
            suffix += f"\nFETCH FIRST {return_body.limit} ROWS ONLY"
        return suffix

    def _reset(self) -> None:
        self._var_kinds = {}
        self._var_sql_names = {}
        self._var_labels = {}
        self._path_variables = {}
        self._path_variable_has_quantifier = {}
        self._pattern_where_expressions = []
        self._auto_node_index = 0
        self._auto_edge_index = 0

    def _as_compare_list(self, value) -> List[CompareExpression]:
        if value is None:
            return []
        if isinstance(value, list):
            return [item for item in value if isinstance(item, CompareExpression)]
        if isinstance(value, CompareExpression):
            return [value]
        return []

    def _translate_match_clause(self, match_clause: MatchClause) -> str:
        path_pattern = match_clause.path_pattern
        if isinstance(path_pattern, list):
            return ", ".join(self._translate_path_pattern(item) for item in path_pattern)
        return self._translate_path_pattern(path_pattern)

    def _translate_path_pattern(self, path_pattern: PathPattern) -> str:
        if path_pattern.edge_pattern_list and all(
            edge.direction == "left" for edge in path_pattern.edge_pattern_list
        ):
            return self._translate_reversed_left_path(path_pattern)
        parts: List[str] = []
        if not path_pattern.node_pattern_list:
            raise ValueError("PathPattern must include at least one node pattern.")
        parts.append(self._translate_node_pattern(path_pattern.node_pattern_list[0]))
        for index, edge in enumerate(path_pattern.edge_pattern_list):
            parts.append(self._translate_edge_pattern(edge))
            parts.append(self._translate_node_pattern(path_pattern.node_pattern_list[index + 1]))
        self._register_path_variable(path_pattern)
        return "".join(parts)

    def _translate_reversed_left_path(self, path_pattern: PathPattern) -> str:
        parts: List[str] = []
        reversed_nodes = list(reversed(path_pattern.node_pattern_list))
        reversed_edges = list(reversed(path_pattern.edge_pattern_list))
        parts.append(self._translate_node_pattern(reversed_nodes[0]))
        for index, edge in enumerate(reversed_edges):
            parts.append(
                self._translate_edge_pattern(
                    EdgePattern(
                        edge.symbolic_name,
                        edge.label,
                        edge.property_maps,
                        "right",
                        edge.hop_range,
                    )
                )
            )
            parts.append(self._translate_node_pattern(reversed_nodes[index + 1]))
        self._register_path_variable(path_pattern)
        return "".join(parts)

    def _register_path_variable(self, path_pattern: PathPattern) -> None:
        if not path_pattern.path_variable:
            return
        elements: List[Tuple[str, str]] = []
        for index, node in enumerate(path_pattern.node_pattern_list):
            if node.symbolic_name:
                elements.append(("node", node.symbolic_name))
            if index < len(path_pattern.edge_pattern_list):
                edge = path_pattern.edge_pattern_list[index]
                if edge.symbolic_name:
                    elements.append(("edge", edge.symbolic_name))
        self._path_variables[path_pattern.path_variable] = elements
        self._path_variable_has_quantifier[path_pattern.path_variable] = any(
            edge.hop_range != (-1, -1) for edge in path_pattern.edge_pattern_list
        )

    def _translate_node_pattern(self, node_pattern: NodePattern) -> str:
        variable = node_pattern.symbolic_name or self._next_node_var()
        node_pattern.symbolic_name = variable
        sql_variable = self._declare_variable(variable, "node", node_pattern.label)
        body = sql_variable
        if node_pattern.label:
            body += f" IS {self._label_expression(node_pattern.label, self.node_label_map, 'NODE')}"
        inline_where = self._property_maps_to_where(sql_variable, node_pattern.property_maps)
        if inline_where:
            self._pattern_where_expressions.append(inline_where)
        return f"({body})"

    def _translate_edge_pattern(self, edge_pattern: EdgePattern) -> str:
        variable = edge_pattern.symbolic_name or self._next_edge_var()
        edge_pattern.symbolic_name = variable
        sql_variable = self._declare_variable(variable, "edge", edge_pattern.label)
        body = sql_variable
        if edge_pattern.label:
            body += f" IS {self._label_expression(edge_pattern.label, self.edge_label_map, 'EDGE')}"
        inline_where = self._property_maps_to_where(sql_variable, edge_pattern.property_maps)
        if inline_where:
            self._pattern_where_expressions.append(inline_where)
        edge = f"[{body}]"
        if edge_pattern.direction == "left":
            edge = f"<-{edge}-"
        elif edge_pattern.direction == "right":
            edge = f"-{edge}->"
        else:
            edge = f"-{edge}-"
        if edge_pattern.hop_range != (-1, -1):
            edge += self._hop_quantifier(edge_pattern.hop_range)
        return edge

    def _property_maps_to_where(
        self, variable: str, property_maps: Iterable[Tuple[str, str]]
    ) -> str:
        expressions = []
        for property_name, property_value in property_maps:
            property_ref = OracleNameSanitizer.quote(property_name, fallback="PROP")
            property_value = self._coerce_literal_for_property(
                variable,
                property_name,
                property_value,
            )
            expressions.append(
                f"{variable}.{property_ref} = {property_value}"
            )
        return " AND ".join(expressions)

    def _coerce_literal_for_property(
        self,
        variable: str,
        property_name: str,
        value: str,
    ) -> str:
        property_type = self._property_type(variable, property_name)
        raw_value = str(value).strip()
        if self._is_string_type(property_type) and raw_value.lower() in {"true", "false"}:
            return f"'{raw_value.lower()}'"
        value = self._translate_sql_expression(raw_value)
        if not self._is_string_type(property_type):
            return value
        if re.fullmatch(r"-?\d+(?:\.\d+)?", value):
            return f"'{value}'"
        date_match = re.fullmatch(r"DATE\s+('[^']*')", value, flags=re.IGNORECASE)
        if date_match:
            return date_match.group(1)
        return value

    def _property_type(self, variable: str, property_name: str) -> str:
        source_variable = self._source_variable(variable)
        label = self._var_labels.get(source_variable, "")
        return self.property_type_map.get(label, {}).get(property_name, "")

    def _is_string_type(self, property_type: str) -> bool:
        normalized = str(property_type or "").upper()
        return "CHAR" in normalized or "CLOB" in normalized or normalized == "STRING"

    def _source_variable(self, variable: str) -> str:
        for source, sql_name in self._var_sql_names.items():
            if variable in (source, sql_name):
                return source
        return variable

    def _hop_quantifier(self, hop_range: Tuple[int, int]) -> str:
        lower, upper = hop_range
        if lower == upper:
            return f"{{{lower}}}"
        if lower == -1:
            return f"{{1,{upper}}}"
        if upper == -1:
            return f"{{{lower},}}"
        return f"{{{lower},{upper}}}"

    def _label_expression(
        self,
        source_label: str,
        label_map: Dict[str, List[str]],
        fallback: str,
    ) -> str:
        labels: List[str] = []
        for source_part in source_label.split("|"):
            source_part = source_part.strip()
            if not source_part:
                continue
            labels.extend(label_map.get(source_part, [source_part]))
        if not labels:
            labels = [source_label]
        return " | ".join(OracleNameSanitizer.quote(label, fallback=fallback) for label in labels)

    def _declare_variable(self, variable: str, kind: str, label: str = "") -> str:
        self._var_kinds[variable] = kind
        if label:
            self._var_labels[variable] = label
        self._var_sql_names.setdefault(variable, OracleNameSanitizer.alias(variable))
        return self._var_sql_names[variable]

    def _where_parts(self, where_expressions: List[CompareExpression]) -> List[str]:
        return self._pattern_where_expressions + [
            self._translate_compare_expression(expr) for expr in where_expressions
        ]

    def _translate_columns(
        self,
        return_body: ReturnBody | None,
        aggregate_query: bool = False,
    ) -> str:
        if return_body is None or not return_body.return_item_list:
            variables = [var for var, kind in self._var_kinds.items() if kind == "node"]
            if not variables:
                variables = list(self._var_kinds.keys())
            return ", ".join(
                f"{self._element_id_expression(var)} AS {OracleNameSanitizer.alias(var + '_ID')}"
                for var in variables
            )

        if aggregate_query:
            return self._translate_aggregate_columns(return_body)

        return_aliases = self._resolved_return_aliases(return_body)
        projections = []
        for item, alias in zip(return_body.return_item_list, return_aliases, strict=True):
            if item.symbolic_name in self._path_variables and not item.property:
                projections.extend(self._translate_path_return_item(item, alias))
            else:
                projections.append(self._translate_return_item(item, alias))
        projected_aliases = {
            OracleNameSanitizer.alias(alias)
            for alias in return_aliases
        }
        for sort_item in return_body.sort_item_list:
            if self._is_aggregate_sort(sort_item):
                continue
            alias = OracleNameSanitizer.alias(self._sort_alias(sort_item, return_body))
            if alias in projected_aliases:
                continue
            if sort_item.expression:
                projections.append(
                    f"{self._translate_sql_expression(sort_item.expression)} AS {alias}"
                )
                projected_aliases.add(alias)
        return ", ".join(projections)

    def _translate_path_return_item(self, return_item: ReturnItem, alias: str) -> List[str]:
        if self._path_variable_has_quantifier.get(return_item.symbolic_name):
            raise ValueError(
                "Returning variable-length path values requires grouped path projection support."
            )
        projections = []
        alias_prefix = OracleNameSanitizer.alias(alias or return_item.symbolic_name)
        for kind, variable in self._path_variables.get(return_item.symbolic_name, []):
            expression = "EDGE_ID" if kind == "edge" else "VERTEX_ID"
            sql_variable = self._var_sql_names.get(variable, variable)
            projection_alias = OracleNameSanitizer.alias(f"{alias_prefix}_{variable}_ID")
            projections.append(f"{expression}({sql_variable}) AS {projection_alias}")
        if not projections:
            raise ValueError(f"Path variable {return_item.symbolic_name} is not declared.")
        return projections

    def _translate_return_item(
        self,
        return_item: ReturnItem,
        alias: str | None = None,
    ) -> str:
        expression = self._return_expression(return_item)
        if return_item.function_name.upper() in self.AGGREGATE_FUNCTIONS:
            expression = f"{return_item.function_name.upper()}({expression})"
        alias = alias or self._return_alias(return_item, expression)
        return f"{expression} AS {OracleNameSanitizer.alias(alias)}"

    def _translate_aggregate_columns(self, return_body: ReturnBody) -> str:
        projections: List[Tuple[str, str]] = []
        return_aliases = self._resolved_return_aliases(return_body)
        for item, alias in zip(return_body.return_item_list, return_aliases, strict=True):
            if self._is_complex_aggregate_item(item):
                for expression, expression_alias in self._aggregate_expression_projections(
                    item.expression
                ):
                    projections.append((expression, expression_alias))
            elif self._is_aggregate_item(item):
                argument_expression, argument_alias = self._aggregate_argument_projection(item)
                if argument_expression:
                    projections.append((argument_expression, argument_alias))
            else:
                expression = self._return_expression(item)
                projections.append((expression, alias))

        outer_aliases = {
            OracleNameSanitizer.alias(alias)
            for alias in return_aliases
        }
        for sort_item in return_body.sort_item_list:
            sort_alias = OracleNameSanitizer.alias(self._sort_alias(sort_item, return_body))
            if sort_alias in outer_aliases:
                continue
            if self._is_complex_aggregate_item(sort_item):
                for expression, expression_alias in self._aggregate_expression_projections(
                    sort_item.expression
                ):
                    projections.append((expression, expression_alias))
            elif self._is_aggregate_sort(sort_item):
                argument_expression, argument_alias = self._aggregate_argument_projection(sort_item)
                if argument_expression:
                    projections.append((argument_expression, argument_alias))
            elif sort_item.expression:
                projections.append(
                    (
                        self._translate_sql_expression(sort_item.expression),
                        self._sort_alias(sort_item, return_body),
                    )
                )

        unique: Dict[str, str] = {}
        for expression, alias in projections:
            unique.setdefault(OracleNameSanitizer.alias(alias), expression)
        if not unique:
            unique["dummy_value"] = "1"
        return ", ".join(f"{expression} AS {alias}" for alias, expression in unique.items())

    def _outer_select(
        self,
        return_body: ReturnBody | None,
        distinct: bool,
        aggregate_query: bool,
        hidden_sort_aliases: List[str] | None = None,
    ) -> str:
        hidden_sort_aliases = hidden_sort_aliases or []
        if return_body is None or not aggregate_query:
            if hidden_sort_aliases and return_body is not None:
                select_items = []
                for alias in self._resolved_return_aliases(return_body):
                    select_items.append(OracleNameSanitizer.alias(alias))
                keyword = "SELECT DISTINCT" if distinct else "SELECT"
                return f"{keyword} " + ", ".join(select_items)
            return "SELECT DISTINCT *" if distinct else "SELECT *"
        select_items = []
        for item in return_body.return_item_list:
            if self._is_complex_aggregate_item(item):
                expression = self._outer_aggregate_sql_expression(item.expression)
                alias = self._return_alias(item, expression)
                select_items.append(f"{expression} AS {OracleNameSanitizer.alias(alias)}")
            elif self._is_aggregate_item(item):
                expression = self._outer_aggregate_expression(item)
                alias = self._return_alias(item, expression)
                select_items.append(f"{expression} AS {OracleNameSanitizer.alias(alias)}")
            else:
                expression = self._return_alias(item, self._return_expression(item))
                select_items.append(OracleNameSanitizer.alias(expression))
        keyword = "SELECT DISTINCT" if distinct else "SELECT"
        return f"{keyword} " + ", ".join(select_items)

    def _outer_aggregate_expression(self, item: ReturnItem | SortItem) -> str:
        function_name = item.function_name.upper()
        if function_name == "COUNT" and item.symbolic_name == "*":
            return "COUNT(*)"
        _, argument_alias = self._aggregate_argument_projection(item)
        distinct = ""
        if item.symbolic_name.startswith("DISTINCT "):
            distinct = "DISTINCT "
        return f"{function_name}({distinct}{OracleNameSanitizer.alias(argument_alias)})"

    def _aggregate_argument_projection(self, item: ReturnItem | SortItem) -> Tuple[str, str]:
        if item.function_name.upper() == "COUNT" and item.symbolic_name == "*":
            return "", "dummy_value"
        symbolic_name = item.symbolic_name.replace("DISTINCT ", "", 1)
        argument = ReturnItem(
            symbolic_name=symbolic_name,
            property=item.property,
            alias="",
            function_name="",
            expression=self._aggregate_argument_expression_text(item, symbolic_name),
        )
        expression = self._return_expression(argument)
        return (
            expression,
            item.property or self._default_expression_alias(symbolic_name, expression),
        )

    def _aggregate_expression_projections(self, expression: str) -> List[Tuple[str, str]]:
        projections: List[Tuple[str, str]] = []
        seen = set()
        for variable, property_name in self._property_references(expression):
            alias = self._property_projection_alias(variable, property_name)
            if alias in seen:
                continue
            seen.add(alias)
            property_ref = OracleNameSanitizer.quote(property_name, fallback="PROP")
            sql_variable = self._var_sql_names.get(variable, variable)
            projections.append((f"{sql_variable}.{property_ref}", alias))
        if not projections:
            projections.append(("1", "dummy_value"))
        return projections

    def _outer_aggregate_sql_expression(self, expression: str) -> str:
        translated = self._translate_sql_expression(expression)
        for variable, property_name in self._property_references(expression):
            sql_variable = self._var_sql_names.get(variable, variable)
            property_ref = OracleNameSanitizer.quote(property_name, fallback="PROP")
            alias = OracleNameSanitizer.alias(
                self._property_projection_alias(variable, property_name)
            )
            translated = translated.replace(f"{sql_variable}.{property_ref}", alias)
        return translated

    def _property_references(self, expression: str) -> List[Tuple[str, str]]:
        protected, _ = self._protect_string_literals(expression or "")
        references: List[Tuple[str, str]] = []
        for match in re.finditer(
            r"\b(?P<variable>[A-Za-z_][A-Za-z0-9_]*)\."
            r"(?P<property>[A-Za-z_][A-Za-z0-9_$#-]*)\b",
            protected,
        ):
            variable = match.group("variable")
            property_name = match.group("property")
            if variable in self._var_kinds:
                references.append((variable, property_name))
        return references

    def _property_projection_alias(self, variable: str, property_name: str) -> str:
        return property_name

    def _aggregate_argument_expression_text(
        self,
        item: ReturnItem | SortItem,
        symbolic_name: str,
    ) -> str:
        if not item.expression:
            return symbolic_name
        expression = item.expression.strip()
        if (
            expression.upper().startswith(f"{item.function_name.upper()}(")
            and expression.endswith(")")
        ):
            expression = expression[len(item.function_name) + 1 : -1].strip()
        return expression.replace("DISTINCT ", "", 1)

    def _translate_sort_item(self, sort_item: SortItem, return_body: ReturnBody) -> str:
        alias = self._sort_alias(sort_item, return_body)
        order = f" {sort_item.order.upper()}" if sort_item.order else ""
        return f"{OracleNameSanitizer.alias(alias)}{order}"

    def _sort_alias(self, sort_item: SortItem, return_body: ReturnBody) -> str:
        for return_item in return_body.return_item_list:
            if (
                return_item.symbolic_name == sort_item.symbolic_name
                and return_item.property == sort_item.property
                and return_item.function_name == sort_item.function_name
            ):
                return (
                    self._return_alias(
                        return_item,
                        return_item.expression or return_item.symbolic_name,
                    )
                )
            if sort_item.expression and sort_item.expression == return_item.alias:
                return return_item.alias
            if sort_item.expression and sort_item.expression == return_item.expression:
                return (
                    return_item.alias
                    or return_item.property
                    or self._default_expression_alias(
                        return_item.symbolic_name,
                        sort_item.expression,
                    )
                )
        alias = sort_item.property or sort_item.symbolic_name
        if sort_item.function_name:
            alias = f"{sort_item.function_name}_{alias}"
        return alias

    def _return_alias(self, return_item: ReturnItem, expression: str) -> str:
        return (
            return_item.alias
            or return_item.property
            or self._default_expression_alias(
                return_item.symbolic_name,
                return_item.expression or expression,
            )
        )

    def _return_expression(self, return_item: ReturnItem) -> str:
        if return_item.expression:
            expression = return_item.expression.strip()
            if (
                expression == return_item.symbolic_name
                and return_item.symbolic_name
                and return_item.symbolic_name in self._var_kinds
            ):
                return self._element_id_expression(return_item.symbolic_name)
            if return_item.function_name:
                if return_item.function_name.upper() not in self.AGGREGATE_FUNCTIONS:
                    return self._translate_sql_expression(expression)
                return self._value_expression(return_item.symbolic_name, return_item.property)
            return self._translate_sql_expression(expression)
        return self._value_expression(return_item.symbolic_name, return_item.property)

    def _value_expression(self, symbolic_name: str, property_name: str) -> str:
        symbolic_name = symbolic_name.replace("DISTINCT ", "")
        if symbolic_name == "*":
            return "*"
        sql_symbolic_name = self._var_sql_names.get(symbolic_name, symbolic_name)
        if property_name:
            property_ref = OracleNameSanitizer.quote(property_name, fallback="PROP")
            return f"{sql_symbolic_name}.{property_ref}"
        return self._element_id_expression(symbolic_name)

    def _element_id_expression(self, variable: str) -> str:
        sql_variable = self._var_sql_names.get(variable, variable)
        if self._var_kinds.get(variable) == "edge":
            return f"EDGE_ID({sql_variable})"
        return f"VERTEX_ID({sql_variable})"

    def _translate_compare_expression(self, compare_expression: CompareExpression) -> str:
        raw_expression = getattr(compare_expression, "raw_expression", "")
        if raw_expression:
            return self._translate_sql_expression(raw_expression)
        prop = ""
        if compare_expression.property:
            prop = f".{OracleNameSanitizer.quote(compare_expression.property, fallback='PROP')}"
        operator = {
            "equal": "=",
            "neq": "<>",
            "less": "<",
            "greater": ">",
            "leq": "<=",
            "geq": ">=",
        }.get(compare_expression.comparison_type, "=")
        sql_symbolic_name = self._var_sql_names.get(
            compare_expression.symbolic_name,
            compare_expression.symbolic_name,
        )
        return f"{sql_symbolic_name}{prop} {operator} {compare_expression.comparison_value}"

    def _translate_sql_expression(self, expression: str) -> str:
        protected, literals = self._protect_string_literals(expression)
        protected = re.sub(
            r"\bdate\s*\(\s*__SQL_LITERAL_(\d+)__\s*\)",
            lambda match: f"DATE __SQL_LITERAL_{match.group(1)}__",
            protected,
            flags=re.IGNORECASE,
        )
        protected = re.sub(
            r"\bdate\s*\(\s*\)",
            "TRUNC(CURRENT_DATE)",
            protected,
            flags=re.IGNORECASE,
        )
        protected = self._translate_date_property_extractors(protected)
        protected = self._translate_property_extractors(protected)
        protected = re.sub(
            r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_$#-]*)\b",
            lambda match: (
                f"{self._var_sql_names.get(match.group(1), match.group(1))}."
                f"{OracleNameSanitizer.quote(match.group(2), fallback='PROP')}"
            ),
            protected,
        )
        protected = self._coerce_typed_property_comparisons(protected)
        protected = self._translate_element_comparisons(protected)
        protected = self._translate_split_size(protected)
        protected = self._translate_modulo(protected)
        protected = re.sub(r"(?<!')\btrue\b(?!')", "1", protected, flags=re.IGNORECASE)
        protected = re.sub(r"(?<!')\bfalse\b(?!')", "0", protected, flags=re.IGNORECASE)
        protected = re.sub(r"\bsize\s*\(", "LENGTH(", protected, flags=re.IGNORECASE)
        protected = re.sub(r"\btoFloat\s*\(", "TO_NUMBER(", protected, flags=re.IGNORECASE)
        protected = self._translate_to_integer(protected)
        protected = re.sub(r"\bCOUNT\s*\(\s*\)", "COUNT(*)", protected, flags=re.IGNORECASE)
        protected = self._restore_string_literals(protected, literals)
        return protected

    def _translate_to_integer(self, expression: str) -> str:
        pattern = re.compile(r"\btoInteger\s*\(", flags=re.IGNORECASE)
        while True:
            match = pattern.search(expression)
            if not match:
                return expression
            body_start = match.end()
            body_end = self._matching_paren_index(expression, body_start - 1)
            if body_end == -1:
                return expression
            body = expression[body_start:body_end]
            replacement = f"CAST({body} AS INTEGER)"
            expression = expression[: match.start()] + replacement + expression[body_end + 1 :]

    def _matching_paren_index(self, expression: str, open_index: int) -> int:
        depth = 0
        index = open_index
        while index < len(expression):
            char = expression[index]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    return index
            index += 1
        return -1

    def _translate_date_property_extractors(self, expression: str) -> str:
        def replace(match: re.Match) -> str:
            variable = match.group("variable")
            property_name = match.group("property")
            field = match.group("field").upper()
            property_ref = (
                f"{self._var_sql_names.get(variable, variable)}."
                f"{OracleNameSanitizer.quote(property_name, fallback='PROP')}"
            )
            if self._is_string_type(self._property_type(variable, property_name)):
                property_ref = f"TO_DATE({property_ref}, 'YYYY-MM-DD')"
            return f"EXTRACT({field} FROM {property_ref})"

        return re.sub(
            r"\bdate\s*\(\s*(?P<variable>[A-Za-z_][A-Za-z0-9_]*)\."
            r"(?P<property>[A-Za-z_][A-Za-z0-9_$#-]*)\s*\)\."
            r"(?P<field>year|month|day)\b",
            replace,
            expression,
            flags=re.IGNORECASE,
        )

    def _translate_property_extractors(self, expression: str) -> str:
        def replace(match: re.Match) -> str:
            variable = match.group("variable")
            property_name = match.group("property")
            field = match.group("field").upper()
            property_ref = (
                f"{self._var_sql_names.get(variable, variable)}."
                f"{OracleNameSanitizer.quote(property_name, fallback='PROP')}"
            )
            if field in {"YEAR", "MONTH", "DAY"}:
                if self._is_string_type(self._property_type(variable, property_name)):
                    property_ref = f"TO_DATE({property_ref}, 'YYYY-MM-DD')"
                return f"EXTRACT({field} FROM {property_ref})"
            return match.group(0)

        return re.sub(
            r"\b(?P<variable>[A-Za-z_][A-Za-z0-9_]*)\."
            r"(?P<property>[A-Za-z_][A-Za-z0-9_$#-]*)\."
            r"(?P<field>year|month|day)\b",
            replace,
            expression,
            flags=re.IGNORECASE,
        )

    def _translate_split_size(self, expression: str) -> str:
        return re.sub(
            r"\bsize\s*\(\s*split\s*\((?P<body>[^,]+),\s*__SQL_LITERAL_\d+__\s*\)\s*\)",
            lambda match: f"REGEXP_COUNT({match.group('body').strip()}, '\\\\S+')",
            expression,
            flags=re.IGNORECASE,
        )

    def _translate_modulo(self, expression: str) -> str:
        expression = re.sub(
            r"(?P<left>EXTRACT\(.+\))\s*%\s*(?P<right>-?\d+(?:\.\d+)?)",
            lambda match: f"MOD({match.group('left')}, {match.group('right')})",
            expression,
        )
        return re.sub(
            r"(?P<left>[A-Za-z_][A-Za-z0-9_.$#\"]*)\s*%\s*"
            r"(?P<right>-?\d+(?:\.\d+)?)",
            lambda match: f"MOD({match.group('left')}, {match.group('right')})",
            expression,
        )

    def _coerce_typed_property_comparisons(self, expression: str) -> str:
        def replace_date(match: re.Match) -> str:
            variable = match.group("variable")
            property_name = match.group("property")
            if self._is_string_type(self._property_type(variable, property_name)):
                return (
                    f'{variable}."{property_name}" {match.group("operator")} '
                    f'__SQL_LITERAL_{match.group("literal")}__'
                )
            return match.group(0)

        expression = re.sub(
            r'\b(?P<variable>[A-Za-z_][A-Za-z0-9_]*)\."(?P<property>[^"]+)"\s*'
            r'(?P<operator><=|>=|<>|=|<|>)\s*DATE\s+__SQL_LITERAL_(?P<literal>\d+)__',
            replace_date,
            expression,
            flags=re.IGNORECASE,
        )

        def replace_boolean(match: re.Match) -> str:
            variable = match.group("variable")
            property_name = match.group("property")
            boolean = match.group("boolean").lower()
            if self._is_string_type(self._property_type(variable, property_name)):
                return (
                    f'{variable}."{property_name}" {match.group("operator")} '
                    f"'{boolean}'"
                )
            return match.group(0)

        expression = re.sub(
            r'\b(?P<variable>[A-Za-z_][A-Za-z0-9_]*)\."(?P<property>[^"]+)"\s*'
            r'(?P<operator><=|>=|<>|=|<|>)\s*(?P<boolean>true|false)\b',
            replace_boolean,
            expression,
            flags=re.IGNORECASE,
        )

        def replace_number(match: re.Match) -> str:
            variable = match.group("variable")
            property_name = match.group("property")
            if self._is_string_type(self._property_type(variable, property_name)):
                return (
                    f'{variable}."{property_name}" {match.group("operator")} '
                    f"'{match.group('number')}'"
                )
            return match.group(0)

        return re.sub(
            r'\b(?P<variable>[A-Za-z_][A-Za-z0-9_]*)\."(?P<property>[^"]+)"\s*'
            r'(?P<operator><=|>=|<>|=|<|>)\s*(?P<number>-?\d+(?:\.\d+)?)\b',
            replace_number,
            expression,
        )

    def _translate_element_comparisons(self, expression: str) -> str:
        def replace(match: re.Match) -> str:
            left = match.group("left")
            right = match.group("right")
            operator = match.group("operator")
            left_source = self._source_variable(left)
            right_source = self._source_variable(right)
            if left_source not in self._var_kinds or right_source not in self._var_kinds:
                return match.group(0)
            if self._var_kinds[left_source] != self._var_kinds[right_source]:
                return match.group(0)
            function_name = (
                "EDGE_EQUAL"
                if self._var_kinds[left_source] == "edge"
                else "VERTEX_EQUAL"
            )
            comparison = f"{function_name}({left}, {right})"
            return comparison if operator == "=" else f"NOT {comparison}"

        return re.sub(
            r"\b(?P<left>[A-Za-z_][A-Za-z0-9_]*)\s*(?P<operator>=|<>)\s*"
            r"(?P<right>[A-Za-z_][A-Za-z0-9_]*)\b",
            replace,
            expression,
        )

    def _protect_string_literals(self, expression: str) -> tuple[str, List[str]]:
        literals: List[str] = []
        result: List[str] = []
        i = 0
        while i < len(expression):
            if expression[i] != "'":
                result.append(expression[i])
                i += 1
                continue
            start = i
            i += 1
            while i < len(expression):
                if expression[i] == "'" and i + 1 < len(expression) and expression[i + 1] == "'":
                    i += 2
                    continue
                if expression[i] == "'":
                    i += 1
                    break
                i += 1
            placeholder = f"__SQL_LITERAL_{len(literals)}__"
            literals.append(expression[start:i])
            result.append(placeholder)
        return "".join(result), literals

    def _restore_string_literals(self, expression: str, literals: List[str]) -> str:
        for index, literal in enumerate(literals):
            expression = expression.replace(f"__SQL_LITERAL_{index}__", literal)
        return expression

    def _default_expression_alias(self, symbolic_name: str, expression: str) -> str:
        if symbolic_name == "*":
            return "COUNT_VALUE"
        if symbolic_name in self._path_variables:
            return symbolic_name
        if symbolic_name.startswith("DISTINCT "):
            symbolic_name = symbolic_name.replace("DISTINCT ", "", 1)
        if symbolic_name in self._var_kinds and expression.strip() in (
            symbolic_name,
            self._element_id_expression(symbolic_name),
        ):
            return f"{symbolic_name}_VALUE"
        compact = re.sub(r"[^A-Za-z0-9_]+", "_", symbolic_name or expression).strip("_")
        return compact or "VALUE"

    def _outer_group_order_and_paging(
        self,
        return_body: ReturnBody,
        aggregate_query: bool = False,
    ) -> str:
        suffix = ""
        if aggregate_query:
            group_aliases = []
            for item in return_body.return_item_list:
                if not self._is_aggregate_item(item):
                    group_aliases.append(
                        OracleNameSanitizer.alias(
                            self._return_alias(item, self._return_expression(item))
                        )
                    )
            if group_aliases:
                suffix += "\nGROUP BY " + ", ".join(dict.fromkeys(group_aliases))
        if return_body.sort_item_list:
            suffix += "\nORDER BY " + ", ".join(
                self._translate_sort_item(item, return_body) for item in return_body.sort_item_list
            )
        if return_body.skip != -1:
            suffix += f"\nOFFSET {return_body.skip} ROWS"
        if return_body.limit != -1:
            suffix += f"\nFETCH FIRST {return_body.limit} ROWS ONLY"
        return suffix

    def _hidden_sort_aliases(
        self,
        return_body: ReturnBody | None,
        aggregate_query: bool,
    ) -> List[str]:
        if return_body is None or aggregate_query:
            return []
        projected_aliases = set(self._resolved_return_aliases(return_body))
        hidden_aliases = []
        for sort_item in return_body.sort_item_list:
            alias = OracleNameSanitizer.alias(self._sort_alias(sort_item, return_body))
            if alias not in projected_aliases:
                hidden_aliases.append(alias)
        return hidden_aliases

    def _resolved_return_aliases(self, return_body: ReturnBody) -> List[str]:
        raw_aliases = [
            self._return_alias(item, self._return_expression(item))
            for item in return_body.return_item_list
        ]
        alias_counts = Counter(OracleNameSanitizer.alias(alias) for alias in raw_aliases)
        seen: Dict[str, int] = {}
        resolved = []
        for item, raw_alias in zip(return_body.return_item_list, raw_aliases, strict=True):
            alias = OracleNameSanitizer.alias(raw_alias)
            if alias_counts[alias] == 1:
                resolved.append(alias)
                continue
            expression = self._return_expression(item)
            base = OracleNameSanitizer.alias(
                self._default_expression_alias(item.symbolic_name, expression)
            )
            index = seen.get(base, 0)
            seen[base] = index + 1
            resolved.append(base if index == 0 else f"{base}_{index + 1}")
        return resolved

    def _has_aggregate(self, return_body: ReturnBody | None) -> bool:
        if return_body is None:
            return False
        return any(self._is_aggregate_item(item) for item in return_body.return_item_list)

    def _is_aggregate_item(self, item: ReturnItem | SortItem) -> bool:
        return (
            item.function_name.upper() in self.AGGREGATE_FUNCTIONS
            or self._is_complex_aggregate_item(item)
        )

    def _is_complex_aggregate_item(self, item: ReturnItem | SortItem) -> bool:
        return (
            not item.function_name
            and bool(item.expression)
            and self._contains_aggregate_function(item.expression)
        )

    def _is_aggregate_sort(self, item: SortItem) -> bool:
        return self._is_aggregate_item(item)

    def _contains_aggregate_function(self, expression: str) -> bool:
        return bool(
            re.search(
                r"\b(?:COUNT|AVG|SUM|MIN|MAX)\s*\(",
                expression or "",
                flags=re.IGNORECASE,
            )
        )

    def _next_node_var(self) -> str:
        self._auto_node_index += 1
        return f"n{self._auto_node_index}"

    def _next_edge_var(self) -> str:
        self._auto_edge_index += 1
        return f"e{self._auto_edge_index}"
