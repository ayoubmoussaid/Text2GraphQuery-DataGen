import json
from pathlib import Path

from dataset_prep.analyze_failures import failure_signature, unsupported_query_signature
from dataset_prep.compare_oracle_neo4j_results import (
    DatasetNeo4jLoader,
    is_nondeterministic_limit_without_order,
    is_order_by_limit_query,
    normalize_rows,
    select_records_for_range,
)
from dataset_prep.discover import DatabaseUnit, discover_database_units, source_query
from dataset_prep.oracle_loader import DatasetOracleLoader
from dataset_prep.translate_validate import detect_unsupported_features, graph_name_for


def test_discover_train_and_dev_layouts(tmp_path: Path):
    train_db = tmp_path / "train" / "movies"
    train_config = train_db / "cypher" / "movies_tugraph_new"
    train_config.mkdir(parents=True)
    (train_db / "4_level_results_ek_results.json").write_text("[]", encoding="utf-8")
    (train_config / "import_config.json").write_text(
        json.dumps({"schema": [], "files": []}),
        encoding="utf-8",
    )

    dev_db = tmp_path / "dev" / "Disney" / "Cypher"
    dev_config = dev_db / "disney__tugraph2"
    dev_config.mkdir(parents=True)
    (dev_db / "disney_cypher.json").write_text("[]", encoding="utf-8")
    (dev_db / "4_level_results_ek_results_refined.json").write_text("[]", encoding="utf-8")
    (dev_config / "import_config.json").write_text(
        json.dumps({"schema": [], "files": []}),
        encoding="utf-8",
    )

    units = discover_database_units(tmp_path, ["train", "dev"])

    assert [(unit.split, unit.database) for unit in units] == [
        ("dev", "Disney"),
        ("train", "movies"),
    ]
    assert units[0].query_path.name == "disney_cypher.json"
    assert units[1].csv_root == train_config


def test_source_query_prefers_cypher_then_gql():
    assert source_query({"initial_cypher": "MATCH (n) RETURN n", "initial_gql": "gql"}) == (
        "initial_cypher",
        "MATCH (n) RETURN n",
    )
    assert source_query({"initial_gql": "MATCH (n) RETURN n"}) == (
        "initial_gql",
        "MATCH (n) RETURN n",
    )


def test_graph_name_is_oracle_safe():
    unit = DatabaseUnit(
        split="test",
        database="Manufacturing_BOM(Bill_Of_Materials)",
        root=Path("."),
        query_path=Path("query.json"),
        import_config_path=Path("import_config.json"),
        csv_root=Path("."),
    )
    name = graph_name_for(unit, "T2GQL")
    assert "-" not in name
    assert "(" not in name
    assert len(name) <= 128


def test_detect_unsupported_oracle_sqlpgq_features():
    assert not detect_unsupported_features("MATCH p = (a)-[e]->(b) RETURN p")
    assert not detect_unsupported_features("MATCH (a)-[:A|B]->(b) RETURN b")
    assert not detect_unsupported_features(
        "MATCH (a) RETURN count(CASE WHEN a.type = 'x' THEN 1 ELSE NULL END)"
    )
    assert "case_label_predicate" in detect_unsupported_features(
        "MATCH (a) RETURN CASE WHEN a:ACCOUNT THEN 1 ELSE 0 END"
    )
    assert not detect_unsupported_features(
        "OPTIONAL MATCH (a)-->(b) RETURN a.name, count(b)"
    )
    assert not detect_unsupported_features(
        "MATCH (g:Group) WITH g ORDER BY g.created_date DESC LIMIT 1 "
        "OPTIONAL MATCH (g)<-[:BelongsTo]-(u:User) RETURN g.name, COUNT(u)"
    )
    assert not detect_unsupported_features(
        "MATCH (q:Question) OPTIONAL MATCH (q)<-[:COMMENTED_ON]-(c:Comment) "
        "WITH q, count(c) AS commentCount RETURN q.title, commentCount"
    )
    assert not detect_unsupported_features(
        "MATCH (q:Question) OPTIONAL MATCH (q)<-[:COMMENTED_ON]-(c:Comment) "
        "WHERE c IS NULL RETURN q.title"
    )
    assert not detect_unsupported_features(
        "MATCH (a) OPTIONAL MATCH (a)--(b) RETURN b"
    )
    assert "optional_match" in detect_unsupported_features(
        "OPTIONAL MATCH (a)-->(b) OPTIONAL MATCH (b)-->(c) RETURN c"
    )
    assert "optional_match" in detect_unsupported_features(
        "MATCH (a) OPTIONAL MATCH (b) RETURN b"
    )
    assert "optional_match" in detect_unsupported_features(
        "OPTIONAL MATCH (a)-->(b) RETURN count(*)"
    )
    assert "multiple_with" in detect_unsupported_features(
        "MATCH (a) WITH a MATCH (a)-->(b) WITH b RETURN b"
    )
    assert "unwind" not in detect_unsupported_features(
        "MATCH (q:Question {title: 'use UNWIND and FOREACH safely'}) RETURN q.title"
    )
    assert "multiple_with" not in detect_unsupported_features(
        'MATCH (q:Question {title: "WITH examples in a title"}) RETURN q.title'
    )
    assert "expensive_variable_length_path" in detect_unsupported_features(
        "MATCH (a:ACCOUNT)-[*..10]-(t:TRANSACTION) RETURN t LIMIT 1"
    )
    assert "expensive_variable_length_path" not in detect_unsupported_features(
        "MATCH (person:PERSON)-[:KNOWS*..3]->(friend:PERSON) RETURN friend"
    )


def test_failure_analysis_groups_unsupported_query_shapes():
    def record(query: str) -> dict:
        return {
            "oracle_validation_status": "unsupported",
            "oracle_translation_category": "Graph-IL Not Support",
            "oracle_source_query": query,
            "oracle_unsupported_features": [],
        }

    assert failure_signature(
        record("MATCH (a) WITH a MATCH (a)-->(b) WITH a RETURN a")
    ) == "multiple_with_skipped"
    multi_with_optional = record(
        "MATCH (a) WITH a OPTIONAL MATCH (a)-->(b) WITH a RETURN a"
    )
    multi_with_optional["oracle_unsupported_features"] = ["optional_match"]
    assert failure_signature(multi_with_optional) == "multiple_with_skipped"
    standalone_optional = record("OPTIONAL MATCH (a)-->(b) RETURN a")
    standalone_optional["oracle_unsupported_features"] = ["optional_match"]
    assert failure_signature(standalone_optional) == "standalone_optional_match"
    optional_after_binding = record("MATCH (a) OPTIONAL MATCH (a)-->(b) RETURN a")
    optional_after_binding["oracle_unsupported_features"] = ["optional_match"]
    assert failure_signature(optional_after_binding) == "optional_match_left_join_required"
    assert failure_signature(
        record("MATCH p=(a)-[:KNOWS*1..3]->(b) RETURN p")
    ) == "path_variable_return"
    assert failure_signature(
        record("MATCH (p:Policy) RETURN AVG(p.effective_date) AS value")
    ) == "temporal_numeric_aggregate"
    assert failure_signature(
        record("MATCH (a)-[e]->(b) RETURN count(DISTINCT e.bad_alias)")
    ) == "invalid_schema_property"
    assert failure_signature(
        record("MATCH (a:Assertion) WHERE a.contradiction_severity > 1 RETURN a")
    ) == "invalid_schema_property"
    assert failure_signature(
        record(
            "MATCH (g:Group) WITH g ORDER BY g.created_date DESC LIMIT 1 "
            "OPTIONAL MATCH (g)<-[:BelongsTo]-(u:User) RETURN g.name, COUNT(u)"
        )
    ) == "invalid_schema_property"
    assert failure_signature(
        record("MATCH (s:Source) RETURN s.name")
    ) == "invalid_schema_property"
    assert failure_signature(
        record("MATCH p=(n1:Resource)-[e]-(n2:Policy) WHERE n2.sensitivity_level <> 'Internal' RETURN p")
    ) == "invalid_schema_property"
    assert unsupported_query_signature(
        "MATCH (s:Supplier)-[:SUPPLIES]->(p:Product)-[:ORDERS]->(o:Order) "
        "WITH s.supplierID AS supplierID, COUNT(DISTINCT o.shipCity) AS cityCount "
        "WHERE cityCount > 3 RETURN supplierID"
    ) != "multi_pattern_match"


def test_failure_analysis_uses_manifest_for_invalid_schema(tmp_path: Path):
    import_config = tmp_path / "import_config.json"
    import_config.write_text(
        json.dumps(
            {
                "schema": [
                    {
                        "label": "Product",
                        "type": "VERTEX",
                        "properties": [{"name": "productName"}, {"name": "productID"}],
                    },
                    {
                        "label": "Order",
                        "type": "VERTEX",
                        "properties": [{"name": "freight"}, {"name": "orderDate"}],
                    },
                    {
                        "label": "ORDERS",
                        "type": "EDGE",
                        "properties": [{"name": "quantity"}],
                        "constraints": [["Order", "Product"]],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    def record(query: str) -> dict:
        return {
            "oracle_validation_status": "unsupported",
            "oracle_translation_category": "Graph-IL Not Support",
            "oracle_source_query": query,
            "oracle_unsupported_features": [],
            "oracle_dataset_meta": {"import_config": str(import_config)},
        }

    assert failure_signature(
        record("MATCH (p:Product)-[:ORDERS]->(o:Order) RETURN o.freight")
    ) == "invalid_schema_direction"
    assert failure_signature(
        record("MATCH (p:Product) RETURN p.missingProperty")
    ) == "invalid_schema_property"


def test_loader_converts_oracle_date_values():
    loader = DatasetOracleLoader.__new__(DatasetOracleLoader)

    assert loader._convert_value("2015-12-29", "DATE").isoformat() == "2015-12-29"
    assert (
        loader._convert_value("2025-05-18 12:14:48", "TIMESTAMP").isoformat()
        == "2025-05-18T12:14:48"
    )
    assert loader._convert_value("", "DATE") is None
    assert loader._convert_value("True", "NUMBER(1)") == 1
    assert loader._convert_value("False", "NUMBER(1)") == 0
    assert loader._convert_value("abcde", "VARCHAR2(3)") == "abc"
    assert loader._convert_value("ééé", "VARCHAR2(4)") == "éé"


def test_compare_normalizes_temporal_strings_and_numeric_precision():
    oracle_rows = [{"created_at": "2025-01-01T12:00:00", "score": 1}]
    neo4j_rows = [{"created_at": "2025-01-01T12:00:00.000000000", "score": 1.0}]

    assert normalize_rows(oracle_rows) == normalize_rows(neo4j_rows)
    assert normalize_rows([{"allocation": 0.764800012112}]) == normalize_rows(
        [{"allocation": 0.7648}]
    )
    assert normalize_rows([{"epoch": 1613446786}]) == normalize_rows(
        [{"epoch": 1613446786.0000002}]
    )
    assert normalize_rows([{"date_value": "2025-01-01T00:00:00"}]) == normalize_rows(
        [{"date_value": "2025-01-01"}]
    )


def test_compare_normalizes_oracle_and_neo4j_node_identity():
    class FakeNode:
        labels = {"director"}

        def items(self):
            return {
                "_id": 57,
                "name": "Pinocchio",
                "director": "Ben Sharpsteen",
            }.items()

    oracle_rows = [
        {
            "director": {
                "ELEM_TABLE": "director",
                "GRAPH_NAME": "G",
                "GRAPH_OWNER": "SYSTEM",
                "KEY_VALUE": {"_id": 57},
            }
        }
    ]
    neo4j_rows = [{"director": FakeNode()}]

    assert normalize_rows(oracle_rows, {"director": "_id"}) == normalize_rows(
        neo4j_rows,
        {"director": "_id"},
    )


def test_compare_normalizes_single_neo4j_path_to_flat_element_sequence():
    class FakeNode:
        def __init__(self, label: str, key: str, value: str):
            self.labels = {label}
            self._props = {key: value}

        def items(self):
            return self._props.items()

    class FakeRelationship:
        type = "POSTS"

        def items(self):
            return {}.items()

    class FakePath:
        nodes = [FakeNode("USER", "user_id", "U000001"), FakeNode("POST", "post_id", "P000001")]
        relationships = [FakeRelationship()]

    oracle_rows = [
        {
            "p_n1_ID": {"ELEM_TABLE": "USER", "KEY_VALUE": {"user_id": "U000001"}},
            "p_e1_ID": {"ELEM_TABLE": "POSTS", "KEY_VALUE": {}},
            "p_x_ID": {"ELEM_TABLE": "POST", "KEY_VALUE": {"post_id": "P000001"}},
        }
    ]
    neo4j_rows = [{"p": FakePath()}]

    assert normalize_rows(oracle_rows, {"USER": "user_id", "POST": "post_id"}) == normalize_rows(
        neo4j_rows,
        {"USER": "user_id", "POST": "post_id"},
    )


def test_compare_detects_nondeterministic_limit_without_order_by():
    assert is_nondeterministic_limit_without_order("MATCH (n) RETURN n LIMIT 10")
    assert not is_nondeterministic_limit_without_order(
        "MATCH (n) RETURN n ORDER BY n.name LIMIT 10"
    )
    assert not is_nondeterministic_limit_without_order(
        "MATCH (n {text: 'ORDER BY words LIMIT examples'}) RETURN n"
    )
    assert is_order_by_limit_query("MATCH (n) RETURN n ORDER BY n.name LIMIT 10")


def test_compare_selects_offset_query_ranges():
    records = [{"id": index} for index in range(5)]

    assert select_records_for_range(records, query_offset=2, limit_queries=2) == [
        {"id": 2},
        {"id": 3},
    ]
    assert select_records_for_range(records, query_offset=3) == [{"id": 3}, {"id": 4}]
    assert select_records_for_range(records, query_offset=-10, limit_queries=1) == [{"id": 0}]


def test_neo4j_compare_prepares_string_backed_boolean_literals():
    loader = DatasetNeo4jLoader.__new__(DatasetNeo4jLoader)
    loader.property_types_by_label = {
        "Question": {"answered": "STRING"},
        "Answer": {"is_accepted": "STRING"},
        "Product": {"discontinued": "STRING"},
        "Role": {"is_compliant": "BOOL"},
    }

    assert loader.prepare_query(
        "MATCH (q:Question {answered: true}) RETURN q"
    ) == "MATCH (q:Question {answered: 'true'}) RETURN q"
    assert loader.prepare_query(
        "MATCH (p:Product) WHERE p.discontinued = false RETURN p"
    ) == "MATCH (p:Product) WHERE p.discontinued = 'false' RETURN p"
    assert loader.prepare_query(
        "MATCH (q:Question) WHERE NOT q.answered RETURN q"
    ) == "MATCH (q:Question) WHERE q.answered = 'false' RETURN q"
    assert loader.prepare_query(
        "MATCH (r:Role) WHERE r.is_compliant = true RETURN r"
    ) == "MATCH (r:Role) WHERE r.is_compliant = true RETURN r"


def test_neo4j_compare_prepares_string_backed_date_comparisons():
    loader = DatasetNeo4jLoader.__new__(DatasetNeo4jLoader)
    loader.property_types_by_label = {
        "Director": {"died": "STRING"},
        "Movie": {"release_date": "STRING"},
        "Question": {"createdAt": "STRING"},
        "Event": {"event_date": "DATE"},
    }

    assert loader.prepare_query(
        "MATCH (d:Director) WHERE d.died > date('2000-01-01') RETURN d"
    ) == "MATCH (d:Director) WHERE date(d.died) > date('2000-01-01') RETURN d"
    assert loader.prepare_query(
        "MATCH (m:Movie) WHERE date('2000-01-01') > m.release_date RETURN m"
    ) == "MATCH (m:Movie) WHERE date('2000-01-01') > date(m.release_date) RETURN m"
    assert loader.prepare_query(
        "MATCH (e:Event) WHERE e.event_date > date('2000-01-01') RETURN e"
    ) == "MATCH (e:Event) WHERE e.event_date > date('2000-01-01') RETURN e"
    assert loader.prepare_query(
        "MATCH (q:Question) WHERE date(q.createdAt).day = 1 RETURN q"
    ) == "MATCH (q:Question) WHERE datetime(q.createdAt).day = 1 RETURN q"
    assert loader.prepare_query(
        "MATCH (q:Question) WHERE q.createdAt.day = 1 RETURN q"
    ) == "MATCH (q:Question) WHERE datetime(q.createdAt).day = 1 RETURN q"


def test_neo4j_compare_prepares_string_backed_numeric_comparisons():
    loader = DatasetNeo4jLoader.__new__(DatasetNeo4jLoader)
    loader.property_types_by_label = {
        "Answer": {"uuid": "STRING"},
        "Product": {"unitsOnOrder": "STRING", "reorderLevel": "INT64"},
        "Order": {"freight": "STRING"},
    }

    assert loader.prepare_query(
        "MATCH (p:Product) WHERE p.unitsOnOrder > 50 RETURN p"
    ) == "MATCH (p:Product) WHERE p.unitsOnOrder > '50' RETURN p"
    assert loader.prepare_query(
        "MATCH (o:Order) WHERE 100 < o.freight RETURN o"
    ) == "MATCH (o:Order) WHERE '100' < o.freight RETURN o"
    assert loader.prepare_query(
        "MATCH (p:Product) WHERE p.reorderLevel > 50 RETURN p"
    ) == "MATCH (p:Product) WHERE p.reorderLevel > 50 RETURN p"
    assert loader.prepare_query(
        "MATCH (a:Answer {uuid: 69273049}) RETURN a"
    ) == "MATCH (a:Answer {uuid: '69273049'}) RETURN a"


def test_loader_uses_vertex_file_when_edge_label_collides():
    loader = DatasetOracleLoader.__new__(DatasetOracleLoader)
    loader.config = {
        "files": [
            {"label": "zip_data", "path": "zip_data.csv", "columns": ["_id"]},
            {
                "label": "zip_data",
                "path": "statezip_dataCBSA.csv",
                "SRC_ID": "state",
                "DST_ID": "CBSA",
                "columns": ["SRC_ID", "DST_ID"],
            },
        ]
    }
    loader.manifest = {
        "vertices": [{"label": "zip_data", "table": "zip_data", "columns": []}],
        "edges": [],
    }
    calls = []

    def fake_load_file(item, file_item, is_edge=False):
        calls.append((item["label"], file_item["path"], is_edge))
        return 1

    loader._load_file = fake_load_file

    assert loader._load_csv_files() == {"zip_data": 1}
    assert calls == [("zip_data", "zip_data.csv", False)]


def test_loader_exposes_oracle_graph_label_map_for_collisions():
    loader = DatasetOracleLoader.__new__(DatasetOracleLoader)
    loader.manifest = {
        "vertices": [{"label": "book", "graph_label": "book"}],
        "edges": [
            {"label": "book", "graph_label": "book_language_book_publisher"},
            {"label": "book", "graph_label": "book_author_book"},
        ],
    }

    assert loader.node_label_map() == {"book": ["book"]}
    assert loader.edge_label_map() == {
        "book": ["book_language_book_publisher", "book_author_book"]
    }


def test_loader_exposes_file_stem_label_aliases():
    loader = DatasetOracleLoader.__new__(DatasetOracleLoader)
    loader.config = {
        "files": [
            {"label": "InfoSource", "path": "Source.csv", "columns": ["infosource_id"]},
        ]
    }
    loader.manifest = {
        "vertices": [{"label": "InfoSource", "graph_label": "InfoSource"}],
        "edges": [],
    }

    assert loader.node_label_map() == {
        "InfoSource": ["InfoSource"],
        "Source": ["InfoSource"],
    }


def test_dataset_loader_manifest_is_tolerant_for_benchmark_data(tmp_path: Path):
    config = {
        "schema": [
            {
                "label": "A",
                "type": "VERTEX",
                "primary": "id",
                "properties": [
                    {"name": "id", "type": "INT64"},
                    {"name": "score", "type": "INT64"},
                ],
            },
            {
                "label": "B",
                "type": "VERTEX",
                "primary": "id",
                "properties": [
                    {"name": "id", "type": "INT64"},
                    {"name": "score", "type": "STRING"},
                ],
            },
            {
                "label": "REL",
                "type": "EDGE",
                "constraints": [["A", "B"]],
                "properties": [],
            },
        ],
        "files": [],
    }
    config_path = tmp_path / "import_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    loader = DatasetOracleLoader.__new__(DatasetOracleLoader)
    loader.import_config_path = config_path
    loader.graph_name = "G"
    manifest = loader._build_manifest()

    assert "FOREIGN KEY" not in manifest["table_ddl"]
    assert "ENFORCED MODE" not in manifest["property_graph_ddl"]
    score_types = [
        column["type"]
        for item in manifest["vertices"]
        for column in item["columns"]
        if column["name"] == "score"
    ]
    assert score_types == ["VARCHAR2(4000)", "VARCHAR2(4000)"]
