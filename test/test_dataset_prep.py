import json
from pathlib import Path

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
    assert "optional_match" in detect_unsupported_features(
        "MATCH (a) OPTIONAL MATCH (a)--(b) RETURN b"
    )


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
