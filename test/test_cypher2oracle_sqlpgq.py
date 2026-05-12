from examples.cypher2oracle_sqlpgq import cypher2oracle_sqlpgq


def _translate(cypher: str) -> str:
    query, category = cypher2oracle_sqlpgq(cypher, graph_name="MOVIE_GRAPH")

    assert category == "Graph-IL Translatable"
    assert query.startswith("SELECT")
    assert "FROM GRAPH_TABLE" in query
    assert '"MOVIE_GRAPH"' in query
    return query


def _translate_with_types(
    cypher: str,
    property_type_map: dict[str, dict[str, str]],
) -> str:
    query, category = cypher2oracle_sqlpgq(
        cypher,
        graph_name="MOVIE_GRAPH",
        property_type_map=property_type_map,
    )

    assert category == "Graph-IL Translatable"
    return query


def _translate_sql(cypher: str) -> str:
    query, category = cypher2oracle_sqlpgq(cypher, graph_name="MOVIE_GRAPH")

    assert category == "Graph-IL Translatable"
    assert "FROM GRAPH_TABLE" in query
    assert '"MOVIE_GRAPH"' in query
    return query


def test_cypher2oracle_sqlpgq_translates_simple_node_return_property():
    query = _translate("MATCH (p:PERSON) RETURN p.name AS person_name")

    assert 'MATCH (p IS "PERSON")' in query
    assert 'COLUMNS (p."name" AS person_name)' in query


def test_cypher2oracle_sqlpgq_translates_directed_edge_and_where():
    query = _translate(
        "MATCH (p:PERSON)-[a:ACTED_IN]->(m:MOVIE) "
        "WHERE p.name = 'Tom Hanks' "
        "RETURN m.title AS movie_title"
    )

    assert 'MATCH (p IS "PERSON")-[a IS "ACTED_IN"]->(m IS "MOVIE")' in query
    assert 'WHERE p."name" = \'Tom Hanks\'' in query
    assert 'COLUMNS (m."title" AS movie_title)' in query


def test_cypher2oracle_sqlpgq_translates_order_skip_limit():
    query = _translate(
        "MATCH (p:PERSON) "
        "RETURN p.name AS person_name "
        "ORDER BY p.name ASC SKIP 5 LIMIT 10"
    )

    assert "ORDER BY person_name ASC" in query
    assert "OFFSET 5 ROWS" in query
    assert "FETCH FIRST 10 ROWS ONLY" in query


def test_cypher2oracle_sqlpgq_translates_variable_length_relationship():
    query = _translate(
        "MATCH (person:PERSON)-[:KNOWS*..3]->(friend:PERSON) "
        "RETURN friend.name AS friend_name"
    )

    assert '[e1 IS "KNOWS"]->{1,3}' in query
    assert 'COLUMNS (friend."name" AS friend_name)' in query


def test_cypher2oracle_sqlpgq_translates_whole_node_return_to_vertex_id():
    query = _translate("MATCH (p:PERSON) RETURN p")

    assert "COLUMNS (VERTEX_ID(p) AS p_VALUE)" in query


def test_cypher2oracle_sqlpgq_expands_named_path_return_to_element_ids():
    query = _translate(
        "MATCH p = (n1:ACCOUNT)-[e1]-(x)-[e2]-(n2:ACCOUNT) "
        "WHERE n1.account_id = 'A000000' AND n2.account_id <> 'A000000' "
        "RETURN p LIMIT 1"
    )

    assert 'MATCH (n1 IS "ACCOUNT")-[e1]-(x)-[e2]-(n2 IS "ACCOUNT")' in query
    assert "VERTEX_ID(n1) AS p_n1_ID" in query
    assert "EDGE_ID(e1) AS p_e1_ID" in query
    assert "VERTEX_ID(x) AS p_x_ID" in query
    assert "EDGE_ID(e2) AS p_e2_ID" in query
    assert "VERTEX_ID(n2) AS p_n2_ID" in query
    assert "FETCH FIRST 1 ROWS ONLY" in query


def test_cypher2oracle_sqlpgq_rejects_variable_length_named_path_return():
    translated_query, category = cypher2oracle_sqlpgq(
        "MATCH p = (a:ACCOUNT)-[e*1..3]->(b:ACCOUNT) RETURN p"
    )

    assert translated_query == "Unable to Translate to Oracle SQL/PGQ"
    assert category == "Graph-IL Not Support"


def test_cypher2oracle_sqlpgq_translates_union_all_path_branches():
    query = _translate(
        'MATCH p = (n1:ACCOUNT)-[e1:BelongsTo]-(x:FINANCIAL_PERIOD) '
        'WHERE n1.account_id = "A000000" RETURN p LIMIT 5 '
        'UNION ALL '
        'MATCH p = (n1:ACCOUNT)-[e1:BelongsTo]-(x:FINANCIAL_PERIOD)-[e2]-(y) '
        'WHERE n1.account_id = "A000000" RETURN p LIMIT 5'
    )

    assert "\nUNION ALL\n" in query
    assert query.count("FROM GRAPH_TABLE") == 2
    assert "NULL AS p_e2_ID" in query
    assert "NULL AS p_y_ID" in query
    assert query.count("FETCH FIRST 5 ROWS ONLY") == 2


def test_cypher2oracle_sqlpgq_rejects_invalid_cypher():
    translated_query, category = cypher2oracle_sqlpgq("MATCH (p:PERSON RETURN p")

    assert translated_query == "Unable to Translate to Oracle SQL/PGQ"
    assert category != "Graph-IL Translatable"


def test_cypher2oracle_sqlpgq_translates_multi_condition_where():
    query = _translate(
        "MATCH (m:Movie) "
        "WHERE m.released >= 1990 AND m.released <= 2000 AND m.votes > 5000 "
        "RETURN m.title, m.released, m.votes"
    )

    assert 'm."released" >= 1990 AND m."released" <= 2000' in query
    assert 'm."votes" > 5000' in query


def test_cypher2oracle_sqlpgq_translates_null_predicate():
    query = _translate(
        "MATCH (c:Character) WHERE c.song IS NOT NULL RETURN c.name AS character_name"
    )

    assert 'WHERE c."song" IS NOT NULL' in query


def test_cypher2oracle_sqlpgq_translates_count_star():
    query = _translate("MATCH (m:Movie) RETURN count(*)")

    assert query.startswith("SELECT COUNT(*) AS COUNT_VALUE")
    assert "COLUMNS (1 AS dummy_value)" in query


def test_cypher2oracle_sqlpgq_translates_backtick_identifiers_and_column_compare():
    query = _translate(
        "MATCH (a:`voice-actors`)-[:MOVIE]->(c:characters) "
        "WHERE a.movie = c.movie_title "
        "RETURN a.`voice-actor` AS actor"
    )

    assert 'a."movie" = c."movie_title"' in query
    assert 'a."voice_actor" AS actor' in query


def test_cypher2oracle_sqlpgq_translates_double_quoted_property_map_strings():
    query = _translate(
        'MATCH (u1:User {label: "inchristbl.bsky.social"}) '
        "RETURN u1.label"
    )

    assert 'u1."label" = \'inchristbl.bsky.social\'' in query
    assert '"inchristbl.bsky.social"' not in query


def test_cypher2oracle_sqlpgq_uses_out_of_line_where_for_property_maps():
    query = _translate(
        "MATCH (target:User {label: 'dwither.bsky.social'})"
        "<-[:INTERACTED]-(user:User) "
        "RETURN user.x, user.y LIMIT 3"
    )

    assert 'MATCH (user_VALUE IS "User")-[e1 IS "INTERACTED"]->(target IS "User")' in query
    assert 'COLUMNS (user_VALUE."x" AS x, user_VALUE."y" AS y)' in query
    assert 'WHERE target."label" = \'dwither.bsky.social\'' in query


def test_cypher2oracle_sqlpgq_projects_hidden_order_by_property():
    query = _translate(
        "MATCH (u:User) WHERE u.size < 2.0 "
        "RETURN u ORDER BY u.size DESC LIMIT 5"
    )

    assert query.startswith("SELECT u_VALUE")
    assert 'COLUMNS (VERTEX_ID(u) AS u_VALUE, u."size" AS size_VALUE)' in query
    assert "ORDER BY size_VALUE DESC" in query


def test_cypher2oracle_sqlpgq_projects_hidden_order_by_scalar_function():
    query = _translate(
        "MATCH (u:User) WHERE u.x IS NOT NULL "
        "RETURN u ORDER BY abs(u.x) LIMIT 3"
    )

    assert query.startswith("SELECT u_VALUE")
    assert 'abs(u."x") AS abs_x' in query
    assert "ORDER BY abs_x" in query


def test_cypher2oracle_sqlpgq_translates_cypher_date_function_to_oracle_literal():
    query = _translate(
        "MATCH (a:ACCOUNT) "
        "WHERE a.opening_date < date('2020-01-01') "
        "RETURN a.status"
    )

    assert 'a."opening_date" < DATE \'2020-01-01\'' in query


def test_cypher2oracle_sqlpgq_translates_cypher_current_date_function():
    query = _translate(
        "MATCH (cr:COMPLIANCE_RULE) "
        "WHERE cr.expiry_date >= date() "
        "RETURN cr.rule_id"
    )

    assert 'cr."expiry_date" >= TRUNC(CURRENT_DATE)' in query
    assert "date()" not in query


def test_cypher2oracle_sqlpgq_coerces_string_backed_date_literals():
    query = _translate_with_types(
        "MATCH (m:Movie) WHERE m.release_date >= date('1990-01-01') RETURN m.title",
        {"Movie": {"release_date": "VARCHAR2(4000)"}},
    )

    assert 'm."release_date" >= \'1990-01-01\'' in query


def test_cypher2oracle_sqlpgq_coerces_string_property_numeric_literals():
    query = _translate_with_types(
        "MATCH (u:User {id: 1}) RETURN u",
        {"User": {"id": "VARCHAR2(4000)"}},
    )

    assert 'u."id" = \'1\'' in query


def test_cypher2oracle_sqlpgq_coerces_string_property_boolean_literals():
    query = _translate_with_types(
        "MATCH (s:Supplier)-[:SUPPLIES]->(p:Product) "
        "WHERE p.discontinued = true RETURN s",
        {"Product": {"discontinued": "VARCHAR2(4000)"}},
    )

    assert 'p."discontinued" = \'true\'' in query
    assert 'p."discontinued" = 1' not in query


def test_cypher2oracle_sqlpgq_coerces_string_property_map_boolean_literals():
    query = _translate_with_types(
        "MATCH (p:Product {discontinued: false}) RETURN p",
        {"Product": {"discontinued": "VARCHAR2(4000)"}},
    )

    assert 'p."discontinued" = \'false\'' in query
    assert 'p."discontinued" = \'0\'' not in query


def test_cypher2oracle_sqlpgq_translates_date_property_extractors():
    query = _translate_with_types(
        "MATCH (m:Movie) WHERE date(m.release_date).year = 1995 RETURN m.title",
        {"Movie": {"release_date": "VARCHAR2(4000)"}},
    )

    assert 'EXTRACT(YEAR FROM TO_DATE(m."release_date", \'YYYY-MM-DD\')) = 1995' in query


def test_cypher2oracle_sqlpgq_translates_property_date_extractors_and_modulo():
    query = _translate_with_types(
        "MATCH (m:Movie) WHERE m.release_date.year % 4 = 0 RETURN m.title",
        {"Movie": {"release_date": "VARCHAR2(4000)"}},
    )

    assert 'MOD(EXTRACT(YEAR FROM TO_DATE(m."release_date", \'YYYY-MM-DD\')), 4) = 0' in query


def test_cypher2oracle_sqlpgq_disambiguates_duplicate_projection_aliases():
    query = _translate(
        "MATCH (dc:DataConsumer)-[:Consumes]->(da:DataAsset) "
        "RETURN dc.name, da.name"
    )

    assert 'dc."name" AS dc' in query
    assert 'da."name" AS da' in query
    assert " AS name" not in query


def test_cypher2oracle_sqlpgq_orders_by_aggregate_alias_without_inner_projection():
    query = _translate(
        "MATCH (u:USER)-[:Initiates]->(t:TRANSACTION) "
        "RETURN u.user_id, SUM(t.amount) AS total_amount "
        "ORDER BY total_amount DESC"
    )

    assert "SELECT user_id, SUM(amount) AS total_amount" in query
    assert "total_amount AS total_amount" not in query
    assert "ORDER BY total_amount DESC" in query


def test_cypher2oracle_sqlpgq_translates_scalar_function_return_expressions():
    query = _translate(
        "MATCH (m:Movie) "
        "RETURN abs(m.revenue - m.budget) AS difference "
        "ORDER BY difference DESC LIMIT 3"
    )

    assert 'abs(m."revenue" - m."budget") AS difference' in query


def test_cypher2oracle_sqlpgq_translates_tofloat_to_oracle_number_cast():
    query = _translate(
        "MATCH (m:Movie) RETURN avg(toFloat(m.budget)) AS average_budget"
    )

    assert 'COLUMNS (TO_NUMBER(m."budget") AS budget)' in query
    assert "SELECT AVG(budget) AS average_budget" in query


def test_cypher2oracle_sqlpgq_translates_tointeger_to_oracle_cast():
    query = _translate(
        "MATCH (se:SystemEnvironment)<-[:RunsIn]-(pj:ProcessingJob)"
        "-[:Transforms]->(da:DataAsset) "
        "WHERE da.sensitivity_level = 'PII' "
        "RETURN se.name AS environment_name, COUNT(pj) AS pii_processing_job_count, "
        "AVG(toInteger(pj.sla_requirements)) AS avg_sla_hours"
    )

    assert 'CAST(pj."sla_requirements" AS INTEGER) AS sla_requirements' in query
    assert "SELECT environment_name, COUNT(pj_VALUE) AS pii_processing_job_count, AVG(sla_requirements) AS avg_sla_hours" in query
    assert "toInteger" not in query


def test_cypher2oracle_sqlpgq_translates_size_split_word_count():
    query = _translate(
        'MATCH (m:Movie) RETURN size(split(m.overview, " ")) AS word_count'
    )

    assert "REGEXP_COUNT(m.\"overview\", '\\\\S+') AS word_count" in query


def test_cypher2oracle_sqlpgq_translates_vertex_comparisons():
    query = _translate(
        "MATCH (m1:Movie) MATCH (m2:Movie) WHERE m1 <> m2 RETURN m2"
    )

    assert "NOT VERTEX_EQUAL(m1, m2)" in query


def test_cypher2oracle_sqlpgq_translates_final_with_projection_aggregate():
    query = _translate(
        "MATCH (u1:User)-[:INTERACTED]->(u2:User) "
        "WHERE u2.size <> 1.5 "
        "WITH u1.size AS interactingUserSize "
        "RETURN sum(interactingUserSize) AS totalSize"
    )

    assert query.startswith("SELECT SUM(interactingUserSize) AS totalSize")
    assert 'u1."size" AS interactingUserSize' in query
    assert "WHERE u2.\"size\" <> 1.5" in query


def test_cypher2oracle_sqlpgq_uses_oracle_edge_label_map():
    query, category = cypher2oracle_sqlpgq(
        "MATCH (a)-[:book]->(b) RETURN b",
        graph_name="G",
        edge_label_map={"book": ["book_language_book_publisher", "book_author_book"]},
    )

    assert category == "Graph-IL Translatable"
    assert 'IS "book_language_book_publisher" | "book_author_book"' in query


def test_cypher2oracle_sqlpgq_translates_relationship_label_alternatives():
    query = _translate(
        "MATCH (fp:FINANCIAL_PERIOD {status: 'Open'})"
        "<-[:BelongsTo]-(t:TRANSACTION)<-[:Initiates|Approves]-(u:USER) "
        "RETURN fp.period_id, COUNT(DISTINCT u) AS unique_users, "
        "SUM(t.amount) AS total_transaction_amount"
    )

    assert 'IS "Initiates" | "Approves"' in query
    assert 'fp."period_id" AS period_id' in query
    assert "COUNT(DISTINCT u_VALUE) AS unique_users" in query
    assert "SUM(amount) AS total_transaction_amount" in query


def test_cypher2oracle_sqlpgq_aggregates_outside_graph_table():
    query = _translate(
        "MATCH (book:book) "
        "WHERE (book.publisher_id = 1929 AND book.num_pages > 500) "
        "RETURN count(*)"
    )

    assert query.startswith("SELECT COUNT(*) AS COUNT_VALUE")
    assert 'MATCH (book IS "book")' in query
    assert 'WHERE (book."publisher_id" = 1929 AND book."num_pages" > 500)' in query
    assert "COLUMNS (1 AS dummy_value)" in query
    assert "COUNT(*) AS COUNT_VALUE)" not in query


def test_cypher2oracle_sqlpgq_average_aggregates_projected_property_outside_graph_table():
    query = _translate(
        "MATCH (:Person)-[r:REVIEWED]->(m:Movie) "
        "WHERE r.rating > 80 "
        "RETURN AVG(m.released) AS averageReleaseYear"
    )

    assert query.startswith("SELECT AVG(released) AS averageReleaseYear")
    assert 'COLUMNS (m."released" AS released)' in query


def test_cypher2oracle_sqlpgq_grouped_aggregate_uses_outer_group_by():
    query = _translate(
        "MATCH (director:director) "
        "RETURN director, count(director.name) ORDER BY count(director.name) DESC LIMIT 1"
    )

    assert query.startswith("SELECT director_VALUE, COUNT(name) AS name")
    assert 'COLUMNS (VERTEX_ID(director) AS director_VALUE, director."name" AS name)' in query
    assert "GROUP BY director_VALUE" in query
    assert "ORDER BY name DESC" in query


def test_cypher2oracle_sqlpgq_translates_aggregate_case_expression():
    query = _translate(
        "MATCH (t1:area_code)<-[zip_code:ZIP_CODE]-(t2:zip_data) "
        "WHERE t1.area_code = 787 "
        "RETURN (count(CASE WHEN t2.type = 'P.O. Box Only' THEN 1 ELSE NULL END) "
        "- count(CASE WHEN t2.type = 'Post Office' THEN 1 ELSE NULL END)) AS DIFFERENCE"
    )

    assert query.startswith(
        "SELECT (count(CASE WHEN type = 'P.O. Box Only' THEN 1 ELSE NULL END) "
        "- count(CASE WHEN type = 'Post Office' THEN 1 ELSE NULL END)) AS DIFFERENCE"
    )
    assert 'COLUMNS (t2."type" AS type)' in query
    assert "GROUP BY" not in query


def test_cypher2oracle_sqlpgq_uses_case_insensitive_label_maps():
    query, category = cypher2oracle_sqlpgq(
        "MATCH (t1:state)<-[state:STATE]-(t2:country) RETURN count(t2.county)",
        graph_name="G",
        node_label_map={"state": ["state"], "country": ["country"]},
        edge_label_map={"state": ["country_STATE_state"]},
    )

    assert category == "Graph-IL Translatable"
    assert '[state IS "country_STATE_state"]->' in query
    assert '[state IS "STATE"]' not in query


def test_cypher2oracle_sqlpgq_maps_sanitized_labels_and_properties_strictly():
    query, category = cypher2oracle_sqlpgq(
        "MATCH (t1:characters)-[hero:HERO]->(t2:`voice-actors`) "
        "WHERE t2.movie = t1.movie_title "
        "RETURN t2.`voice-actor`",
        graph_name="G",
        node_label_map={
            "characters": ["characters"],
            "voice_actors": ["voice_actors"],
        },
        edge_label_map={"HERO": ["characters_HERO_voice_actors"]},
        property_type_map={
            "characters": {"movie_title": "VARCHAR2(4000)"},
            "voice_actors": {
                "movie": "VARCHAR2(4000)",
                "voice_actor": "VARCHAR2(4000)",
            },
            "characters_HERO_voice_actors": {},
        },
        strict_property_validation=True,
    )

    assert category == "Graph-IL Translatable"
    assert 't2 IS "voice_actors"' in query
    assert '[hero IS "characters_HERO_voice_actors"]->' in query
    assert 't2."voice_actor" AS voice_actor' in query


def test_cypher2oracle_sqlpgq_translates_string_predicates():
    starts = _translate("MATCH (a:author) WHERE a.name STARTS WITH 'George' RETURN a.name")
    ends = _translate("MATCH (c:customer) WHERE c.email ENDS WITH '@x.test' RETURN c.email")
    contains = _translate("MATCH (p:publisher) WHERE p.name CONTAINS 'book' RETURN count(*)")

    assert 'a."name" LIKE \'George\' || \'%\'' in starts
    assert 'c."email" LIKE \'%\' || \'@x.test\'' in ends
    assert 'INSTR(p."name", \'book\') > 0' in contains


def test_cypher2oracle_sqlpgq_translates_label_predicates():
    query, category = cypher2oracle_sqlpgq(
        "MATCH (entity)-[:ClassifiedUnder]->(d:Domain) "
        "WHERE entity:Concept OR entity:Assertion "
        "RETURN d.name, entity:Concept AS is_concept",
        graph_name="G",
        node_label_map={"Concept": ["Concept"], "Assertion": ["Assertion"]},
    )

    assert category == "Graph-IL Translatable"
    assert 'entity IS LABELED "Concept"' in query
    assert 'entity IS LABELED "Assertion"' in query
    assert 'entity:Concept' not in query


def test_cypher2oracle_sqlpgq_maps_identity_to_primary_key():
    query, category = cypher2oracle_sqlpgq(
        "MATCH (n:PaymentTransaction) RETURN count(n.identity)",
        graph_name="G",
        property_type_map={
            "PaymentTransaction": {
                "transaction_id": "VARCHAR2(4000)",
                "status": "VARCHAR2(4000)",
            }
        },
        node_primary_key_map={"PaymentTransaction": "transaction_id"},
        strict_property_validation=True,
    )

    assert category == "Graph-IL Translatable"
    assert 'n."transaction_id" AS identity' in query
    assert 'n."identity"' not in query


def test_cypher2oracle_sqlpgq_rejects_missing_properties_when_strict():
    query, category = cypher2oracle_sqlpgq(
        "MATCH (n:PaymentTransaction) RETURN n.missing_property",
        graph_name="G",
        property_type_map={"PaymentTransaction": {"transaction_id": "VARCHAR2(4000)"}},
        strict_property_validation=True,
    )

    assert query == "Unable to Translate to Oracle SQL/PGQ"
    assert category == "Graph-IL Not Support"


def test_cypher2oracle_sqlpgq_translates_aggregate_arithmetic_over_elements():
    query = _translate(
        "MATCH (g:Group)<-[:BelongsTo]-(u:User)-[:HasRole]->(r:Role) "
        "RETURN g.Group_id, g.name, COUNT(DISTINCT u) AS user_count, "
        "COUNT(r) / COUNT(DISTINCT u) AS avg_roles_per_user"
    )

    assert "COUNT(r_VALUE) / COUNT(DISTINCT u_VALUE) AS avg_roles_per_user" in query
    assert "VERTEX_ID(r) AS r_VALUE" in query
    assert "VERTEX_ID(u) AS u_VALUE" in query
    assert "r) / COUNT(u AS" not in query


def test_cypher2oracle_sqlpgq_translates_aggregate_arithmetic_over_properties():
    query = _translate(
        "MATCH (fp:FINANCIAL_PERIOD)<-[:BelongsTo]-(t:TRANSACTION), "
        "(b:BUDGET)-[:AllocatedTo]->(a:ACCOUNT)-[:BelongsTo]->(fp) "
        "RETURN fp.period_id, SUM(b.amount) - SUM(t.amount) AS budget_variance"
    )

    assert "SUM(b_amount) - SUM(t_amount) AS budget_variance" in query
    assert 'b."amount" AS b_amount' in query
    assert 't."amount" AS t_amount' in query


def test_cypher2oracle_sqlpgq_translates_aggregate_with_to_cte():
    query = _translate_sql(
        "MATCH (u:USER)-[:POSTS]->(t:TWEET) "
        "WITH u.username AS username, COUNT(t) AS tweet_count "
        "WHERE tweet_count > 10 "
        "RETURN username, tweet_count ORDER BY tweet_count DESC LIMIT 5"
    )

    assert query.startswith("WITH stage_1 AS")
    assert "SELECT username, COUNT(t_VALUE) AS tweet_count" in query
    assert "GROUP BY username" in query
    assert "WHERE tweet_count > 10" in query
    assert "ORDER BY tweet_count DESC" in query


def test_cypher2oracle_sqlpgq_carries_with_variable_properties_to_cte():
    query = _translate_sql(
        "MATCH (u:USER)-[:Initiates]->(t:TRANSACTION) "
        "WITH u, COUNT(t) AS transaction_count, SUM(t.amount) AS total_amount "
        "WHERE transaction_count > 5 "
        "RETURN u.user_id, total_amount"
    )

    assert 'u."user_id" AS u_user_id' in query
    assert "GROUP BY u_VALUE, u_user_id" in query
    assert "SELECT u_user_id AS user_id, total_amount AS total_amount" in query
    assert "SELECT u AS user_id" not in query


def test_cypher2oracle_sqlpgq_carries_with_variable_properties_for_final_aggregate():
    query = _translate_sql(
        "MATCH (a:ACCOUNT)-[:GovernedBy]->(cr:COMPLIANCE_RULE) "
        "WITH a, COUNT(cr) AS rule_count "
        "WHERE rule_count > 1 "
        "RETURN AVG(a.balance) AS average_balance"
    )

    assert 'a."balance" AS a_balance' in query
    assert "GROUP BY a_VALUE, a_balance" in query
    assert "SELECT AVG(a_balance) AS average_balance" in query
    assert "AVG(a)" not in query


def test_cypher2oracle_sqlpgq_translates_ordered_with_to_cte():
    query = _translate_sql(
        "MATCH (c:Character) "
        "WITH c.name AS name, c.rank AS rank ORDER BY rank DESC LIMIT 5 "
        "RETURN name, rank"
    )

    assert query.startswith("WITH stage_1 AS")
    assert 'c."name" AS name' in query
    assert 'c."rank" AS rank' in query
    assert "ORDER BY rank DESC" in query
    assert "FETCH FIRST 5 ROWS ONLY" in query
    assert query.strip().endswith("FROM stage_1")


def test_cypher2oracle_sqlpgq_rejects_unlowered_pattern_exists():
    query, category = cypher2oracle_sqlpgq(
        "MATCH (a:ACTOR) "
        "WHERE NOT EXISTS((a)-[:ACTED_IN]->(:MOVIE)) "
        "RETURN a.name",
        graph_name="G",
    )

    assert query == "Unable to Translate to Oracle SQL/PGQ"
    assert category == "Graph-IL Not Support"
