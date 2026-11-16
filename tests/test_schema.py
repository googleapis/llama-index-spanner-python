# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, List, Optional, Tuple

import pytest
from llama_index.core.graph_stores.types import EntityNode, Relation

from llama_index_spanner.property_graph_store import SpannerPropertyGraphStore
from llama_index_spanner.schema import GraphDocumentUtility, SpannerGraphSchema
from llama_index_spanner.spanner import SpannerInterface


class _StubSpanner(SpannerInterface):
    """In-memory stub of SpannerInterface for hermetic unit tests."""

    def __init__(self) -> None:
        self.ddls: List[List[str]] = []
        self.inserts: List[Tuple[str, Tuple[str, ...], List[List[Any]]]] = []
        self.deletes: List[Tuple[str, List[List[Any]]]] = []
        self.query_results: List[Dict[str, Any]] = []

    def query(
        self, query: str, params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        if "INFORMATION_SCHEMA.PROPERTY_GRAPHS" in query:
            return []
        return list(self.query_results)

    def apply_ddls(
        self, ddls: List[str], options: Optional[Dict[str, Any]] = None
    ) -> None:
        self.ddls.append(list(ddls))

    def insert_or_update(
        self, table: str, columns: Tuple[str, ...], values: List[List[Any]]
    ) -> None:
        self.inserts.append((table, columns, values))

    def delete(self, table: str, keysets: List[List[Any]]) -> None:
        self.deletes.append((table, keysets))


@pytest.mark.parametrize(
    "identifier",
    [
        "a",
        "Person",
        "my_graph_1",
        "A" + "b" * 127,
    ],
)
def test_is_valid_identifier_accepts_valid(identifier: str) -> None:
    assert GraphDocumentUtility.is_valid_identifier(identifier)
    assert GraphDocumentUtility.to_identifier(identifier) == f"`{identifier}`"


@pytest.mark.parametrize(
    "identifier",
    [
        "",
        "1abc",
        "_abc",
        "a" * 129,
        "valid\n",
        "\u017f",
        "g_Evil` (id STRING(MAX)) PRIMARY KEY(id); DROP TABLE victim; --",
        "prop_bad`key",
        "has space",
        "has-hyphen",
        "semi;colon",
    ],
)
def test_to_identifier_rejects_invalid(identifier: str) -> None:
    assert not GraphDocumentUtility.is_valid_identifier(identifier)
    with pytest.raises(ValueError, match="Invalid Spanner identifier"):
        GraphDocumentUtility.to_identifier(identifier)


def test_graph_schema_rejects_trailing_newline_graph_name() -> None:
    with pytest.raises(ValueError, match="not a valid identifier"):
        SpannerGraphSchema("valid_graph\n", use_flexible_schema=False)


def test_to_identifiers_validates_all_items() -> None:
    assert list(GraphDocumentUtility.to_identifiers(["id", "name"])) == [
        "`id`",
        "`name`",
    ]
    with pytest.raises(ValueError, match="Invalid Spanner identifier"):
        list(GraphDocumentUtility.to_identifiers(["id", "bad`col"]))


def test_upsert_nodes_rejects_malicious_label_before_ddl() -> None:
    stub = _StubSpanner()
    store = SpannerPropertyGraphStore(
        instance_id="i",
        database_id="d",
        graph_name="g",
        impl=stub,
        use_flexible_schema=False,
    )
    malicious_node = EntityNode(
        name="n",
        label="Evil` (id STRING(MAX)) PRIMARY KEY(id); DROP TABLE victim; --",
    )

    with pytest.raises(ValueError, match="Invalid Spanner identifier"):
        store.upsert_nodes([malicious_node])

    assert stub.ddls == []
    assert stub.inserts == []


def test_upsert_nodes_rejects_malicious_property_key() -> None:
    stub = _StubSpanner()
    store = SpannerPropertyGraphStore(
        instance_id="i",
        database_id="d",
        graph_name="g",
        impl=stub,
        use_flexible_schema=False,
    )

    # Initial CREATE TABLE path.
    bad_create_node = EntityNode(
        label="Person",
        name="Alice",
        properties={"bad`col": "value"},
    )
    with pytest.raises(ValueError, match="Invalid Spanner identifier"):
        store.upsert_nodes([bad_create_node])
    assert stub.ddls == []

    # Valid node succeeds.
    good_node = EntityNode(
        label="Person",
        name="Alice",
        properties={"age": 30},
    )
    store.upsert_nodes([good_node])
    assert len(stub.ddls) == 1
    assert len(stub.inserts) == 1

    # ALTER TABLE ADD COLUMN evolution path.
    bad_alter_node = EntityNode(
        label="Person",
        name="Bob",
        properties={
            "age": 25,
            "evil` STRING(MAX); DROP TABLE victim; --": "x",
        },
    )
    with pytest.raises(ValueError, match="Invalid Spanner identifier"):
        store.upsert_nodes([bad_alter_node])
    assert len(stub.ddls) == 1
    assert len(stub.inserts) == 1


def test_upsert_relations_rejects_malicious_label_before_ddl() -> None:
    stub = _StubSpanner()
    store = SpannerPropertyGraphStore(
        instance_id="i",
        database_id="d",
        graph_name="g",
        impl=stub,
        use_flexible_schema=False,
    )

    alice = EntityNode(label="Person", name="Alice")
    bob = EntityNode(label="Person", name="Bob")
    store.upsert_nodes([alice, bob])
    store.schema.graph_exists = True
    ddl_count_after_nodes = len(stub.ddls)

    stub.query_results = [
        {"id": alice.id, "type": "Person", "node_table_label": "Person"},
        {"id": bob.id, "type": "Person", "node_table_label": "Person"},
    ]
    malicious_rel = Relation(
        source_id=alice.id,
        target_id=bob.id,
        label="KNOWS`; DROP TABLE victim; --",
    )

    with pytest.raises(ValueError, match="Invalid Spanner identifier"):
        store.upsert_relations([malicious_rel])

    assert len(stub.ddls) == ddl_count_after_nodes

    malicious_rel_prop = Relation(
        source_id=alice.id,
        target_id=bob.id,
        label="KNOWS",
        properties={"bad`prop": "value"},
    )
    with pytest.raises(ValueError, match="Invalid Spanner identifier"):
        store.upsert_relations([malicious_rel_prop])

    assert len(stub.ddls) == ddl_count_after_nodes


def test_flexible_schema_rejects_malicious_static_property() -> None:
    stub = _StubSpanner()
    store = SpannerPropertyGraphStore(
        instance_id="i",
        database_id="d",
        graph_name="g",
        impl=stub,
        use_flexible_schema=True,
        static_node_properties=["bad`prop"],
    )
    node = EntityNode(
        label="Person",
        name="Alice",
        properties={"bad`prop": "value"},
    )

    with pytest.raises(ValueError, match="Invalid Spanner identifier"):
        store.upsert_nodes([node])

    assert stub.ddls == []


def test_upsert_nodes_valid_succeeds() -> None:
    stub = _StubSpanner()
    store = SpannerPropertyGraphStore(
        instance_id="i",
        database_id="d",
        graph_name="g",
        impl=stub,
        use_flexible_schema=False,
    )
    node = EntityNode(label="Person", name="Alice", properties={"age": 30})
    store.upsert_nodes([node])

    assert len(stub.ddls) == 1
    assert any("CREATE TABLE `g_Person`" in ddl for ddl in stub.ddls[0])
    graph_ddl = "CREATE OR REPLACE PROPERTY GRAPH `g`"
    assert any(graph_ddl in ddl for ddl in stub.ddls[0])
    assert len(stub.inserts) == 1
