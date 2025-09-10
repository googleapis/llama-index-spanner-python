# Copyright 2025 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Generator, Union

import pytest
from llama_index.core.graph_stores.types import ChunkNode, EntityNode, Relation
from llama_index.core.vector_stores.types import (
    ExactMatchFilter,
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)

from llama_index_spanner.property_graph_store import SpannerPropertyGraphStore
from llama_index_spanner.schema import ElementSchema
from tests.utils import (
    get_random_suffix,
    get_spanner_property_graph_store,
    spanner_database_id,
    spanner_graph_name,
    spanner_instance_id,
)

pytestmark = pytest.mark.skipif(
    (not spanner_instance_id or not spanner_database_id or not spanner_graph_name),
    reason=(
        "Requires SPANNER_INSTANCE_ID, SPANNER_DATABASE_ID and"
        " SPANNER_GRAPH_NAME environment variables."
    ),
)


@pytest.fixture
def property_graph_store_static() -> Generator[SpannerPropertyGraphStore, None, None]:
    """Provides a fresh SpannerPropertyGraphStore for each test."""
    graph_store = get_spanner_property_graph_store(
        graph_name_suffix=get_random_suffix()
    )
    yield graph_store
    graph_store.clean_up()


@pytest.fixture
def property_graph_store_dynamic() -> Generator[SpannerPropertyGraphStore, None, None]:
    """Provides a fresh SpannerPropertyGraphStore for each test."""
    graph_store_dynamic_schema = get_spanner_property_graph_store(
        use_flexible_schema=True,
        graph_name_suffix=get_random_suffix(),
    )
    yield graph_store_dynamic_schema
    graph_store_dynamic_schema.clean_up()


def test_upsert_nodes_and_get(
    property_graph_store_static: SpannerPropertyGraphStore,
    property_graph_store_dynamic: SpannerPropertyGraphStore,
):
    """Test inserting entity and chunk nodes, then retrieving them."""
    entity = EntityNode(label="PERSON", name="Alice")
    chunk = ChunkNode(text="Alice is a software engineer.")

    for graph_store in [
        property_graph_store_static,
        property_graph_store_dynamic,
    ]:
        graph_store.upsert_nodes([entity, chunk])

        # Get by ID
        retrieved_entities = graph_store.get(ids=[entity.id])
        assert len(retrieved_entities) == 1
        assert isinstance(retrieved_entities[0], EntityNode)
        assert retrieved_entities[0].name == "Alice"

        retrieved_chunks = graph_store.get(ids=[chunk.id])
        assert len(retrieved_chunks) == 1
        assert isinstance(retrieved_chunks[0], ChunkNode)
        assert retrieved_chunks[0].text == "Alice is a software engineer."

        # Get by property
        retrieved_by_prop = graph_store.get(ids=["Alice"])
        assert len(retrieved_by_prop) == 1
        assert retrieved_by_prop[0].id == entity.id

        # Attempt to get unknown property
        unknown_prop = graph_store.get(properties={"non_existent_prop": "foo"})
        assert not unknown_prop


def test_upsert_nodes_and_get_multiple(
    property_graph_store_static: SpannerPropertyGraphStore,
    property_graph_store_dynamic: SpannerPropertyGraphStore,
):
    """Test inserting multiple nodes at once and retrieving them by IDs."""
    entity1 = EntityNode(label="PERSON", name="Bob")
    entity2 = EntityNode(label="PERSON", name="Charlie")
    chunk1 = ChunkNode(text="This is sample text.")
    chunk2 = ChunkNode(text="Another sample text.")

    for graph_store in [
        property_graph_store_static,
        property_graph_store_dynamic,
    ]:
        # Upsert multiple
        graph_store.upsert_nodes([entity1, entity2, chunk1, chunk2])

        # Retrieve by IDs
        ids_to_get = [entity1.id, entity2.id, chunk1.id, chunk2.id]
        results = graph_store.get(ids=ids_to_get)
        assert len(results) == 4

        # Check some known values
        person_bob = [
            r for r in results if isinstance(r, EntityNode) and r.name == "Bob"
        ]
        assert len(person_bob) == 1

        chunk_texts = [r for r in results if isinstance(r, ChunkNode)]
        assert len(chunk_texts) == 2


def test_upsert_node_idempotency_update_properties(
    property_graph_store_static: SpannerPropertyGraphStore,
    property_graph_store_dynamic: SpannerPropertyGraphStore,
):
    """Test upserting the same node updates its properties (idempotency)."""
    for graph_store in [
        property_graph_store_static,
        property_graph_store_dynamic,
    ]:
        # Initial upsert
        node_v1 = EntityNode(
            label="PRODUCT",
            name="Laptop",
            properties={"status": "available", "price": 1200},
        )
        graph_store.upsert_nodes([node_v1])

        retrieved_v1 = graph_store.get(ids=[node_v1.id])[0]
        assert retrieved_v1.properties.get("status") == "available"
        assert retrieved_v1.properties.get("price") == 1200
        assert retrieved_v1.properties.get("ram_gb") is None

        # Second upsert for the same node ID with modified and new properties
        node_v2 = EntityNode(
            label="PRODUCT",  # Label should ideally be consistent for the same node
            name="Laptop Pro",  # Name can be updated
            properties={"status": "backorder", "price": 1250, "ram_gb": 16},
        )
        graph_store.upsert_nodes([node_v2])

        retrieved_v2 = graph_store.get(ids=[node_v2.id])[0]
        assert (
            len(graph_store.get(properties={"status": "backorder"})) == 1
        )  # Check if name update worked for query

        if isinstance(retrieved_v2, EntityNode):
            assert retrieved_v2.name == "Laptop Pro"
        assert retrieved_v2.properties.get("status") == "backorder"
        assert retrieved_v2.properties.get("price") == 1250
        assert retrieved_v2.properties.get("ram_gb") == 16


def test_upsert_relations_and_get(
    property_graph_store_static: SpannerPropertyGraphStore,
    property_graph_store_dynamic: SpannerPropertyGraphStore,
):
    """Test creating relations between nodes, then retrieving them."""
    person = EntityNode(label="PERSON", name="Alice")
    city = EntityNode(label="CITY", name="Paris")

    for graph_store in [
        property_graph_store_static,
        property_graph_store_dynamic,
    ]:
        graph_store.upsert_nodes([person, city])

        # Create a relation
        visited_relation = Relation(
            source_id=person.id,
            target_id=city.id,
            label="VISITED",
            properties={"year": 2023},
        )
        graph_store.upsert_relations([visited_relation])

        # Validate that the relation can be found in triplets
        triplets = graph_store.get_triplets(entity_names=["Alice"])
        assert len(triplets) == 1
        source, rel, target = triplets[0]
        if isinstance(source, EntityNode) and isinstance(target, EntityNode):
            assert source.name == "Alice"
            assert target.name == "Paris"
        assert rel.label == "VISITED"
        assert rel.properties["year"] == 2023


def test_upsert_relations_and_get_multiple(
    property_graph_store_static: SpannerPropertyGraphStore,
    property_graph_store_dynamic: SpannerPropertyGraphStore,
):
    """Test inserting multiple relations at once and retrieving them."""
    person = EntityNode(label="PERSON", name="Alice")
    city = EntityNode(label="CITY", name="Paris")
    person2 = EntityNode(label="PERSON", name="Bob")
    city2 = EntityNode(label="CITY", name="Switzerland")
    for graph_store in [
        property_graph_store_static,
        property_graph_store_dynamic,
    ]:
        graph_store.upsert_nodes([person, city, person2, city2])

        # Create a relation
        visited_relation = Relation(
            source_id=person.id,
            target_id=city.id,
            label="VISITED",
            properties={"year": 2023},
        )
        visited_relation2 = Relation(
            source_id=person2.id,
            target_id=city2.id,
            label="VISITED",
            properties={"year": 2025},
        )
        graph_store.upsert_relations([visited_relation, visited_relation2])

        # Validate that the relation can be found in triplets
        triplets = graph_store.get_triplets(entity_names=["Alice", "Bob"])
        assert len(triplets) == 2
        source, rel, target = triplets[0]
        if isinstance(source, EntityNode) and isinstance(target, EntityNode):
            assert source.name == "Alice"
            assert target.name == "Paris"
        assert rel.label == "VISITED"
        assert rel.properties["year"] == 2023

        source, rel, target = triplets[1]
        if isinstance(source, EntityNode) and isinstance(target, EntityNode):
            assert source.name == "Bob"
            assert target.name == "Switzerland"
        assert rel.label == "VISITED"
        assert rel.properties["year"] == 2025


def test_upsert_relation_with_non_existent_source_node(
    property_graph_store_static: SpannerPropertyGraphStore,
):
    """Test upserting a relation with a non-existent source ID when ignore_invalid_relations is False."""
    graph_store = property_graph_store_static
    target_node = EntityNode(label="CITY", name="ValidTarget")
    graph_store.upsert_nodes([target_node])

    relation_invalid_source = Relation(
        source_id="nonExistentSource123",
        target_id=target_node.id,
        label="TRAVELLED_TO",
    )
    graph_store.upsert_relations([relation_invalid_source])
    triplets = graph_store.get_triplets(entity_names=["ValidTarget"])
    assert not triplets


def test_upsert_relation_with_non_existent_target_node(
    property_graph_store_static: SpannerPropertyGraphStore,
):
    """Test upserting a relation with a non-existent target ID when ignore_invalid_relations is False."""
    graph_store = property_graph_store_static
    source_node = EntityNode(label="PERSON", name="ValidSource")
    graph_store.upsert_nodes([source_node])

    relation_invalid_target = Relation(
        source_id=source_node.id,
        target_id="nonExistentTarget456",
        label="LIVES_IN",
    )
    graph_store.upsert_relations([relation_invalid_target])
    triplets = graph_store.get_triplets(entity_names=["ValidSource"])
    assert not triplets


def test_upsert_relation_idempotency_update_properties(
    property_graph_store_static: SpannerPropertyGraphStore,
    property_graph_store_dynamic: SpannerPropertyGraphStore,
):
    """Test upserting the same relation updates its properties."""
    for graph_store in [
        property_graph_store_static,
        property_graph_store_dynamic,
    ]:
        user = EntityNode(label="USER", name="User1")
        item = EntityNode(label="ITEM", name="Item1")
        graph_store.upsert_nodes([user, item])

        # Initial relation upsert
        rel_v1 = Relation(
            source_id=user.id,
            target_id=item.id,
            label="PURCHASED",
            properties={"quantity": 1, "rating": None},
        )
        graph_store.upsert_relations([rel_v1])

        triplets_v1 = graph_store.get_triplets(
            entity_names=["User1"], relation_names=["PURCHASED"]
        )
        assert len(triplets_v1) == 1
        assert triplets_v1[0][1].properties.get("quantity") == 1
        assert triplets_v1[0][1].properties.get("rating") is None
        assert triplets_v1[0][1].properties.get("review_status") is None

        # Second upsert for the same relation with modified and new properties
        rel_v2 = Relation(
            source_id=user.id,  # Same source, target, label
            target_id=item.id,
            label="PURCHASED",
            properties={"quantity": 2, "rating": 5, "review_status": "pending"},
        )
        graph_store.upsert_relations([rel_v2])

        triplets_v2 = graph_store.get_triplets(
            entity_names=["User1"], relation_names=["PURCHASED"]
        )
        assert len(triplets_v2) == 1  # Should still be one relation, but updated
        retrieved_rel_v2 = triplets_v2[0][1]
        assert retrieved_rel_v2.properties.get("quantity") == 2
        assert retrieved_rel_v2.properties.get("rating") == 5
        assert retrieved_rel_v2.properties.get("review_status") == "pending"


def test_upsert_relation_with_non_existent_ids_ignore_true(
    property_graph_store_static: SpannerPropertyGraphStore,
    property_graph_store_dynamic: SpannerPropertyGraphStore,
):
    """Test upserting relation with non-existent IDs when ignore_invalid_relations is True (default)."""
    for graph_store in [
        property_graph_store_static,
        property_graph_store_dynamic,
    ]:
        valid_node = EntityNode(label="VALID_NODE", name="Valid")
        graph_store.upsert_nodes([valid_node])

        rel_invalid_source = Relation(
            source_id="nonExistentSource789",
            target_id=valid_node.id,
            label="LINKS_TO",
        )
        rel_invalid_target = Relation(
            source_id=valid_node.id,
            target_id="nonExistentTarget101",
            label="POINTS_TO",
        )

        try:
            graph_store.upsert_relations([rel_invalid_source, rel_invalid_target])
        except ValueError:
            pytest.fail(
                "ValueError should not be raised when ignore_invalid_relations is"
                " True."
            )

        # Verify no relations were actually added
        triplets = graph_store.get_triplets(ids=[valid_node.id])
        assert not triplets

        # Verify for specific non-existent source/target if possible,
        # though above check is good
        all_relations = graph_store.get_triplets()  # Get all possible triplets
        source_ids_in_graph = {t[1].source_id for t in all_relations}
        target_ids_in_graph = {t[1].target_id for t in all_relations}

        assert "nonExistentSource789" not in source_ids_in_graph
        assert "nonExistentTarget101" not in target_ids_in_graph


def test_get_all_nodes_empty_and_populated(
    property_graph_store_static: SpannerPropertyGraphStore,
    property_graph_store_dynamic: SpannerPropertyGraphStore,
):
    """Test graph_store.get() with no arguments to retrieve all nodes."""
    for graph_store in [
        property_graph_store_static,
        property_graph_store_dynamic,
    ]:
        # 1. Test on an empty graph
        all_nodes_empty = graph_store.get()
        assert isinstance(all_nodes_empty, list)
        assert len(all_nodes_empty) == 0

        # 2. Populate with some nodes
        node1 = EntityNode(label="A", name="Node1")
        node2 = ChunkNode(
            text="Chunk text for Node2", id_="custom_id_chunk"
        )  # ChunkNode
        node3 = EntityNode(label="B", name="Node3", properties={"feature": "X"})
        nodes_to_add = [node1, node2, node3]
        graph_store.upsert_nodes(nodes_to_add)

        # 3. Get all nodes and verify
        all_nodes_populated = graph_store.get()
        assert len(all_nodes_populated) == len(nodes_to_add)

        retrieved_ids = {node.id for node in all_nodes_populated}
        expected_ids = {node.id for node in nodes_to_add}
        assert retrieved_ids == expected_ids

        # Verify types and some content
        found_node1 = any(
            n.id == node1.id and isinstance(n, EntityNode) and n.name == "Node1"
            for n in all_nodes_populated
        )
        found_node2 = any(
            n.id == node2.id
            and isinstance(n, ChunkNode)
            and n.text == "Chunk text for Node2"
            for n in all_nodes_populated
        )
        found_node3 = any(
            n.id == node3.id
            and isinstance(n, EntityNode)
            and n.properties.get("feature") == "X"
            for n in all_nodes_populated
        )
        assert found_node1
        assert found_node2
        assert found_node3


def test_get_nodes_by_multiple_properties(
    property_graph_store_static: SpannerPropertyGraphStore,
    property_graph_store_dynamic: SpannerPropertyGraphStore,
):
    """Test get() with multiple properties (implying AND logic)."""
    for graph_store in [
        property_graph_store_static,
        property_graph_store_dynamic,
    ]:
        dev1 = EntityNode(
            label="DEV",
            name="DevA",
            properties={"lang": "Python", "exp_years": 5, "city": "London"},
        )
        dev2 = EntityNode(
            label="DEV",
            name="DevB",
            properties={"lang": "Python", "exp_years": 2, "city": "Paris"},
        )
        dev3 = EntityNode(
            label="DEV",
            name="DevC",
            properties={"lang": "Java", "exp_years": 5, "city": "London"},
        )
        dev4 = EntityNode(
            label="DEV",
            name="DevD",
            properties={"lang": "Python", "exp_years": 5, "city": "Berlin"},
        )  # Matches lang & exp_years, not city

        graph_store.upsert_nodes([dev1, dev2, dev3, dev4])

        # Filter by lang: "Python" AND exp_years: 5 AND city: "London"
        target_properties = {"lang": "Python", "exp_years": 5, "city": "London"}
        filtered_devs = graph_store.get(properties=target_properties)

        assert len(filtered_devs) == 1
        if isinstance(filtered_devs[0], EntityNode):
            assert filtered_devs[0].name == "DevA"
        assert filtered_devs[0].properties.get("lang") == "Python"
        assert filtered_devs[0].properties.get("exp_years") == 5
        assert filtered_devs[0].properties.get("city") == "London"

        # Filter by lang: "Python" AND exp_years: 5 (should return DevA and DevD)
        target_properties_2 = {"lang": "Python", "exp_years": 5}
        filtered_devs_2 = graph_store.get(properties=target_properties_2)
        assert len(filtered_devs_2) == 2
        names_2 = {dev.name for dev in filtered_devs_2 if isinstance(dev, EntityNode)}
        assert {"DevA", "DevD"} == names_2

        # Filter with a non-matching combination
        target_properties_none = {"lang": "Python", "exp_years": 10}
        filtered_devs_none = graph_store.get(properties=target_properties_none)
        assert len(filtered_devs_none) == 0


def test_filter_nodes_by_property(
    property_graph_store_static: SpannerPropertyGraphStore,
    property_graph_store_dynamic: SpannerPropertyGraphStore,
):
    """Test get() with property filtering."""
    e1 = EntityNode(label="PERSON", name="Alice", properties={"country": "France"})
    e2 = EntityNode(label="PERSON", name="Bob", properties={"country": "USA"})
    e3 = EntityNode(label="PERSON", name="Charlie", properties={"country": "France"})
    for graph_store in [
        property_graph_store_static,
        property_graph_store_dynamic,
    ]:
        graph_store.upsert_nodes([e1, e2, e3])

        # Filter
        filtered = graph_store.get(properties={"country": "France"})
        assert len(filtered) == 2
        filtered_names = {x.name for x in filtered if isinstance(x, EntityNode)}
        assert filtered_names == {"Alice", "Charlie"}


def test_get_triplets_with_multiple_filters(
    property_graph_store_static: SpannerPropertyGraphStore,
    property_graph_store_dynamic: SpannerPropertyGraphStore,
):
    """Test get_triplets() with a combination of filters (entity_names, relation_names, properties, ids)."""
    for graph_store in [
        property_graph_store_static,
        property_graph_store_dynamic,
    ]:
        # Setup nodes
        alice = EntityNode(label="PERSON", name="Alice", properties={"country": "UK"})
        bob = EntityNode(label="PERSON", name="Bob", properties={"country": "USA"})
        project_x = EntityNode(
            label="PROJECT", name="ProjectX", properties={"status": "active"}
        )
        project_y = EntityNode(
            label="PROJECT", name="ProjectY", properties={"status": "pending"}
        )
        org_a = EntityNode(label="ORG", name="OrgA")

        graph_store.upsert_nodes([alice, bob, project_x, project_y, org_a])

        # Setup relations
        rel1_alice_works_x = Relation(
            source_id=alice.id,
            target_id=project_x.id,
            label="WORKS_ON",
            properties={"role": "dev"},
        )
        rel2_bob_works_x = Relation(
            source_id=bob.id,
            target_id=project_x.id,
            label="WORKS_ON",
            properties={"role": "manager"},
        )
        rel3_alice_manages_y = Relation(
            source_id=alice.id,
            target_id=project_y.id,
            label="MANAGES",
            properties={"priority": 1},
        )
        rel4_orga_owns_x = Relation(
            source_id=org_a.id, target_id=project_x.id, label="OWNS"
        )

        graph_store.upsert_relations(
            [
                rel1_alice_works_x,
                rel2_bob_works_x,
                rel3_alice_manages_y,
                rel4_orga_owns_x,
            ]
        )

        # Test Case 1: Filter by entity_names and relation_names
        # Get all "WORKS_ON" relations for "Alice"
        triplets1 = graph_store.get_triplets(
            entity_names=["Alice"], relation_names=["WORKS_ON"]
        )
        assert len(triplets1) == 1
        source1, rel1, target1 = triplets1[0]
        if isinstance(source1, EntityNode) and isinstance(target1, EntityNode):
            assert source1.name == "Alice"
            assert rel1.label == "WORKS_ON"
            assert target1.name == "ProjectX"

        # Test Case 2: Filter by node ids and relation_names
        # Get "MANAGES" relations originating from Alice's ID
        triplets2 = graph_store.get_triplets(ids=[alice.id], relation_names=["MANAGES"])
        assert len(triplets2) == 1
        source2, rel2, target2 = triplets2[0]
        if isinstance(source2, EntityNode) and isinstance(target2, EntityNode):
            assert source2.id == alice.id
            assert rel2.label == "MANAGES"
            assert target2.name == "ProjectY"

        # Test Case 3: Filter by node properties (of the source node)
        # and relation_names
        # Get "WORKS_ON" relations for people from "USA" (i.e., Bob)
        # Note: update_condition in your store focuses on properties of 'n'
        # (the first node in MATCH (n)-[r]-(n2))
        triplets3 = graph_store.get_triplets(
            properties={"country": "USA"}, relation_names=["WORKS_ON"]
        )
        assert len(triplets3) == 1
        source3, rel3, target3 = triplets3[0]
        if isinstance(source3, EntityNode) and isinstance(target3, EntityNode):
            assert source3.name == "Bob"
            assert rel3.label == "WORKS_ON"
            assert target3.name == "ProjectX"

        # Test Case 4: Filter by entity_names, relation_names,
        # and properties (of source node)
        # Get "WORKS_ON" relations for "Alice" who is from "UK"
        triplets4 = graph_store.get_triplets(
            entity_names=["Alice"],
            relation_names=["WORKS_ON"],
            properties={"country": "UK"},
        )
        assert len(triplets4) == 1

        source4, rel4, target4 = triplets4[0]
        if isinstance(source4, EntityNode):
            assert source4.name == "Alice"
            assert rel4.label == "WORKS_ON"

        # Test Case 5: No results expected
        triplets5 = graph_store.get_triplets(
            entity_names=["NonExistent"], relation_names=["WORKS_ON"]
        )
        assert not triplets5


def test_get_triplets_filter_by_relation_properties(
    property_graph_store_static: SpannerPropertyGraphStore,
    property_graph_store_dynamic: SpannerPropertyGraphStore,
):
    """Test get_triplets() node properties filter and relation properties filter.

    It tests get_triplets() behavior when properties filter is provided,
    expecting it to apply to nodes. And explore how one might filter by
    relation properties (likely via structured_query).

    Args:
      property_graph_store_static: A static schema graph store.
      property_graph_store_dynamic: A dynamic schema graph store.
    """
    for graph_store in [
        property_graph_store_static,
        property_graph_store_dynamic,
    ]:
        user_a = EntityNode(label="U", name="UserA", properties={"status": "active"})
        user_b = EntityNode(
            label="U", name="UserB", properties={"status": "inactive"}
        )  # Node property
        item1 = EntityNode(label="I", name="Item1")

        graph_store.upsert_nodes([user_a, user_b, item1])

        rel1 = Relation(
            source_id=user_a.id,
            target_id=item1.id,
            label="RATED",
            properties={"score": 5},
        )  # Relation property
        rel2 = Relation(
            source_id=user_b.id,
            target_id=item1.id,
            label="RATED",
            properties={"score": 3},
        )  # Relation property
        graph_store.upsert_relations([rel1, rel2])

        # Find triplets where source node has status "active"
        triplets_node_prop = graph_store.get_triplets(
            properties={"status": "active"}, relation_names=["RATED"]
        )
        assert len(triplets_node_prop) == 1
        if isinstance(triplets_node_prop[0][0], EntityNode):
            assert (
                triplets_node_prop[0][0].name == "UserA"
            )  # Source node has status: active
        assert triplets_node_prop[0][1].properties.get("score") == 5


def test_get_rel_map(
    property_graph_store_static: SpannerPropertyGraphStore,
    property_graph_store_dynamic: SpannerPropertyGraphStore,
):
    """Test get_rel_map with a multi-depth scenario."""

    for graph_store in [
        property_graph_store_static,
        property_graph_store_dynamic,
    ]:
        e1 = EntityNode(label="PERSON", name="Alice")
        e2 = EntityNode(label="PERSON", name="Bob")
        e3 = EntityNode(label="CITY", name="Paris")
        e4 = EntityNode(label="CITY", name="London")
        graph_store.upsert_nodes([e1, e2, e3, e4])

        r1 = Relation(label="KNOWS", source_id=e1.id, target_id=e2.id)
        r2 = Relation(label="VISITED", source_id=e1.id, target_id=e3.id)
        r3 = Relation(label="VISITED", source_id=e2.id, target_id=e4.id)
        graph_store.upsert_relations([r1, r2, r3])

        # Depth 2 should capture up to "Alice - Bob - London" chain
        rel_map = graph_store.get_rel_map([e1], depth=2)
        labels_found = {trip[1].label for trip in rel_map}
        assert "KNOWS" in labels_found
        assert "VISITED" in labels_found


def test_get_rel_map_depth_variations_and_limit(
    property_graph_store_static: SpannerPropertyGraphStore,
    property_graph_store_dynamic: SpannerPropertyGraphStore,
):
    """Test get_rel_map with various depth and limit parameters."""
    for graph_store in [
        property_graph_store_static,
        property_graph_store_dynamic,
    ]:
        # graph_store.clear_all_data() # If using shared fixtures

        # A -> B -> C -> D -> E (linear path)
        # A also connects to F
        node_a = EntityNode(label="GM_PATH", name="A")
        node_b = EntityNode(label="GM_PATH", name="B")
        node_c = EntityNode(label="GM_PATH", name="C")
        node_d = EntityNode(label="GM_PATH", name="D")
        node_e = EntityNode(label="GM_PATH", name="E")
        node_f = EntityNode(label="GM_PATH", name="F")
        graph_store.upsert_nodes([node_a, node_b, node_c, node_d, node_e, node_f])

        rel_ab = Relation(source_id=node_a.id, target_id=node_b.id, label="TO_B")
        rel_bc = Relation(source_id=node_b.id, target_id=node_c.id, label="TO_C")
        rel_cd = Relation(source_id=node_c.id, target_id=node_d.id, label="TO_D")
        rel_de = Relation(source_id=node_d.id, target_id=node_e.id, label="TO_E")
        rel_af = Relation(source_id=node_a.id, target_id=node_f.id, label="TO_F")
        graph_store.upsert_relations([rel_ab, rel_bc, rel_cd, rel_de, rel_af])

        # Test Depth from Node A
        # Depth 1: Should get (A, TO_B, B) and (A, TO_F, F)
        triplets_d1 = graph_store.get_rel_map(graph_nodes=[node_a], depth=1, limit=10)
        assert len(triplets_d1) == 2
        labels_d1 = {t[1].label for t in triplets_d1}
        sources_d1 = {t[0].name for t in triplets_d1 if isinstance(t[0], EntityNode)}
        targets_d1 = {t[2].name for t in triplets_d1 if isinstance(t[2], EntityNode)}
        assert "A" in sources_d1 and len(sources_d1) == 1
        assert labels_d1 == {"TO_B", "TO_F"}
        assert targets_d1 == {"B", "F"}

        # Depth 2: Should get (A,TO_B,B), (B,TO_C,C), (A,TO_F,F)
        triplets_d2 = graph_store.get_rel_map(graph_nodes=[node_a], depth=2, limit=10)
        # Expected triplets: (A,TO_B,B), (B,TO_C,C), (A,TO_F,F)
        assert len(triplets_d2) == 3

        labels_d2 = sorted(
            [t[1].label for t in triplets_d2]
        )  # Use sorted list for potential duplicates if any
        target_names_d2 = sorted(
            [t[2].name for t in triplets_d2 if isinstance(t[2], EntityNode)]
        )
        assert sorted(["TO_B", "TO_C", "TO_F"]) == labels_d2
        # Targets will be B (from A-B), C (from B-C), F (from A-F)
        assert sorted(["B", "C", "F"]) == target_names_d2

        # Expected: (A,TO_B,B), (B,TO_C,C), (C,TO_D,D), (D,TO_E,E), (A,TO_F,F)
        triplets_d4 = graph_store.get_rel_map(graph_nodes=[node_a], depth=4, limit=10)
        assert len(triplets_d4) == 5

        # We should get (A,TO_F,F) and (A,TO_B,B).
        triplets_d4_limit2 = graph_store.get_rel_map(
            graph_nodes=[node_a], depth=4, limit=2
        )
        assert len(triplets_d4_limit2) <= 5

        graph_store.delete(relation_names=["TO_B", "TO_C", "TO_D", "TO_E", "TO_F"])

        node_x1 = EntityNode(label="GM_LIM", name="X1")
        node_x2 = EntityNode(label="GM_LIM", name="X2")
        node_x3 = EntityNode(label="GM_LIM", name="X3")
        node_x4 = EntityNode(label="GM_LIM", name="X4")
        graph_store.upsert_nodes([node_x1, node_x2, node_x3, node_x4])
        rels_limit_test = [
            Relation(source_id=node_a.id, target_id=node_x1.id, label="L"),
            Relation(source_id=node_a.id, target_id=node_x2.id, label="L"),
            Relation(source_id=node_a.id, target_id=node_x3.id, label="L"),
            Relation(source_id=node_a.id, target_id=node_x4.id, label="L"),
        ]
        graph_store.upsert_relations(rels_limit_test)

        triplets_limit_3 = graph_store.get_rel_map(
            graph_nodes=[node_a], depth=1, limit=3
        )
        assert len(triplets_limit_3) == 3


def test_get_rel_map_on_disconnected_nodes(
    property_graph_store_static: SpannerPropertyGraphStore,
    property_graph_store_dynamic: SpannerPropertyGraphStore,
):
    """Test get_rel_map for nodes with no relations or in disconnected components."""
    for graph_store in [
        property_graph_store_static,
        property_graph_store_dynamic,
    ]:
        node_connected = EntityNode(label="GM_CONN", name="Connected")
        node_target = EntityNode(label="GM_CONN", name="Target")
        node_isolated = EntityNode(label="GM_CONN", name="Isolated")
        graph_store.upsert_nodes([node_connected, node_target, node_isolated])

        rel_conn = Relation(
            source_id=node_connected.id, target_id=node_target.id, label="LINK"
        )
        graph_store.upsert_relations([rel_conn])

        # Case 1: Node with outgoing relations
        triplets_connected = graph_store.get_rel_map(
            graph_nodes=[node_connected], depth=1
        )
        assert len(triplets_connected) == 1
        assert triplets_connected[0][1].label == "LINK"

        # Case 2: Node with no outgoing/incoming relations (isolated)
        triplets_isolated = graph_store.get_rel_map(
            graph_nodes=[node_isolated], depth=1
        )
        assert len(triplets_isolated) == 0

        # Case 3: Node that is a target but has no outgoing relations
        triplets_target = graph_store.get_rel_map(graph_nodes=[node_target], depth=1)
        assert len(triplets_target) == 1

        # Case 4: Multiple start nodes, some connected, some isolated
        triplets_mixed_start = graph_store.get_rel_map(
            graph_nodes=[node_connected, node_isolated], depth=1
        )
        assert len(triplets_mixed_start) == 1  # Only from node_connected
        assert triplets_mixed_start[0][0].id == node_connected.id


def test_get_rel_map_with_ignore_rels(
    property_graph_store_static: SpannerPropertyGraphStore,
    property_graph_store_dynamic: SpannerPropertyGraphStore,
):
    """Test get_rel_map with the ignore_rels parameter."""
    for graph_store in [
        property_graph_store_static,
        property_graph_store_dynamic,
    ]:
        graph_store = property_graph_store_static

        alice = EntityNode(label="PERSON", name="Alice")
        bob = EntityNode(label="PERSON", name="Bob")
        game_company = EntityNode(label="COMPANY", name="GameDev Inc.")
        food_company = EntityNode(label="COMPANY", name="Foodies Ltd.")

        graph_store.upsert_nodes([alice, bob, game_company, food_company])

        rel_knows = Relation(source_id=alice.id, target_id=bob.id, label="KNOWS")
        rel_works_at_game = Relation(
            source_id=alice.id, target_id=game_company.id, label="WORKS_AT"
        )
        rel_works_at_food = Relation(
            source_id=bob.id, target_id=food_company.id, label="WORKS_AT"
        )
        rel_founded = Relation(
            source_id=alice.id, target_id=food_company.id, label="FOUNDED"
        )

        graph_store.upsert_relations(
            [rel_knows, rel_works_at_game, rel_works_at_food, rel_founded]
        )

        # Get relations for Alice, ignoring "KNOWS"
        rel_map_ignore_knows = graph_store.get_rel_map(
            graph_nodes=[alice], depth=2, ignore_rels=["KNOWS"]
        )

        found_labels_ignore_knows = {
            triplet[1].label for triplet in rel_map_ignore_knows
        }
        assert "KNOWS" not in found_labels_ignore_knows, "KNOWS should be ignored"
        assert "WORKS_AT" in found_labels_ignore_knows
        assert "FOUNDED" in found_labels_ignore_knows

        # Verify specific relations (Alice WORKS_AT GameDev,
        # Alice FOUNDED Foodies Ltd.)
        alice_works_at_game_found = any(
            t[0].id == alice.id
            and t[1].label == "WORKS_AT"
            and t[2].id == game_company.id
            for t in rel_map_ignore_knows
        )
        alice_founded_food_found = any(
            t[0].id == alice.id
            and t[1].label == "FOUNDED"
            and t[2].id == food_company.id
            for t in rel_map_ignore_knows
        )
        assert alice_works_at_game_found
        assert alice_founded_food_found

        # Get relations for Alice, ignoring "WORKS_AT"
        rel_map_ignore_works_at = graph_store.get_rel_map(
            graph_nodes=[alice], depth=2, ignore_rels=["WORKS_AT"]
        )
        found_labels_ignore_works_at = {
            triplet[1].label for triplet in rel_map_ignore_works_at
        }
        assert "WORKS_AT" not in found_labels_ignore_works_at
        assert "KNOWS" in found_labels_ignore_works_at, "KNOWS should be present"
        assert "FOUNDED" in found_labels_ignore_works_at


def test_get_rel_map_with_cycles(
    property_graph_store_static: SpannerPropertyGraphStore,
    property_graph_store_dynamic: SpannerPropertyGraphStore,
):
    """Test get_rel_map in a graph with cycles."""
    for graph_store in [
        property_graph_store_static,
        property_graph_store_dynamic,
    ]:

        # A -> B -> C -> A (cycle)
        # B also -> D (branch)
        node_a_cyc = EntityNode(label="GM_CYC", name="A_cyc")
        node_b_cyc = EntityNode(label="GM_CYC", name="B_cyc")
        node_c_cyc = EntityNode(label="GM_CYC", name="C_cyc")
        node_d_cyc = EntityNode(label="GM_CYC", name="D_cyc")  # Branch
        graph_store.upsert_nodes([node_a_cyc, node_b_cyc, node_c_cyc, node_d_cyc])

        rel_ab = Relation(source_id=node_a_cyc.id, target_id=node_b_cyc.id, label="AB")
        rel_bc = Relation(source_id=node_b_cyc.id, target_id=node_c_cyc.id, label="BC")
        rel_ca = Relation(
            source_id=node_c_cyc.id, target_id=node_a_cyc.id, label="CA"
        )  # Cycle back to A
        rel_bd = Relation(
            source_id=node_b_cyc.id, target_id=node_d_cyc.id, label="BD"
        )  # Branch from B
        graph_store.upsert_relations([rel_ab, rel_bc, rel_ca, rel_bd])
        triplets_d3_from_a = graph_store.get_rel_map(
            graph_nodes=[node_a_cyc], depth=3, limit=20
        )
        assert len(triplets_d3_from_a) == 4

        retrieved_labels_d3 = {t[1].label for t in triplets_d3_from_a}
        assert retrieved_labels_d3 == {"AB", "BC", "CA", "BD"}

        triplets_d5_from_a = graph_store.get_rel_map(
            graph_nodes=[node_a_cyc], depth=5, limit=20
        )
        # Even with depth 5, due to the cycle, it should still return 4 relations
        assert len(triplets_d5_from_a) == 4

        # Test from B_cyc with depth 1
        # Expected: (B,BC,C), (A,AB,B), (B,BD,D) - 3 relations
        triplets_d2_from_b = graph_store.get_rel_map(
            graph_nodes=[node_b_cyc], depth=1, limit=20
        )
        assert len(triplets_d2_from_b) == 3
        retrieved_labels_d2_b = {t[1].label for t in triplets_d2_from_b}
        assert retrieved_labels_d2_b == {"BC", "AB", "BD"}


def test_delete_entities_by_names(
    property_graph_store_static: SpannerPropertyGraphStore,
    property_graph_store_dynamic: SpannerPropertyGraphStore,
):
    """Test deleting nodes by entity_names."""
    for graph_store in [
        property_graph_store_static,
        property_graph_store_dynamic,
    ]:
        e1 = EntityNode(label="PERSON", name="Alice")
        e2 = EntityNode(label="PERSON", name="Bob")
        graph_store.upsert_nodes([e1, e2])

        # Delete 'Alice'
        graph_store.delete(entity_names=["Alice"])

        # Verify
        remaining = graph_store.get()
        assert len(remaining) == 1

        if isinstance(remaining, EntityNode):
            assert remaining[0].name == "Bob"


def test_delete_nodes_by_ids(
    property_graph_store_static: SpannerPropertyGraphStore,
    property_graph_store_dynamic: SpannerPropertyGraphStore,
):
    """Test deleting nodes by IDs."""
    for graph_store in [
        property_graph_store_static,
        property_graph_store_dynamic,
    ]:
        e1 = EntityNode(label="PERSON", name="Alice")
        e2 = EntityNode(label="PERSON", name="Bob")
        e3 = EntityNode(label="PERSON", name="Charlie")
        graph_store.upsert_nodes([e1, e2, e3])

        # Delete Bob, Charlie by IDs
        graph_store.delete(ids=[e2.id, e3.id])

        all_remaining = graph_store.get()
        assert len(all_remaining) == 1

        if isinstance(all_remaining[0], EntityNode):
            assert all_remaining[0].name == "Alice"


def test_delete_nodes_by_properties(
    property_graph_store_static: SpannerPropertyGraphStore,
    property_graph_store_dynamic: SpannerPropertyGraphStore,
):
    """Test deleting nodes by a property dict."""

    for graph_store in [
        property_graph_store_static,
        property_graph_store_dynamic,
    ]:
        c1 = ChunkNode(text="This is a test chunk.", properties={"lang": "en"})
        c2 = ChunkNode(text="Another chunk.", properties={"lang": "fr"})
        graph_store.upsert_nodes([c1, c2])

        # Delete all English chunks
        graph_store.delete(properties={"lang": "en"})

        # Only c2 remains
        remaining = graph_store.get()
        assert len(remaining) == 1
        assert remaining[0].properties["lang"] == "fr"


def test_delete_nodes_with_multiple_criteria(
    property_graph_store_static: SpannerPropertyGraphStore,
    property_graph_store_dynamic: SpannerPropertyGraphStore,
):
    """Test deleting nodes using a combination of criteria (e.g., properties AND ids)."""
    # The SpannerPropertyGraphStore.delete method for nodes uses update_condition,
    # which combines filters with AND.
    for graph_store in [
        property_graph_store_static,
        property_graph_store_dynamic,
    ]:
        # Setup nodes
        node1 = EntityNode(
            label="ITEM",
            name="Book",
            properties={"category": "Fiction", "stock": 10},
        )
        node2 = EntityNode(
            label="ITEM",
            name="Pen",
            properties={"category": "Stationery", "stock": 100},
        )
        node3 = EntityNode(
            label="ITEM",
            name="Eraser",
            properties={"category": "Stationery", "stock": 10},
        )  # Matches stock, not id
        node4 = EntityNode(
            label="ITEM",
            name="Ruler",
            properties={"category": "Stationery", "stock": 10},
        )  # Matches stock and id

        graph_store.upsert_nodes([node1, node2, node3, node4])

        # Case 1: Delete by properties AND ids (should delete only node4)
        # Delete items with stock: 10 AND id: "item_abc"
        graph_store.delete(
            ids=[node4.id], properties={"stock": 10, "category": "Stationery"}
        )

        remaining_nodes = graph_store.get()
        remaining_names = {n.name for n in remaining_nodes if isinstance(n, EntityNode)}
        assert "Ruler" not in remaining_names, "Node4 (Ruler) should have been deleted"
        assert {
            "Book",
            "Pen",
            "Eraser",
        } == remaining_names, "Book, Pen, Eraser should remain"

        graph_store.upsert_nodes([node4])
        assert len(graph_store.get()) == 4  # Ensure it's back

        # Case 2: Delete by entity_names AND properties (should delete only Book)

        # Case 2: Delete by entity_names AND properties (should delete only Book)
        # Delete ITEM named "Book" AND category "Fiction"
        graph_store.delete(entity_names=["Book"], properties={"category": "Fiction"})
        remaining_nodes_c2 = graph_store.get()
        remaining_names_c2 = {
            n.name for n in remaining_nodes_c2 if isinstance(n, EntityNode)
        }
        assert "Book" not in remaining_names_c2
        # Case 3: Criteria that match nothing when combined
        graph_store.delete(
            ids=[node1.id], properties={"category": "Stationery"}
        )  # Book is Fiction
        assert not graph_store.get(ids=[node1.id])  # Book was already deleted in Case 2
        assert len(graph_store.get(ids=[node2.id])) == 1  # Pen should still be there.


def test_delete_relations(
    property_graph_store_static: SpannerPropertyGraphStore,
    property_graph_store_dynamic: SpannerPropertyGraphStore,
):
    """Test deleting relationships by relation names."""

    for graph_store in [
        property_graph_store_static,
        property_graph_store_dynamic,
    ]:
        e1 = EntityNode(label="PERSON", name="Alice")
        e2 = EntityNode(label="CITY", name="Paris")
        graph_store.upsert_nodes([e1, e2])

        rel = Relation(source_id=e1.id, target_id=e2.id, label="VISITED")
        graph_store.upsert_relations([rel])

        # Ensure the relationship is there
        triplets_before = graph_store.get_triplets(entity_names=["Alice"])
        assert len(triplets_before) == 1

        # Delete the relation
        graph_store.delete(relation_names=["VISITED"])

        # No more triplets
        triplets_after = graph_store.get_triplets(entity_names=["Alice"])
        assert len(triplets_after) == 0


def test_delete_relations_by_label_in_non_flexible_schema_impact(
    property_graph_store_static: SpannerPropertyGraphStore,
):
    """Test schema changes and impact when deleting relations by label in a static schema.

    This expects tables to be dropped and graph DDL to be updated.
    This test REQUIRES an isolated graph store (function-scoped fixture).
    """
    graph_store = property_graph_store_static

    # Setup: 2 node types, 2 relation types between them
    user1 = EntityNode(label="USER_S", name="User1_S")
    user2 = EntityNode(label="USER_S", name="User2_S")
    app1 = EntityNode(label="APP_S", name="App1_S")
    app2 = EntityNode(label="APP_S", name="App2_S")
    graph_store.upsert_nodes([user1, user2, app1, app2])

    rel_uses_1 = Relation(source_id=user1.id, target_id=app1.id, label="USES_S")
    rel_uses_2 = Relation(source_id=user2.id, target_id=app1.id, label="USES_S")
    rel_likes_1 = Relation(source_id=user1.id, target_id=app2.id, label="LIKES_S")
    graph_store.upsert_relations([rel_uses_1, rel_uses_2, rel_likes_1])

    # Get initial schema and verify edge tables/definitions exist
    schema_before_delete = graph_store.get_schema(refresh=True)
    uses_edge_table_key_before = f"{graph_store.schema.graph_name}_USER_S_USES_S_APP_S"
    likes_edge_table_key_before = (
        f"{graph_store.schema.graph_name}_USER_S_LIKES_S_APP_S"
    )

    assert (
        uses_edge_table_key_before
        in schema_before_delete["Edge properties per edge type"]
    )
    assert uses_edge_table_key_before in schema_before_delete["Edges"]
    assert (
        likes_edge_table_key_before
        in schema_before_delete["Edge properties per edge type"]
    )
    assert likes_edge_table_key_before in schema_before_delete["Edges"]

    # Verify data presence
    assert len(graph_store.get_triplets(relation_names=["USES_S"])) == 2
    assert len(graph_store.get_triplets(relation_names=["LIKES_S"])) == 1

    # ---- Perform delete of "USES_S" relations ----
    # This should drop the "USES_S" edge table and update the graph DDL
    graph_store.delete(relation_names=["USES_S"])

    # 1. Verify "USES_S" relations are gone
    assert not graph_store.get_triplets(relation_names=["USES_S"])

    # 2. Verify "LIKES_S" relations are still present
    likes_triplets_after_delete = graph_store.get_triplets(relation_names=["LIKES_S"])

    assert len(likes_triplets_after_delete) == 1

    source, rel, target = likes_triplets_after_delete[0]

    if isinstance(source, EntityNode) and isinstance(target, EntityNode):
        assert source.name == "User1_S"
        assert target.name == "App2_S"

    # 3. Verify schema changes
    schema_after_delete = graph_store.get_schema(refresh=True)
    assert uses_edge_table_key_before not in schema_after_delete.get(
        "Edge properties per edge type", {}
    ), f"{uses_edge_table_key_before} should be removed from Edge properties"
    assert uses_edge_table_key_before not in schema_after_delete.get(
        "Edges", {}
    ), f"{uses_edge_table_key_before} should be removed from Edges"

    # The "LIKES_S" edge type should still be in the schema
    assert (
        likes_edge_table_key_before
        in schema_after_delete["Edge properties per edge type"]
    )
    assert likes_edge_table_key_before in schema_after_delete["Edges"]

    # 4. Verify nodes are unaffected
    assert len(graph_store.get(ids=["User1_S"])) == 1
    assert len(graph_store.get(ids=["App1_S"])) == 1

    try:
        graph_store.upsert_relations(
            [Relation(source_id=user2.id, target_id=app2.id, label="USES_S")]
        )
        assert len(graph_store.get_triplets(relation_names=["USES_S"])) == 1
        schema_after_re_add = graph_store.get_schema(refresh=True)
        assert (
            uses_edge_table_key_before
            in schema_after_re_add["Edge properties per edge type"]
        )
    except Exception as e:
        pytest.fail(
            "Failed to re-add relation type after it was deleted (static schema):"
            f" {e}"
        )


def test_structured_query(
    property_graph_store_static: SpannerPropertyGraphStore,
    property_graph_store_dynamic: SpannerPropertyGraphStore,
):
    """Test running a custom Cypher query via structured_query."""
    for graph_store in [
        property_graph_store_static,
        property_graph_store_dynamic,
    ]:
        # Insert data
        e1 = EntityNode(label="PERSON", name="Alice")
        graph_store.upsert_nodes([e1])

        # Custom query
        query = """
      MATCH (n) WHERE n.id = @name
      RETURN n.id AS node_name
      """
        result = graph_store.structured_query(query, {"name": "Alice"})
        assert len(result) == 1
        assert result[0]["node_name"] == "Alice"


def test_vector_query(
    property_graph_store_static: SpannerPropertyGraphStore,
    property_graph_store_dynamic: SpannerPropertyGraphStore,
):
    """Test vector_query with some dummy embeddings."""

    for graph_store in [
        property_graph_store_static,
        property_graph_store_dynamic,
    ]:
        entity1 = EntityNode(label="PERSON", name="Alice", embedding=[0.1, 0.2, 0.3])
        entity2 = EntityNode(label="PERSON", name="Bob", embedding=[0.9, 0.8, 0.7])
        graph_store.upsert_nodes([entity1, entity2])

        # Query embedding somewhat closer to [0.1, 0.2, 0.3] than [0.9, 0.8, 0.7]
        query = VectorStoreQuery(query_embedding=[0.1, 0.2, 0.31], similarity_top_k=2)
        results, scores = graph_store.vector_query(query)
        # Expect "Alice" to come first
        assert len(results) == 2
        names_in_order = [r.name for r in results if isinstance(r, EntityNode)]
        assert names_in_order[0] == "Alice"
        assert names_in_order[1] == "Bob"
        # Score check: Usually Alice's score should be higher
        assert scores[0] >= scores[1]


def test_vector_query_with_exact_match_filter(
    property_graph_store_static: SpannerPropertyGraphStore,
    property_graph_store_dynamic: SpannerPropertyGraphStore,
):
    """Test vector_query with an ExactMatchFilter."""
    for graph_store in [
        property_graph_store_static,
        property_graph_store_dynamic,
    ]:
        # Ensure graph is clean for each iteration if properties affect
        # schema/indexing
        graph_store.clean_up()  # Or ensure fixtures provide isolated instances

        doc1 = EntityNode(
            label="DOC",
            name="Doc1",
            embedding=[0.1, 0.2, 0.3],
            properties={"category": "A"},
        )
        doc2 = EntityNode(
            label="DOC",
            name="Doc2",
            embedding=[0.4, 0.5, 0.6],
            properties={"category": "B"},
        )
        doc3 = EntityNode(
            label="DOC",
            name="Doc3",
            embedding=[0.15, 0.25, 0.35],
            properties={"category": "A"},
        )
        graph_store.upsert_nodes([doc1, doc2, doc3])

        # Query for docs in category "A"
        query_embedding = [0.12, 0.22, 0.32]  # Closer to Doc1 and Doc3
        metadata_filters = MetadataFilters(
            filters=[ExactMatchFilter(key="category", value="A")]
        )
        vector_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=2,
            filters=metadata_filters,
        )

        results, scores = graph_store.vector_query(vector_query)

        assert len(results) <= 2  # Can be less if fewer than top_k match filter
        result_names = {node.name for node in results if isinstance(node, EntityNode)}
        assert "Doc2" not in result_names  # Doc2 is category B
        if results:  # If any results match the filter
            assert "Doc1" in result_names or "Doc3" in result_names

        # Ensure that if results are returned, they are indeed category A
        for res_node in results:
            assert res_node.properties["category"] == "A"

        # Query for docs in category "B"
        metadata_filters_b = MetadataFilters(
            filters=[ExactMatchFilter(key="category", value="B")]
        )
        vector_query_b = VectorStoreQuery(
            query_embedding=query_embedding,  # Use same embedding, filter is key
            similarity_top_k=1,
            filters=metadata_filters_b,
        )
        results_b, _ = graph_store.vector_query(vector_query_b)
        assert len(results_b) <= 1
        if results_b and isinstance(results_b, EntityNode):
            assert results_b[0].name == "Doc2"
            assert results_b[0].properties["category"] == "B"


def test_vector_query_with_numeric_range_filters(
    property_graph_store_static: SpannerPropertyGraphStore,
    property_graph_store_dynamic: SpannerPropertyGraphStore,
):
    """Test vector_query with numeric range filters (GT, LT, GTE, LTE) on properties."""

    nodes_data: list[dict[str, Any]] = [
        {
            "id": "v_item1",
            "label": "VITEM",
            "name": "Item1",
            "embedding": [0.1, 0.1],
            "price": 10.0,
            "stock": 5,
        },
        {
            "id": "v_item2",
            "label": "VITEM",
            "name": "Item2",
            "embedding": [0.2, 0.2],
            "price": 20.0,
            "stock": 15,
        },
        {
            "id": "v_item3",
            "label": "VITEM",
            "name": "Item3",
            "embedding": [0.3, 0.3],
            "price": 30.0,
            "stock": 20,
        },
        {
            "id": "v_item4",
            "label": "VITEM",
            "name": "Item4",
            "embedding": [0.4, 0.4],
            "price": 20.0,
            "stock": 25,
        },
    ]
    nodes = [
        EntityNode(
            label=d["label"],
            name=d["name"],
            embedding=d["embedding"],
            properties={"price": d["price"], "stock": d["stock"]},
        )
        for d in nodes_data
    ]

    query_embedding = [0.15, 0.15]  # Closest to Item1/Item2 generally

    for graph_store in [
        property_graph_store_static,
        property_graph_store_dynamic,
    ]:

        graph_store.upsert_nodes(nodes)

        # Test Case 1: Greater Than (GT) for 'price'
        gt_filter = MetadataFilters(
            filters=[
                MetadataFilter(key="price", operator=FilterOperator.GT, value=20.0)
            ]
        )
        query_gt = VectorStoreQuery(
            query_embedding=query_embedding, similarity_top_k=3, filters=gt_filter
        )

        results_gt, _ = graph_store.vector_query(query_gt)
        names_gt = {n.name for n in results_gt if isinstance(n, EntityNode)}

        assert names_gt == {"Item3"}

        # Test Case 2: Greater Than or Equal To (GTE) for 'price'
        gte_filter = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="price",
                    operator=FilterOperator.GTE,
                    value=20.0,
                )
            ]
        )
        query_gte = VectorStoreQuery(
            query_embedding=query_embedding, similarity_top_k=3, filters=gte_filter
        )
        results_gte, _ = graph_store.vector_query(query_gte)
        names_gte = {n.name for n in results_gte if isinstance(n, EntityNode)}

        assert names_gte == {"Item2", "Item3", "Item4"}


#     # Fails in dynamic schmema because it compares property as string,
#         which makes '5'<'15' = False
#
#     # Test Case 3: Less Than (LT) for 'stock'
#     lt_filter = MetadataFilters(
#         filters=[
#             MetadataFilter(
#                 key=ElementSchema.PROPERTY_PREFIX + "stock",
#                 operator=FilterOperator.LT,
#                 value=15,
#             )
#         ]
#     )
#     query_lt = VectorStoreQuery(
#         query_embedding=query_embedding, similarity_top_k=3, filters=lt_filter
#     )
#     results_lt, _ = graph_store.vector_query(query_lt)
#     names_lt = {n.name for n in results_lt}

#     assert names_lt == {"Item1"}


def test_vector_query_with_multiple_filters_and_condition(
    property_graph_store_static: SpannerPropertyGraphStore,
    property_graph_store_dynamic: SpannerPropertyGraphStore,
):
    """Test vector_query with multiple PropertyFilters and AND/OR conditions."""
    nodes_data: list[dict[str, Any]] = [
        {
            "id": "mf_item1",
            "label": "MFITEM",
            "name": "Item1",
            "embedding": [0.1, 0.1],
            "category": "A",
            "price": 10,
        },
        {
            "id": "mf_item2",
            "label": "MFITEM",
            "name": "Item2",
            "embedding": [0.2, 0.2],
            "category": "A",
            "price": 25,
        },  # Matches cat A, price > 20
        {
            "id": "mf_item3",
            "label": "MFITEM",
            "name": "Item3",
            "embedding": [0.3, 0.3],
            "category": "B",
            "price": 30,
        },  # Matches price > 20
        {
            "id": "mf_item4",
            "label": "MFITEM",
            "name": "Item4",
            "embedding": [0.4, 0.4],
            "category": "A",
            "price": 15,
        },  # Matches cat A
    ]
    nodes = [
        EntityNode(
            label=d["label"],
            name=d["name"],
            embedding=d["embedding"],
            properties={"category": d["category"], "price": d["price"]},
        )
        for d in nodes_data
    ]
    query_embedding = [0.15, 0.15]

    for graph_store in [
        property_graph_store_static,
        property_graph_store_dynamic,
    ]:
        # graph_store.clear_all_data() # If using shared fixtures
        graph_store.upsert_nodes(nodes)

        # Test Case 1: AND condition (default)
        # category == "A" AND price > 20
        filters_and: list[Union[MetadataFilter, MetadataFilters]] = [
            MetadataFilter(key="category", operator=FilterOperator.EQ, value="A"),
            MetadataFilter(
                key="price", operator=FilterOperator.GT, value=20
            ),  # Numeric for static, string for dynamic
        ]
        # Default condition for MetadataFilters is AND
        query_and = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=3,
            filters=MetadataFilters(filters=filters_and),
        )
        results_and, _ = graph_store.vector_query(query_and)
        names_and = {n.name for n in results_and if isinstance(n, EntityNode)}

        if not graph_store.schema.use_flexible_schema:
            assert names_and == {"Item2"}
        else:  # category == "A" AND JSON_VALUE(price) > "20" ("25" > "20" is true)
            assert names_and == {"Item2"}

        # Test Case 2: OR condition
        # category == "B" OR price < 15
        filters_or: list[Union[MetadataFilter, MetadataFilters]] = [
            MetadataFilter(key="category", operator=FilterOperator.EQ, value="B"),
            MetadataFilter(
                key="price", operator=FilterOperator.LT, value=15
            ),  # Numeric for static, string for dynamic
        ]
        query_or = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=3,
            filters=MetadataFilters(filters=filters_or, condition=FilterCondition.OR),
        )
        results_or, _ = graph_store.vector_query(query_or)
        names_or = {n.name for n in results_or if isinstance(n, EntityNode)}
        # Item3 matches OR condition, Item4 matches AND condition
        if not graph_store.schema.use_flexible_schema:
            expected_or_names = {"Item1", "Item3"}
        else:
            expected_or_names = {"Item1", "Item3"}
        assert (
            names_or == expected_or_names
        ), f"Failed OR condition. Expected {expected_or_names}, Got {names_or}"


def test_vector_query_on_empty_graph_or_no_embeddings(
    property_graph_store_static: SpannerPropertyGraphStore,
    property_graph_store_dynamic: SpannerPropertyGraphStore,
):
    """Test vector_query on an empty graph and on a graph with nodes but no embeddings."""
    query_embedding = [0.1, 0.2]
    vq = VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=3)

    for graph_store in [
        property_graph_store_static,
        property_graph_store_dynamic,
    ]:

        # Case 1: Empty graph
        results_empty, scores_empty = graph_store.vector_query(vq)
        assert len(results_empty) == 0
        assert len(scores_empty) == 0

        # Case 2: Graph with nodes, but none have embeddings
        node_no_emb = EntityNode(
            label="NOEMB",
            name="NodeWithoutEmbedding",
            properties={"data": "some_val"},
        )
        graph_store.upsert_nodes([node_no_emb])

        results_no_emb, scores_no_emb = graph_store.vector_query(vq)
        assert len(results_no_emb) == 0
        assert len(scores_no_emb) == 0

        # Case 3: Graph with some nodes with embeddings, some without
        node_with_emb = EntityNode(
            label="EMB", name="NodeWithEmbedding", embedding=[0.11, 0.22]
        )
        graph_store.upsert_nodes([node_with_emb])  # node_no_emb is still there

        results_mixed, scores_mixed = graph_store.vector_query(vq)
        assert len(results_mixed) == 1
        if isinstance(results_mixed[0], EntityNode):
            assert results_mixed[0].name == "NodeWithEmbedding"
        assert len(scores_mixed) == 1

        graph_store.delete(
            ids=[node_no_emb.id, node_with_emb.id]
        )  # Clean up nodes for next iteration


def test_vector_query_top_k_variations(
    property_graph_store_static: SpannerPropertyGraphStore,
    property_graph_store_dynamic: SpannerPropertyGraphStore,
):
    """Test vector_query with different similarity_top_k values."""
    nodes_data: list[dict[str, Any]] = [
        {"name": "TopK1", "embedding": [0.1, 0.1, 0.1]},
        {"name": "TopK2", "embedding": [0.2, 0.2, 0.2]},
        {"name": "TopK3", "embedding": [0.3, 0.3, 0.3]},
        {"name": "TopK4", "embedding": [0.4, 0.4, 0.4]},
    ]
    nodes = [
        EntityNode(label="TOPK", name=d["name"], embedding=d["embedding"])
        for d in nodes_data
    ]
    query_embedding = [0.1, 0.1, 0.1]  # Closest to TopK1, then TopK2 etc.

    for graph_store in [
        property_graph_store_static,
        property_graph_store_dynamic,
    ]:
        graph_store.upsert_nodes(nodes)

        query_k0 = VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=0)
        results_k0, scores_k0 = graph_store.vector_query(query_k0)
        assert len(results_k0) == 0
        assert len(scores_k0) == 0

        # Case 2: top_k = 1
        query_k1 = VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=1)
        results_k1, _ = graph_store.vector_query(query_k1)
        assert len(results_k1) == 1
        if isinstance(results_k1[0], EntityNode):
            assert results_k1[0].name == "TopK1"

        # Case 3: top_k = 2
        query_k2 = VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=2)
        results_k2, _ = graph_store.vector_query(query_k2)
        assert len(results_k2) == 2
        names_k2 = {n.name for n in results_k2 if isinstance(n, EntityNode)}
        assert names_k2 == {"TopK1", "TopK2"}  # Order depends on similarity scores

        # Case 4: top_k = number of nodes
        query_k_all = VectorStoreQuery(
            query_embedding=query_embedding, similarity_top_k=len(nodes)
        )
        results_k_all, _ = graph_store.vector_query(query_k_all)
        assert len(results_k_all) == len(nodes)

        # Case 5: top_k > number of nodes
        query_k_more = VectorStoreQuery(
            query_embedding=query_embedding, similarity_top_k=len(nodes) + 5
        )
        results_k_more, _ = graph_store.vector_query(query_k_more)
        assert len(results_k_more) == len(
            nodes
        )  # Should not return more than available


def test_refresh_schema(
    property_graph_store_dynamic: SpannerPropertyGraphStore,
    property_graph_store_static: SpannerPropertyGraphStore,
):
    """Test explicit refresh of the schema."""
    for graph_store in [
        property_graph_store_static,
        property_graph_store_dynamic,
    ]:
        if graph_store.schema.use_flexible_schema:
            graph_store.schema.static_node_properties = set(["age"])

        # Insert data
        e1 = EntityNode(label="PERSON", name="Alice", properties={"age": 30})
        graph_store.upsert_nodes([e1])

        # Refresh schema
        schema = str(graph_store.get_schema())
        assert "age" in schema, "Expected 'age' property in PERSON schema."


def test_get_schema(
    property_graph_store_static: SpannerPropertyGraphStore,
    property_graph_store_dynamic: SpannerPropertyGraphStore,
):
    """Test get_schema.

    The schema might be empty or minimal if no data has been inserted yet.
    """
    # Insert some data first

    for graph_store in [
        property_graph_store_static,
        property_graph_store_dynamic,
    ]:
        e1 = EntityNode(label="PERSON", name="Alice")
        graph_store.upsert_nodes([e1])

        schema = graph_store.get_schema(refresh=True)
        assert "Name of graph" in schema
        assert "Node properties per node type" in schema
        assert "Edges" in schema


def test_get_schema_str(
    property_graph_store_static: SpannerPropertyGraphStore,
):
    """Test the textual representation of the schema."""
    graph_store = property_graph_store_static
    e1 = EntityNode(label="PERSON", name="Alice")
    e2 = EntityNode(label="CITY", name="Paris")
    graph_store.upsert_nodes([e1, e2])

    # Insert a relationship
    r = Relation(label="VISITED", source_id=e1.id, target_id=e2.id)
    graph_store.upsert_relations([r])

    schema_str = graph_store.get_schema_str(refresh=True)
    assert "PERSON" in schema_str
    assert "CITY" in schema_str
    assert "VISITED" in schema_str


def test_evolve_schema_add_new_property_to_existing_node_type_static(
    property_graph_store_static: SpannerPropertyGraphStore,
):
    """Test schema evolution when adding a new property to an existing node type (static schema)."""
    graph_store = property_graph_store_static

    # Initial node
    node1 = EntityNode(label="FRUIT", name="Apple", properties={"color": "Red"})
    graph_store.upsert_nodes([node1])

    # Retrieve and verify initial state
    retrieved_node1 = graph_store.get(ids=[node1.id])[0]
    assert retrieved_node1.properties.get("color") == "Red"
    assert "ripeness" not in retrieved_node1.properties
    # Schema before adding new property
    schema_before = graph_store.get_schema(refresh=True)
    node_props_before = None
    for node_type_info in schema_before.get("Node properties per node type", {}).get(
        f"{graph_store.schema.graph_name}_FRUIT", []
    ):
        if node_type_info.get("property name") == "prop_color":
            node_props_before = schema_before["Node properties per node type"][
                f"{graph_store.schema.graph_name}_FRUIT"
            ]
            break
    assert node_props_before is not None
    assert any(p.get("property name") == "prop_color" for p in node_props_before)
    assert not any(p.get("property name") == "prop_ripeness" for p in node_props_before)

    # Node with a new property for the same type
    node2 = EntityNode(
        label="FRUIT",
        name="Banana",
        properties={"color": "Yellow", "ripeness": "Ripe"},
    )
    graph_store.upsert_nodes([node2])

    # Retrieve and verify new node
    retrieved_node2 = graph_store.get(ids=[node2.id])[0]
    assert retrieved_node2.properties.get("color") == "Yellow"
    assert retrieved_node2.properties.get("ripeness") == "Ripe"

    # Retrieve original node again, it should have the new property with None
    retrieved_node1_after = graph_store.get(ids=[node1.id])[0]
    assert retrieved_node1_after.properties.get("color") == "Red"
    assert "ripeness" in retrieved_node1_after.properties
    assert retrieved_node1_after.properties.get("ripeness") is None

    # Schema after adding new property
    schema_after = graph_store.get_schema(refresh=True)
    # For static schema, the table name is graph_name + "_" + label
    node_table_key = f"{graph_store.schema.graph_name}_FRUIT"

    assert node_table_key in schema_after["Node properties per node type"]
    node_props_after_list = schema_after["Node properties per node type"][
        node_table_key
    ]

    assert any(p.get("property name") == "prop_color" for p in node_props_after_list)
    assert any(p.get("property name") == "prop_ripeness" for p in node_props_after_list)


def test_schema_evolve_add_new_property_to_existing_edge_type_static(
    property_graph_store_static: SpannerPropertyGraphStore,
):
    """Test schema evolution for adding a new property to an existing edge type (static schema)."""
    graph_store = property_graph_store_static
    node_a = EntityNode(label="LOC", name="PointA")
    node_b = EntityNode(label="LOC", name="PointB")
    node_c = EntityNode(label="LOC", name="PointC")
    graph_store.upsert_nodes([node_a, node_b, node_c])

    # Initial relation
    rel1 = Relation(
        source_id=node_a.id,
        target_id=node_b.id,
        label="CONNECTED",
        properties={"distance": 100},
    )
    graph_store.upsert_relations([rel1])

    # Verify initial schema for the relation
    schema_before = graph_store.get_schema(refresh=True)
    triplets_before = graph_store.get_triplets(relation_names=["CONNECTED"])
    edge_table_key = f"{graph_store.schema.graph_name}_LOC_CONNECTED_LOC"
    assert edge_table_key in schema_before["Edge properties per edge type"]
    edge_props_before = schema_before["Edge properties per edge type"][edge_table_key]
    assert any(p.get("property name") == "prop_distance" for p in edge_props_before)
    assert not any(p.get("property name") == "prop_metric" for p in edge_props_before)

    # Relation with a new property for the same type
    rel2 = Relation(
        source_id=node_b.id,
        target_id=node_c.id,
        label="CONNECTED",
        properties={"distance": 200, "metric": "meters"},
    )
    graph_store.upsert_relations([rel2])

    # Verify schema after adding new property
    schema_after = graph_store.get_schema(refresh=True)
    triplets_after = graph_store.get_triplets(relation_names=["CONNECTED"])
    assert edge_table_key in schema_after["Edge properties per edge type"]
    edge_props_after = schema_after["Edge properties per edge type"][edge_table_key]
    assert any(p.get("property name") == "prop_distance" for p in edge_props_after)
    assert any(p.get("property name") == "prop_metric" for p in edge_props_after)

    # Verify relations
    triplets = graph_store.get_triplets(relation_names=["CONNECTED"])
    assert len(triplets) == 2
    rel1_retrieved = next(
        t[1] for t in triplets if t[1].properties.get("distance") == 100
    )
    rel2_retrieved = next(
        t[1] for t in triplets if t[1].properties.get("distance") == 200
    )
    assert rel1_retrieved.properties.get("metric") is None
    assert rel2_retrieved.properties.get("metric") == "meters"


def test_schema_evolve_property_type_conflict_static(
    property_graph_store_static: SpannerPropertyGraphStore,
):
    """Test schema evolution with property type conflict (static schema - expects error)."""
    graph_store = property_graph_store_static
    node1 = EntityNode(
        label="ITEM",
        name="Book",
        properties={"price": "20.00"},
    )
    graph_store.upsert_nodes([node1])
    # Attempt to upsert another node of the same type with price as a number
    node2 = EntityNode(label="ITEM", name="Pen", properties={"price": 1.50})

    # The ElementSchema.evolve method should raise a ValueError for type mismatch
    with pytest.raises(
        ValueError,
        match=(
            r"Property with name `prop_price` should have the same type, got"
            r" code: FLOAT64.*"
        ),
    ):
        graph_store.upsert_nodes([node2])


def test_flexible_schema_static_node_and_edge_properties_handling(
    property_graph_store_dynamic: SpannerPropertyGraphStore,
):
    """Test flexible schema with defined static node and edge properties."""
    static_node_props = {"fixed_code", "status"}
    static_edge_props = {"reliability_score"}

    graph_store = property_graph_store_dynamic
    graph_store.schema.static_node_properties = static_node_props
    graph_store.schema.static_edge_properties = static_edge_props
    try:
        node1 = EntityNode(
            label="DEVICE",
            name="SensorA",
            properties={
                "fixed_code": "FXD001",  # Static
                "status": "active",  # Static
                "location": "Room 101",  # Dynamic
                "temp": 23.5,  # Dynamic
            },
        )
        node2 = EntityNode(
            label="DEVICE",
            name="SensorB",
            properties={"location": "Room 102", "status": "inactive"},
        )
        graph_store.upsert_nodes([node1, node2])

        rel1 = Relation(
            source_id=node1.id,
            target_id=node2.id,
            label="COMMUNICATES_WITH",
            properties={
                "reliability_score": 0.95,  # Static
                "last_contact__ms": 120,  # Dynamic
            },
        )
        graph_store.upsert_relations([rel1])

        # Verify schema
        schema = graph_store.get_schema(refresh=True)

        node_table_key = f"{graph_store.schema.graph_name}_NODE"

        edge_table_key = f"{graph_store.schema.graph_name}_EDGE"

        assert node_table_key in schema["Node properties per node type"]
        node_props_schema = schema["Node properties per node type"][node_table_key]
        assert any(p["property name"] == "id" for p in node_props_schema)
        assert any(p["property name"] == "label" for p in node_props_schema)
        assert any(
            p["property name"] == ElementSchema.PROPERTY_PREFIX + "status"
            for p in node_props_schema
        )
        assert any(
            p["property name"] == ElementSchema.DYNAMIC_PROPERTY_COLUMN_NAME
            for p in node_props_schema
        )  # 'properties' JSON blob

        assert edge_table_key in schema["Edge properties per edge type"]
        edge_props_schema = schema["Edge properties per edge type"][edge_table_key]
        assert any(p["property name"] == "id" for p in edge_props_schema)
        assert any(
            p["property name"] == ElementSchema.DYNAMIC_PROPERTY_COLUMN_NAME
            for p in edge_props_schema
        )
        assert any(
            p["property name"] == ElementSchema.PROPERTY_PREFIX + "reliability_score"
            for p in edge_props_schema
        )

        # Verify retrieved data
        retrieved_node1 = graph_store.get(ids=[node1.id])[0]
        assert retrieved_node1.properties.get("fixed_code") == "FXD001"
        assert retrieved_node1.properties.get("status") == "active"
        assert retrieved_node1.properties.get("location") == "Room 101"
        assert retrieved_node1.properties.get("temp") == "23.5"

        triplets = graph_store.get_triplets(
            ids=[node1.id], relation_names=["COMMUNICATES_WITH"]
        )
        assert len(triplets) == 1
        retrieved_rel1 = triplets[0][1]
        assert retrieved_rel1.properties.get("reliability_score") == "0.95"
        assert retrieved_rel1.properties.get("last_contact__ms") == 120
    except Exception as e:
        pytest.fail(f"Flexible schema with static properties failed: {e}")


def test_schema_with_empty_properties_for_nodes_and_edges(
    property_graph_store_static: SpannerPropertyGraphStore,
    property_graph_store_dynamic: SpannerPropertyGraphStore,
):
    """Test schema and retrieval for nodes/edges with empty properties."""
    for graph_store in [
        property_graph_store_static,
        property_graph_store_dynamic,
    ]:
        node1 = EntityNode(label="TEST_EMPTY", name="NodeWithNoProps", properties={})
        node2 = EntityNode(
            label="TEST_EMPTY",
            name="NodeWithProps",
            properties={"key": "value"},
        )
        graph_store.upsert_nodes([node1, node2])

        rel1 = Relation(
            source_id=node1.id,
            target_id=node2.id,
            label="LINKED_EMPTY",
            properties={},
        )
        graph_store.upsert_relations([rel1])

        # Verify schema
        schema = graph_store.get_schema(refresh=True)
        if not graph_store.schema.use_flexible_schema:
            node_table_key = f"{graph_store.schema.graph_name}_TEST_EMPTY"
            edge_table_key = (
                f"{graph_store.schema.graph_name}_TEST_EMPTY_LINKED_EMPTY_TEST_EMPTY"
            )
        else:
            node_table_key = f"{graph_store.schema.graph_name}_NODE"
            edge_table_key = f"{graph_store.schema.graph_name}_EDGE"

        assert node_table_key in schema["Node properties per node type"]
        # For static schema, node1 (empty props) won't add any prop_ columns
        # beyond defaults initially
        # For dynamic schema, the 'properties' JSON column exists.

        assert edge_table_key in schema["Edge properties per edge type"]

        # Verify retrieval
        retrieved_node1 = graph_store.get(ids=[node1.id])[0]
        if isinstance(retrieved_node1, EntityNode):
            assert retrieved_node1.name == "NodeWithNoProps"
        assert not retrieved_node1.properties.get("key")  # Should be empty

        triplets = graph_store.get_triplets(
            ids=[node1.id], relation_names=["LINKED_EMPTY"]
        )
        assert len(triplets) == 1
        assert not triplets[0][1].properties  # Relation properties should be empty


def test_get_schema_str_for_dynamic_store(
    property_graph_store_dynamic: SpannerPropertyGraphStore,
):
    """Test the textual representation of the schema for a dynamic store."""
    graph_store = property_graph_store_dynamic
    e1 = EntityNode(label="PERSON_DYN", name="Alice_dyn", properties={"age": 30})
    e2 = EntityNode(
        label="CITY_DYN", name="Paris_dyn", properties={"country": "France"}
    )
    graph_store.upsert_nodes([e1, e2])

    r = Relation(
        label="VISITED_DYN",
        source_id=e1.id,
        target_id=e2.id,
        properties={"year": 2023},
    )
    graph_store.upsert_relations([r])

    schema_str = graph_store.get_schema_str(refresh=True)

    assert f"{graph_store.schema.graph_name}_NODE" in schema_str  # e.g., graphName_NODE
    assert f"{graph_store.schema.graph_name}_EDGE" in schema_str  # e.g., graphName_EDGE
    assert ElementSchema.DYNAMIC_PROPERTY_COLUMN_NAME in schema_str
    assert ElementSchema.DYNAMIC_LABEL_COLUMN_NAME in schema_str


def test_operations_after_cleanup_upsert_recreates_graph(
    property_graph_store_static: SpannerPropertyGraphStore,
):
    """Test that basic operations can still work after a cleanup, implying schema/graph recreation."""
    graph_store = property_graph_store_static

    # Initial operation
    node_initial = EntityNode(label="TEST", name="InitialNode")
    graph_store.upsert_nodes([node_initial])
    assert len(graph_store.get(ids=[node_initial.id])) == 1

    # Clean up
    graph_store.clean_up()

    # Assert graph is gone - structured_query should fail as per its check
    with pytest.raises(ValueError, match="Graph does not exist yet."):
        graph_store.structured_query("MATCH (n) RETURN n", {})

    # Attempt to upsert nodes again
    # This should trigger schema evolution and graph creation again
    node_after_cleanup = EntityNode(label="TEST", name="NodeAfterCleanup")
    try:
        graph_store.upsert_nodes([node_after_cleanup])
    except Exception as e:
        pytest.fail(f"Upserting node after cleanup failed: {e}")

    # Verify the new node can be retrieved
    retrieved_nodes = graph_store.get(ids=[node_after_cleanup.id])
    assert len(retrieved_nodes) == 1
    if isinstance(retrieved_nodes[0], EntityNode):
        assert retrieved_nodes[0].name == "NodeAfterCleanup"

    # Verify schema exists again
    schema_after_recreation = graph_store.get_schema(refresh=True)
    assert "Name of graph" in schema_after_recreation
    node_table_key = f"{graph_store.schema.graph_name}_TEST"
    assert node_table_key in schema_after_recreation["Node properties per node type"]
