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

import itertools
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from google.cloud import spanner  # type: ignore
from llama_index.core.graph_stores.types import (
    ChunkNode,
    EntityNode,
    LabelledNode,
    PropertyGraphStore,
    Relation,
    Triplet,
)
from llama_index.core.prompts import PromptTemplate, PromptType
from llama_index.core.vector_stores.types import MetadataFilter, VectorStoreQuery

from .prompts import DEFAULT_SPANNER_GQL_TEMPLATE
from .schema import (
    ElementSchema,
    GraphDocumentUtility,
    SpannerGraphSchema,
    group_edges,
    group_nodes,
)
from .spanner import SpannerImpl, SpannerInterface, client_with_user_agent
from .version import __version__


def remove_property_prefix(property_name: str) -> str:
    property_prefix = ElementSchema.PROPERTY_PREFIX
    return property_name[
        property_name.startswith(property_prefix) and len(property_prefix) :
    ]


def node_from_json(label: str, json_node_properties: Dict[str, Any]) -> LabelledNode:
    """Converts a node from JSON to a LabelledNode.

    Args:
      label: The label of the node.
      json_node_properties: A dictionary of node properties in JSON format.

    Returns:
      A LabelledNode.
    """
    id_, name, text, properties, embedding = "", "", None, {}, None
    for k, v in json_node_properties.items():
        if k == ElementSchema.NODE_KEY_COLUMN_NAME:
            id_ = v
        elif k == ElementSchema.NODE_EMBEDDING_COLUMN_NAME:
            embedding = v
        elif k == ElementSchema.ENTITY_NODE_NAME_COLUMN_NAME:
            name = v
        elif k == ElementSchema.CHUNK_NODE_TEXT_COLUMN_NAME:
            text = v
        elif k == ElementSchema.DYNAMIC_PROPERTY_COLUMN_NAME:
            properties.update(v)
        else:
            properties[remove_property_prefix(k)] = v
    if text is not None:
        return ChunkNode(
            id_=id_,
            text=text,
            label=label,
            properties=properties,
            embedding=embedding,
        )
    return EntityNode(
        name=name, label=label, properties=properties, embedding=embedding
    )


def edge_from_json(json_edge_properties: Dict[str, Any]) -> Relation:
    """Converts an edge from JSON to a Relation.

    Args:
      json_edge_properties: A dictionary of edge properties in JSON format.

    Returns:
      A Relation.
    """
    source_id, target_id, properties, label = "", "", {}, ""
    for k, v in json_edge_properties.items():
        if k == ElementSchema.NODE_KEY_COLUMN_NAME:
            source_id = v
        elif k == ElementSchema.TARGET_NODE_KEY_COLUMN_NAME:
            target_id = v
        elif k == ElementSchema.DYNAMIC_PROPERTY_COLUMN_NAME:
            properties.update(v)
        elif k == ElementSchema.DYNAMIC_LABEL_COLUMN_NAME:
            label = v
        else:
            properties[remove_property_prefix(k)] = v
    return Relation(
        source_id=source_id,
        target_id=target_id,
        label=label,
        properties=properties,
    )


def update_condition(
    cond: List[str],
    params: Dict[str, Any],
    schema: SpannerGraphSchema,
    ids: Optional[List[str]] = None,
    properties: Optional[Dict[str, Any]] = None,
    entity_names: Optional[List[str]] = None,
) -> bool:
    """Updates the condition and params based on the schema and other args.

    Args:
      cond: The condition to update.
      params: The params to update.
      schema: The schema to use.
      ids: The ids to filter by.
      properties: The properties to filter by.
      entity_names: The entity names to filter by.

    Returns:
      True if the condition got updated without errors, False otherwise.
    """
    property_field = ElementSchema.DYNAMIC_PROPERTY_COLUMN_NAME
    node_key_field = ElementSchema.NODE_KEY_COLUMN_NAME
    entity_name_field = ElementSchema.ENTITY_NODE_NAME_COLUMN_NAME
    property_prefix = ElementSchema.PROPERTY_PREFIX

    if properties:
        for i, prop in enumerate(properties):
            prefixed_prop = property_prefix + prop
            if (
                prefixed_prop not in schema.node_properties
                and not schema.use_flexible_schema
            ):
                return False
            inner_cond = []
            if prefixed_prop in schema.node_properties:
                inner_cond.append(
                    f"(PROPERTY_EXISTS(n, {prefixed_prop}) AND"
                    f" n.{prefixed_prop} = @property_{i})"
                )
                params[f"property_{i}"] = properties[prop]
            if schema.use_flexible_schema:
                inner_cond.append(
                    f"JSON_VALUE(n.{property_field}.{prop}) = @property_str_{i}"
                )
                params[f"property_str_{i}"] = str(properties[prop])
            cond.append(
                f"""(
            {" OR ".join(inner_cond)}
          )"""
            )

    if ids:
        cond.append(f"""n.{node_key_field} IN UNNEST(@ids)""")
        params["ids"] = ids

    if entity_names:
        if entity_name_field not in schema.node_properties:
            return False
        cond.append(
            f"""PROPERTY_EXISTS(n, {entity_name_field}) AND n.{entity_name_field} IN UNNEST(@entity_names)"""
        )
        params["entity_names"] = entity_names

    return True


def convert_operator(operator: str) -> str:
    mapping = {"==": "=", "!=": "<>", "nin": "in"}
    return mapping.get(operator, operator)


USER_AGENT_GRAPH_STORE = "llama-index-spanner-python:graphstore/" + __version__


class SpannerPropertyGraphStore(PropertyGraphStore):
    """Spanner Property Graph Store."""

    supports_structured_queries: bool = True
    supports_vector_queries: bool = True
    text_to_cypher_template: PromptTemplate = PromptTemplate(
        DEFAULT_SPANNER_GQL_TEMPLATE,
        prompt_type=PromptType.TEXT_TO_GRAPH_QUERY,
    )

    def __init__(
        self,
        instance_id: str,
        database_id: str,
        graph_name: str,
        clean_up: bool = False,
        client: Optional[spanner.Client] = None,
        impl: Optional[SpannerInterface] = None,
        use_flexible_schema: bool = False,
        static_node_properties: Optional[List[str]] = None,
        static_edge_properties: Optional[List[str]] = None,
        ignore_invalid_relations: bool = True,
    ) -> None:
        """Initializes the SpannerPropertyGraphStore.

        Args:
            instance_id: The ID of the Spanner instance to use.
            database_id: The ID of the Spanner database to use.
            graph_name: The name of the graph to use.
            clean_up: Whether to clean up the graph before initializing.
            client: An optional Spanner client to use. If not provided, a new client
              will be created.
            impl: An optional Spanner implementation to use. If not provided, a new
              Spanner implementation will be created.
            use_flexible_schema: Whether to use a flexible schema.
            static_node_properties: A list of node properties that are not
              dynamically added.
            static_edge_properties: A list of edge properties that are not
              dynamically added.
            ignore_invalid_relations: Whether to ignore invalid relations. If True,
              relations with unknown source or target ids will be ignored instead of
              raising an error.
        """
        self.impl = impl or SpannerImpl(
            instance_id,
            database_id,
            client_with_user_agent(client, USER_AGENT_GRAPH_STORE),
        )
        self.ignore_invalid_relations = ignore_invalid_relations
        self.schema = SpannerGraphSchema(
            graph_name,
            use_flexible_schema,
            static_node_properties,
            static_edge_properties,
        )
        self.refresh_schema()
        if clean_up:
            self.clean_up()

    @property
    def client(self):
        return self.impl

    def upsert_nodes(self, nodes: Sequence[LabelledNode]) -> None:
        """Upserts nodes into the graph store.

        This method takes a list of LabelledNodes and upserts them into the
        Spanner graph. It first groups the nodes by their type, then evolves
        the schema if necessary, and finally inserts or updates the nodes in
        the corresponding tables.

        Args:
            nodes: A list of LabelledNodes to upsert.
        """
        nodes_group = group_nodes(nodes)
        ddls, node_schema_mapping = self.schema.evolve_from_nodes(nodes_group)
        if ddls:
            self.impl.apply_ddls(ddls)
            self.refresh_schema()
        else:
            print("No schema change required...")

        for name, elements in nodes_group.items():
            for table, columns, rows in node_schema_mapping[name].add_nodes(elements):
                print(f"Insert nodes of type `{name}`...")
                self.impl.insert_or_update(table, columns, rows)

    def upsert_relations(self, relations: List[Relation]) -> None:
        """Upserts relations into the graph store.

        This method takes a list of Relations and upserts them into the
        Spanner graph. It first groups the relations by their type, then evolves
        the schema if necessary, and finally inserts or updates the relations in
        the corresponding tables.

        Args:
            relations: A list of Relations to upsert.

        Raises:
            ValueError: If any of the relations have unknown source or target ids if
            ignore_invalid_relations is False.
        """
        ids = list(
            set([r.source_id for r in relations] + [r.target_id for r in relations])
        )
        label_expr = (
            f"n.{ElementSchema.DYNAMIC_LABEL_COLUMN_NAME}"
            if self.schema.use_flexible_schema
            else "labels(n)[0]"
        )
        data = self.structured_query(
            f"""MATCH (n)
          WHERE n.id in UNNEST(@ids)
          RETURN n.id AS id,
          {label_expr} AS type,
          labels(n)[0] AS node_table_label
        """,
            param_map={"ids": ids},
        )
        node_mapping = {
            record["id"]: (
                record["type"],
                self.schema.labels[record["node_table_label"]].base_table_name,
            )
            for record in data
        }

        relation_data = []
        for r in relations:
            if r.source_id not in node_mapping or r.target_id not in node_mapping:
                if self.ignore_invalid_relations:
                    continue
                else:
                    raise ValueError(
                        "Some relations have unknown source or target ids."
                    )
            relation_data.append(
                (
                    r,
                    node_mapping[r.source_id][0],
                    node_mapping[r.source_id][1],
                    node_mapping[r.target_id][0],
                    node_mapping[r.target_id][1],
                )
            )

        edges_group = group_edges(relation_data)
        ddls, edge_schema_mapping = self.schema.evolve_from_edges(edges_group)
        if ddls:
            self.impl.apply_ddls(ddls)
            self.refresh_schema()
        else:
            print("No schema change required...")

        for name, elements in edges_group.items():
            for table, columns, rows in edge_schema_mapping[name].add_edges(
                [ele for ele, _, _ in elements]
            ):
                print(f"Insert edges of type `{name}`...")
                self.impl.insert_or_update(table, columns, rows)

    def get(
        self,
        properties: Optional[dict[str, Any]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[LabelledNode]:
        """Get nodes from the graph store.

        Args:
            properties: A dictionary of properties to filter by.
            ids: A list of ids to filter by.

        Returns:
            A list of LabelledNodes.
        """
        if not self.schema.graph_exists:
            return []

        label_expr = (
            f"n.{ElementSchema.DYNAMIC_LABEL_COLUMN_NAME}"
            if self.schema.use_flexible_schema
            else "labels(n)[0]"
        )
        cond = ["1 = 1"]
        params: Dict[str, Any] = {}

        if not update_condition(
            cond, params, self.schema, ids=ids, properties=properties
        ):
            return []

        data = self.structured_query(
            f"""MATCH (n)
          WHERE {" AND ".join(cond)}
          RETURN {label_expr} AS type,
          to_json(n).properties as json_node_properties
        """,
            param_map=params,
        )
        nodes = []
        for record in data:
            nodes.append(node_from_json(record["type"], record["json_node_properties"]))
        return nodes

    def get_triplets(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        properties: Optional[Dict[str, Any]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[Triplet]:
        """Get triplets from the graph store.

        Args:
            entity_names: A list of entity names to filter by.
            relation_names: A list of relation names to filter by.
            properties: A dictionary of properties to filter by.
            ids: A list of ids to filter by.

        Returns:
            A list of Triplets.
        """
        if not self.schema.graph_exists:
            return []

        label_field = ElementSchema.DYNAMIC_LABEL_COLUMN_NAME

        cond = ["1 = 1"]
        params: Dict[str, Any] = {}

        if not update_condition(
            cond,
            params,
            self.schema,
            entity_names=entity_names,
            ids=ids,
            properties=properties,
        ):
            return []

        if relation_names:
            cond.append(f"r.{label_field} IN UNNEST(@relation_names)")
            params["relation_names"] = relation_names

        if self.schema.use_flexible_schema:
            label_expr = f"n.{label_field}"
            target_label_expr = f"n2.{label_field}"
        else:
            label_expr = "labels(n)[0]"
            target_label_expr = "labels(n2)[0]"

        data = self.structured_query(
            f"""MATCH (n)-[r]->(n2)
          WHERE {" AND ".join(cond)}
          RETURN {label_expr} AS source_type,
          to_json(n).properties as source_json_node_properties,
          to_json(r).properties as json_edge_properties,
          {target_label_expr} AS target_type,
          to_json(n2).properties as target_json_node_properties
        """,
            param_map=params,
        )
        triples = []
        for record in data:
            source = node_from_json(
                record["source_type"], record["source_json_node_properties"]
            )
            target = node_from_json(
                record["target_type"], record["target_json_node_properties"]
            )
            rel = edge_from_json(record["json_edge_properties"])
            triples.append((source, rel, target))
        return triples

    def get_rel_map(
        self,
        graph_nodes: List[LabelledNode],
        depth: int = 2,
        limit: int = 30,
        ignore_rels: Optional[List[str]] = None,
    ) -> List[Triplet]:
        """Get depth-aware rel map from the graph store.

        Args:
          graph_nodes: A list of LabelledNodes to get the rel map for.
          depth: The depth of the rel map to get.
          limit: The limit of the rel map to get.
          ignore_rels: A list of relation names to ignore.

        Returns:
          A list of Triplets.
        """
        if not self.schema.graph_exists:
            return []

        if not graph_nodes:
            return []

        label_field = ElementSchema.DYNAMIC_LABEL_COLUMN_NAME
        ids = [node.id for node in graph_nodes]

        params = {"ids": ids, "limit": limit}
        edge_cond = ""

        if ignore_rels:
            edge_cond = f"WHERE r.{label_field} NOT IN UNNEST(@ignore_rels)"
            params["ignore_rels"] = ignore_rels

        data = self.structured_query(
            f"""MATCH p = TRAIL (n)-[r {edge_cond}]-{{1,{depth}}}()
          WHERE n.id IN UNNEST(@ids)
          RETURN to_json(p) as json_path
          LIMIT @limit
          """,
            param_map=params,
        )
        node_mapping, edges_mapping = {}, {}
        for record in data:
            path = json.loads((record["json_path"]).serialize())
            for ele in path:
                if ele["kind"] == "node":
                    node = node_from_json(
                        (
                            ele["properties"][label_field]
                            if label_field in ele["properties"]
                            else ele["labels"][0]
                        ),
                        ele["properties"],
                    )
                    node_mapping[node.id] = node
                elif ele["kind"] == "edge":
                    edge = edge_from_json(ele["properties"])
                    edges_mapping[(edge.source_id, edge.target_id, edge.label)] = edge
        triplet = []
        for edge in edges_mapping.values():
            source = node_mapping[edge.source_id]
            target = node_mapping[edge.target_id]
            triplet.append((source, edge, target))
        return triplet

    def delete(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        properties: Optional[dict[str, Any]] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """Delete matching data from the graph store.

        Args:
          entity_names: A list of entity names to delete.
          relation_names: A list of relation names to delete.
          properties: A dictionary of properties to delete.
          ids: A list of ids to delete.
        """
        if not self.schema.graph_exists:
            return

        label_field = ElementSchema.DYNAMIC_LABEL_COLUMN_NAME
        node_key_field = ElementSchema.NODE_KEY_COLUMN_NAME
        target_node_key_field = ElementSchema.TARGET_NODE_KEY_COLUMN_NAME

        cond: List[str] = []
        params: Dict[str, Any] = {}

        if (
            update_condition(
                cond,
                params,
                self.schema,
                entity_names=entity_names,
                ids=ids,
                properties=properties,
            )
            and cond
        ):
            data = self.structured_query(
                f"""MATCH (n) WHERE {" AND ".join(cond)}
          RETURN labels(n)[0] AS node_table_label,
          n.{node_key_field} AS id
          """,
                param_map=params,
            )
            to_delete_nodes = [
                (record["node_table_label"], record["id"]) for record in data
            ]
            if to_delete_nodes:
                for node_table_label, nodes in itertools.groupby(
                    to_delete_nodes, lambda x: x[0]
                ):
                    self.impl.delete(
                        self.schema.labels[node_table_label].base_table_name,
                        [[node_id] for _, node_id in nodes],
                    )

        if relation_names and self.schema.edge_tables:
            if self.schema.use_flexible_schema:
                data = self.structured_query(
                    f"""MATCH ()-[r]->()
              WHERE r.{label_field} IN UNNEST(@relation_names)
              return r.{node_key_field} AS id,
              r.{target_node_key_field} AS target_id,
              r.{label_field} AS label,
              labels(r)[0] AS edge_label
            """,
                    param_map={"relation_names": relation_names},
                )
                to_delete_edges = [
                    (
                        record["edge_label"],
                        record["id"],
                        record["target_id"],
                        record["label"],
                    )
                    for record in data
                ]
                if to_delete_edges:
                    for edge_label, edges in itertools.groupby(
                        to_delete_edges, lambda x: x[0]
                    ):
                        self.impl.delete(
                            self.schema.labels[edge_label].base_table_name,
                            [
                                [edge_id, edge_target_id, edge_label]
                                for _, edge_id, edge_target_id, edge_label in edges
                            ],
                        )
            else:
                data = self.structured_query(
                    f"""MATCH ()-[r]->()
              WHERE r.{label_field} IN UNNEST(@relation_names)
              return distinct labels(r)[0] AS edge_label
            """,
                    param_map={"relation_names": relation_names},
                )
                to_drop_edges = [record["edge_label"] for record in data]
                if to_drop_edges:
                    to_identifier = GraphDocumentUtility.to_identifier
                    drop_tables_ddls = []
                    for edge_label in to_drop_edges:
                        edge_name = self.schema.labels[edge_label].base_table_name
                        drop_tables_ddls.append(
                            f"DROP TABLE IF EXISTS {to_identifier(edge_name)}"
                        )
                        del self.schema.edge_tables[edge_name]
                    self.impl.apply_ddls([self.schema.to_ddl()])
                    self.impl.apply_ddls(drop_tables_ddls)

    def structured_query(
        self,
        query: str,
        param_map: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Query the graph store with statement and parameters.

        Args:
          query: The query to execute, in Cypher format.
          param_map: A dictionary of parameters to pass to the query.

        Returns:
          The result of the query, as a list of dictionaries.
        """
        if not self.schema.graph_exists:
            raise ValueError("Graph does not exist yet.")

        return self.impl.query(
            f"GRAPH `{self.schema.graph_name}` \n" + query,
            {} if param_map is None else param_map,
        )

    def vector_query(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> Tuple[List[LabelledNode], List[float]]:
        """Query the graph store with vector query.

        Args:
          query: The vector query to execute.
          **kwargs: Additional keyword arguments to pass to the query.

        Returns:
          A tuple of two lists:
          - A list of LabelledNodes, representing the nodes that were found.
          - A list of floats, representing the similarity scores of the nodes.
        """
        if not self.schema.graph_exists or query.query_embedding is None:
            return ([], [])

        query_condition = "1 = 1"
        params = {}
        property_field = ElementSchema.DYNAMIC_PROPERTY_COLUMN_NAME
        property_prefix = ElementSchema.PROPERTY_PREFIX

        if query.filters:
            cond = []
            for i, query_filter in enumerate(query.filters.filters):
                if not isinstance(
                    query_filter, MetadataFilter
                ):  # doesn't support nested MetadataFilters
                    continue

                if (
                    query_filter.key not in self.schema.node_properties
                    and (property_prefix + query_filter.key)
                    not in self.schema.node_properties
                    and not self.schema.use_flexible_schema
                ):
                    return ([], [])

                inner_cond = []
                prefix = "NOT" if query_filter.operator.value in ["nin"] else ""
                operator = convert_operator(query_filter.operator.value)

                if query_filter.key in self.schema.node_properties:
                    inner_cond.append(
                        f"(PROPERTY_EXISTS(n, {query_filter.key}) AND"
                        f" {prefix} n.{query_filter.key} {operator} @property_{i})"
                    )
                    params[f"property_{i}"] = query_filter.value
                elif (
                    property_prefix + query_filter.key
                ) in self.schema.node_properties:
                    inner_cond.append(
                        f"(PROPERTY_EXISTS(n, {property_prefix+query_filter.key}) AND"
                        f" {prefix} n.{property_prefix+query_filter.key} {operator}"
                        f" @property_{i})"
                    )
                    params[f"property_{i}"] = query_filter.value
                if (
                    self.schema.use_flexible_schema
                ):  # operator other than `=` won't work with flexible schema.
                    inner_cond.append(
                        f"({prefix} JSON_VALUE(n.{property_field}.{query_filter.key})"
                        f" {operator} @property_str_{i})"
                    )
                    params[f"property_str_{i}"] = str(query_filter.value)
                cond.append(
                    f"""(
              {" OR ".join(inner_cond)}
            )"""
                )
            if cond:
                operator = (
                    "and"
                    if query.filters.condition is None
                    else query.filters.condition.value
                )
                query_condition = f" {operator} ".join(cond)

        embedding_field = ElementSchema.NODE_EMBEDDING_COLUMN_NAME
        label_expr = (
            f"n.{ElementSchema.DYNAMIC_LABEL_COLUMN_NAME}"
            if self.schema.use_flexible_schema
            else "labels(n)[0]"
        )

        data = self.structured_query(
            f"""
        MATCH (n)
        WHERE 
            n.{embedding_field} IS NOT NULL 
            AND ARRAY_LENGTH(n.{embedding_field}) = @dimension 
            AND ({query_condition})
        WITH n, 
            1-COSINE_DISTANCE(n.{embedding_field}, ARRAY[{",".join(map(str, query.query_embedding))}]) AS score
        ORDER BY score DESC 
        LIMIT @limit
        RETURN 
            {label_expr} AS type,
            to_json(n).properties as json_node_properties,
            score
        """,
            param_map={
                "dimension": len(query.query_embedding),
                "limit": query.similarity_top_k,
                **params,
            },
        )
        data = data if data else []
        nodes = []
        scores = []
        for record in data:
            node = node_from_json(record["type"], record["json_node_properties"])
            nodes.append(node)
            scores.append(record["score"])

        return (nodes, scores)

    def refresh_schema(self) -> None:
        """Refreshes the Spanner Graph schema information.

        This method queries the Spanner information schema to retrieve the latest
        schema information for the graph. It then updates the internal schema
        representation.

        Raises:
            Exception: If the information schema query returns an unexpected number
            of rows.
        """
        results = self.impl.query(
            SpannerGraphSchema.GRAPH_INFORMATION_SCHEMA_QUERY_TEMPLATE.format(
                self.schema.graph_name
            )
        )
        if not results:
            return

        if len(results) != 1:
            raise Exception(
                f"Unexpected number of rows from information schema: {len(results)}"
            )

        self.schema.from_information_schema(results[0]["property_graph_metadata_json"])

    def get_schema(self, refresh: bool = False) -> Any:
        """Get the schema of the Spanner graph store.

        Args:
            refresh: Whether to refresh the schema information. If True, the schema
              will be refreshed from the database before returning. Defaults to
              False.

        Returns:
            A dictionary representing the schema of the graph.
            The dictionary will contain information about nodes and edges,
            including their properties and types.
        """
        if refresh:
            self.refresh_schema()
        return json.loads(repr(self.schema))

    def clean_up(self):
        """Removes all data from your Spanner Graph.

        USE IT WITH CAUTION!

        The graph, tables and the associated data will all be removed.
        """
        to_identifier = GraphDocumentUtility.to_identifier
        self.impl.apply_ddls(
            [f"DROP PROPERTY GRAPH IF EXISTS {to_identifier(self.schema.graph_name)}"]
        )
        self.impl.apply_ddls(
            [
                f"DROP TABLE IF EXISTS {to_identifier(edge.base_table_name)}"
                for edge in self.schema.edge_tables.values()
            ]
        )
        self.impl.apply_ddls(
            [
                f"DROP TABLE IF EXISTS {to_identifier(node.base_table_name)}"
                for node in self.schema.node_tables.values()
            ]
        )
        self.schema = SpannerGraphSchema(
            self.schema.graph_name, self.schema.use_flexible_schema
        )
