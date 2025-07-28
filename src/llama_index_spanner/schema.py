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

from __future__ import annotations

import json
import re
from typing import Any, Dict, Generator, Iterable, List, Optional, Sequence, Tuple

from google.cloud.spanner_v1 import JsonObject, param_types
from llama_index.core.graph_stores.types import (
    ChunkNode,
    EntityNode,
    LabelledNode,
    Relation,
)
from requests.structures import CaseInsensitiveDict

from .type_utils import TypeUtility

NODE_KIND = "NODE"
EDGE_KIND = "EDGE"


def remove_empty_values(input_dict):
    return {key: value for key, value in input_dict.items() if value is not None}


def group_nodes(
    nodes: Sequence[LabelledNode],
) -> Dict[str, List[LabelledNode]]:
    """Groups nodes by their respective types.

    Args:
      nodes: A list of LabelledNodes to group.

    Returns:
      A dictionary mapping node types to lists of LabelledNodes.
    """
    nodes_group: CaseInsensitiveDict[Dict[str, LabelledNode]] = CaseInsensitiveDict()
    for node in nodes:
        ns = nodes_group.setdefault(node.label, dict())
        nw = node.id
        if nw in ns:
            ns[nw].properties.update(node.properties)
        else:
            ns[nw] = node
        ns[nw].properties = remove_empty_values(ns[nw].properties)
    return {name: [n for _, n in ns.items()] for name, ns in nodes_group.items()}


def group_edges(
    edges: List[Tuple[Relation, str, str, str, str]],
) -> Dict[str, List[Tuple[Relation, str, str]]]:
    """Groups edges by their respective types.

    Args:
      edges: A list of tuples, where each tuple represents an edge and contains: -
        The Relation object. - The source node label. - The source node table
        name. - The target node label. - The target node table name.

    Returns:
      A dictionary mapping edge types to lists of tuples, each
      containing:
        - The Relation object.
        - The source node table name.
        - The target node table name.
    """
    edges_group: CaseInsensitiveDict[
        Dict[Tuple[str, str, str], Tuple[Relation, str, str]]
    ] = CaseInsensitiveDict()
    for edge, source_label, source_table, target_label, target_table in edges:
        edge_name = (
            "{}_{}_{}".format(source_label, edge.label, target_label)
            .replace(" ", "_")
            .replace("-", "_")
        )
        es = edges_group.setdefault(edge_name, dict())
        ew = (edge.source_id, edge.label, edge.target_id)
        if ew in es:
            es[ew][0].properties.update(edge.properties)
        else:
            es[ew] = (edge, source_table, target_table)
        es[ew][0].properties = remove_empty_values(es[ew][0].properties)
    return {name: [e for _, e in es.items()] for name, es in edges_group.items()}


class GraphDocumentUtility:
    """Utilities to process graph documents."""

    @staticmethod
    def is_valid_identifier(s: str) -> bool:
        return re.match(r"^[a-z][a-z0-9_]{0,127}$", s, re.IGNORECASE) is not None

    @staticmethod
    def to_identifier(s: str) -> str:
        return "`" + s + "`"

    @staticmethod
    def to_identifiers(s: List[str]) -> Iterable[str]:
        return map(GraphDocumentUtility.to_identifier, s)


class Label(object):
    """Schema representation of a label."""

    def __init__(self, name: str, base_table_name: str, prop_names: set[str]):
        self.name = name
        self.base_table_name = base_table_name
        self.prop_names = prop_names


class NodeReference(object):
    """Schema representation of a source or destination node reference."""

    def __init__(self, node_name: str, node_keys: List[str], edge_keys: List[str]):
        """Initializes a NodeReference.

        Args:
            node_name: The name of the referenced node table.
            node_keys: A list of column names that constitute the primary key of the
              referenced node table.
            edge_keys: A list of column names in the edge table that reference the
              node table's primary key.
        """
        self.node_name = node_name
        self.node_keys = node_keys
        self.edge_keys = edge_keys


class ElementSchema(object):
    """Schema representation of a node or an edge."""

    NODE_KEY_COLUMN_NAME: str = "id"
    NODE_EMBEDDING_COLUMN_NAME: str = "embedding"
    CHUNK_NODE_TEXT_COLUMN_NAME: str = "text"
    ENTITY_NODE_NAME_COLUMN_NAME: str = "name"
    TARGET_NODE_KEY_COLUMN_NAME: str = "target_id"
    DYNAMIC_PROPERTY_COLUMN_NAME: str = "properties"
    DYNAMIC_LABEL_COLUMN_NAME: str = "label"
    PROPERTY_PREFIX: str = "prop_"

    name: str
    kind: str
    key_columns: List[str]
    base_table_name: str
    labels: List[str]
    properties: CaseInsensitiveDict[str]
    # types: A dictionary where keys are property names (strings) and values are Spanner type definitions
    types: CaseInsensitiveDict[param_types.Type]
    source: NodeReference
    target: NodeReference

    def is_dynamic_schema(self) -> bool:
        """
        Checks if the current schema is dynamic.
        A schema is considered dynamic if the type of the `DYNAMIC_PROPERTY_COLUMN_NAME`
        is defined as JSON. This indicates that the properties of the nodes/edges
        are stored in a JSON column, allowing for a flexible schema.
        """
        return (
            self.types.get(ElementSchema.DYNAMIC_PROPERTY_COLUMN_NAME, None)
            == param_types.JSON
        )

    @staticmethod
    def make_node_schema(
        node_label: str,
        graph_name: str,
        property_types: CaseInsensitiveDict[param_types.Type],
    ) -> ElementSchema:
        """Creates a node schema for a given node type and label."""
        node = ElementSchema()
        node.types = property_types
        node.properties = CaseInsensitiveDict({prop: prop for prop in node.types})
        node.labels = [node_label]
        node.base_table_name = "%s_%s" % (graph_name, node_label)
        node.name = node.base_table_name
        node.kind = NODE_KIND
        node.key_columns = [ElementSchema.NODE_KEY_COLUMN_NAME]
        return node

    @staticmethod
    def make_edge_schema(
        edge_label: str,
        graph_schema: SpannerGraphSchema,
        key_columns: List[str],
        property_types: CaseInsensitiveDict[param_types.Type],
        source_node_table: str,
        target_node_table: str,
    ) -> ElementSchema:
        """Creates an edge schema for a given edge type and label."""
        edge = ElementSchema()
        edge.types = property_types
        edge.properties = CaseInsensitiveDict({prop: prop for prop in edge.types})

        edge.labels = [edge_label]
        edge.base_table_name = "%s_%s" % (graph_schema.graph_name, edge_label)
        edge.key_columns = key_columns
        edge.name = edge.base_table_name
        edge.kind = EDGE_KIND

        source_node_schema = graph_schema.node_tables.get(source_node_table)

        if source_node_schema is None:
            raise ValueError("No source node schema `%s` found" % source_node_table)

        target_node_schema = graph_schema.node_tables.get(target_node_table)

        if target_node_schema is None:
            raise ValueError("No target node schema `%s` found" % target_node_table)

        edge.source = NodeReference(
            source_node_schema.name,
            [ElementSchema.NODE_KEY_COLUMN_NAME],
            [ElementSchema.NODE_KEY_COLUMN_NAME],
        )
        edge.target = NodeReference(
            target_node_schema.name,
            [ElementSchema.NODE_KEY_COLUMN_NAME],
            [ElementSchema.TARGET_NODE_KEY_COLUMN_NAME],
        )
        return edge

    @staticmethod
    def from_static_nodes(
        name: str, nodes: List[LabelledNode], graph_schema: SpannerGraphSchema
    ) -> ElementSchema:
        """Builds ElementSchema from a list of nodes.

        Args:
          name: name of the schema.
          nodes: a non-empty list of nodes.
          graph_schema: schema of the graph.

        Returns:
          ElementSchema: schema representation of the nodes.

        Raises:
          ValueError: An error occured building element schema.
        """
        if not nodes:
            raise ValueError("The list of nodes should not be empty")
        property_prefix = ElementSchema.PROPERTY_PREFIX

        types = CaseInsensitiveDict(
            {
                (property_prefix + k): TypeUtility.value_to_param_type(v)
                for n in nodes
                for k, v in n.properties.items()
            }
        )
        if ElementSchema.NODE_KEY_COLUMN_NAME in types:
            raise ValueError(
                "Node properties should not contain property with name: `%s`"
                % ElementSchema.NODE_KEY_COLUMN_NAME
            )
        types[ElementSchema.NODE_KEY_COLUMN_NAME] = TypeUtility.value_to_param_type(
            nodes[0].id
        )
        types[ElementSchema.NODE_EMBEDDING_COLUMN_NAME] = param_types.Array(
            param_types.FLOAT64
        )
        if any([isinstance(n, ChunkNode) for n in nodes]):
            types[ElementSchema.CHUNK_NODE_TEXT_COLUMN_NAME] = param_types.STRING
        if any([isinstance(n, EntityNode) for n in nodes]):
            types[ElementSchema.ENTITY_NODE_NAME_COLUMN_NAME] = param_types.STRING

        return ElementSchema.make_node_schema(name, graph_schema.graph_name, types)

    @staticmethod
    def from_dynamic_nodes(
        nodes: List[LabelledNode], graph_schema: SpannerGraphSchema
    ) -> ElementSchema:
        """Builds ElementSchema from a list of nodes.

        Args:
          nodes: a non-empty list of nodes.
          graph_schema: schema of the graph;

        Returns:
          ElementSchema: schema representation of the nodes.

        Raises:
          ValueError: An error occured building element schema.
        """
        if not nodes:
            raise ValueError("The list of nodes should not be empty")

        types = CaseInsensitiveDict(
            {
                ElementSchema.DYNAMIC_PROPERTY_COLUMN_NAME: param_types.JSON,
                ElementSchema.DYNAMIC_LABEL_COLUMN_NAME: param_types.STRING,
                ElementSchema.NODE_KEY_COLUMN_NAME: TypeUtility.value_to_param_type(
                    nodes[0].id
                ),
                ElementSchema.NODE_EMBEDDING_COLUMN_NAME: param_types.Array(
                    param_types.FLOAT64
                ),
            }
        )
        if any([isinstance(n, ChunkNode) for n in nodes]):
            types[ElementSchema.CHUNK_NODE_TEXT_COLUMN_NAME] = param_types.STRING
        if any([isinstance(n, EntityNode) for n in nodes]):
            types[ElementSchema.ENTITY_NODE_NAME_COLUMN_NAME] = param_types.STRING
        property_prefix = ElementSchema.PROPERTY_PREFIX

        types.update(
            CaseInsensitiveDict(
                {
                    (property_prefix + k): TypeUtility.value_to_param_type(v)
                    for n in nodes
                    for k, v in n.properties.items()
                    if k in graph_schema.static_node_properties
                }
            )
        )
        return ElementSchema.make_node_schema(NODE_KIND, graph_schema.graph_name, types)

    @staticmethod
    def from_static_edges(
        name: str,
        edges: List[Tuple[Relation, str, str]],
        graph_schema: SpannerGraphSchema,
    ) -> ElementSchema:
        """Builds ElementSchema from a list of edges.

        Args:
          name: name of the schema;
          edges: a non-empty list of edges.
          graph_schema: schema of the graph;

        Returns:
          ElementSchema: schema representation of the edges.

        Raises:
          ValueError: An error occured building element schema.
        """
        if not edges:
            raise ValueError("The list of edges should not be empty")
        property_prefix = ElementSchema.PROPERTY_PREFIX

        types = CaseInsensitiveDict(
            {
                (property_prefix + k): TypeUtility.value_to_param_type(v)
                for e, _, _ in edges
                for k, v in e.properties.items()
            }
        )

        for col_name in [
            ElementSchema.NODE_KEY_COLUMN_NAME,
            ElementSchema.TARGET_NODE_KEY_COLUMN_NAME,
        ]:
            if col_name in types:
                raise ValueError(
                    "Edge properties should not contain property with name: `%s`"
                    % col_name
                )
        types[ElementSchema.NODE_KEY_COLUMN_NAME] = TypeUtility.value_to_param_type(
            edges[0][0].source_id
        )
        types[ElementSchema.TARGET_NODE_KEY_COLUMN_NAME] = (
            TypeUtility.value_to_param_type(edges[0][0].target_id)
        )
        types[ElementSchema.DYNAMIC_LABEL_COLUMN_NAME] = param_types.STRING
        return ElementSchema.make_edge_schema(
            name,
            graph_schema,
            [
                ElementSchema.NODE_KEY_COLUMN_NAME,
                ElementSchema.TARGET_NODE_KEY_COLUMN_NAME,
            ],
            types,
            edges[0][1],
            edges[0][2],
        )

    @staticmethod
    def from_dynamic_edges(
        edges: List[Tuple[Relation, str, str]],
        graph_schema: SpannerGraphSchema,
    ) -> ElementSchema:
        """Builds ElementSchema from a list of edges.

        Args:
          edges: a non-empty list of edges.
          graph_schema: schema of the graph.

        Returns:
          ElementSchema: schema representation of the edges.

        Raises:
          ValueError: An error occured building element schema.
        """
        if not edges:
            raise ValueError("The list of edges should not be empty")

        types = CaseInsensitiveDict(
            {
                ElementSchema.DYNAMIC_PROPERTY_COLUMN_NAME: param_types.JSON,
                ElementSchema.DYNAMIC_LABEL_COLUMN_NAME: param_types.STRING,
                ElementSchema.NODE_KEY_COLUMN_NAME: TypeUtility.value_to_param_type(
                    edges[0][0].source_id
                ),
                ElementSchema.TARGET_NODE_KEY_COLUMN_NAME: (
                    TypeUtility.value_to_param_type(edges[0][0].target_id)
                ),
            }
        )
        property_prefix = ElementSchema.PROPERTY_PREFIX

        types.update(
            CaseInsensitiveDict(
                {
                    (property_prefix + k): TypeUtility.value_to_param_type(v)
                    for e, _, _ in edges
                    for k, v in e.properties.items()
                    if k in graph_schema.static_edge_properties
                }
            )
        )
        return ElementSchema.make_edge_schema(
            EDGE_KIND,
            graph_schema,
            [
                ElementSchema.NODE_KEY_COLUMN_NAME,
                ElementSchema.TARGET_NODE_KEY_COLUMN_NAME,
                ElementSchema.DYNAMIC_LABEL_COLUMN_NAME,
            ],
            types,
            edges[0][1],
            edges[0][2],
        )

    def add_nodes(
        self, nodes: List[LabelledNode]
    ) -> Generator[Tuple[str, Tuple[str, ...], List[List[Any]]], None, None]:
        """Builds the data required to add a list of nodes to Spanner.

        Args:
          nodes: a list of Nodes.

        Yields:
          A tuple consists of the following:
            str: a table name;
            Tuple[str, ...]: a tuple of column names;
            List[List[Any]]: a list of rows.

        Raises:
          ValueError: An error occured adding nodes.
        """
        if not nodes:
            raise ValueError("Empty list of nodes")

        rows_by_columns: Dict[Tuple[str, ...], List[List[Any]]] = {}
        property_prefix = ElementSchema.PROPERTY_PREFIX
        for node in nodes:
            properties = {(property_prefix + k): v for k, v in node.properties.items()}
            properties[ElementSchema.NODE_KEY_COLUMN_NAME] = node.id
            properties[ElementSchema.NODE_EMBEDDING_COLUMN_NAME] = node.embedding

            if isinstance(node, ChunkNode):
                properties[ElementSchema.CHUNK_NODE_TEXT_COLUMN_NAME] = node.text
            elif isinstance(node, EntityNode):
                properties[ElementSchema.ENTITY_NODE_NAME_COLUMN_NAME] = node.name

            # If the schema is dynamic, add the dynamic properties to the properties dictionary
            if self.is_dynamic_schema():
                dynamic_properties = {
                    k: TypeUtility.value_for_json(v)
                    for k, v in node.properties.items()
                    if k not in self.types
                }
                properties[ElementSchema.DYNAMIC_PROPERTY_COLUMN_NAME] = JsonObject(
                    json.loads(json.dumps(dynamic_properties))
                )
                properties[ElementSchema.DYNAMIC_LABEL_COLUMN_NAME] = node.label

            columns = tuple(sorted((k for k in properties if k in self.types)))
            row = [properties[k] for k in columns]
            rows_by_columns.setdefault(columns, []).append(row)

        for columns, rows in rows_by_columns.items():
            yield self.base_table_name, columns, rows

    def add_edges(
        self, edges: List[Relation]
    ) -> Generator[Tuple[str, Tuple[str, ...], List[List[Any]]], None, None]:
        """Builds the data required to add a list of edges to Spanner.

        Args:
          edges: a list of Relationships.

        Yields:
          A tuple consists of the following:
            str: a table name;
            Tuple[str, ...]: a tuple of column names;
            List[List[Any]]: a list of rows.

        Raises:
          ValueError: An error occured adding edges.
        """
        if not edges:
            raise ValueError("Empty list of edges")

        rows_by_columns: Dict[Tuple[str, ...], List[List[Any]]] = {}
        property_prefix = ElementSchema.PROPERTY_PREFIX
        for edge in edges:
            properties = {(property_prefix + k): v for k, v in edge.properties.items()}
            properties[ElementSchema.NODE_KEY_COLUMN_NAME] = edge.source_id
            properties[ElementSchema.TARGET_NODE_KEY_COLUMN_NAME] = edge.target_id
            properties[ElementSchema.DYNAMIC_LABEL_COLUMN_NAME] = edge.label

            if self.is_dynamic_schema():
                dynamic_properties = {
                    k: TypeUtility.value_for_json(v)
                    for k, v in edge.properties.items()
                    if k not in self.types
                }
                properties[ElementSchema.DYNAMIC_PROPERTY_COLUMN_NAME] = JsonObject(
                    json.loads(json.dumps(dynamic_properties))
                )

            columns = tuple(sorted((k for k in properties if k in self.types)))
            row = [properties[k] for k in columns]
            rows_by_columns.setdefault(columns, []).append(row)

        for columns, rows in rows_by_columns.items():
            yield self.base_table_name, columns, rows

    @staticmethod
    def from_info_schema(
        element_schema: Dict[str, Any],
        property_decls: List[Any],
    ) -> ElementSchema:
        """Builds ElementSchema from information schema representation of an element.

        Args:
          element_schema: the information schema representation of an element;
          property_decls: the information schema representation of property
            declarations.

        Returns:
          ElementSchema

        Raises:
          ValueError: An error occured building graph schema.
        """
        element = ElementSchema()
        element.name = element_schema["name"]
        element.kind = element_schema["kind"]
        if element.kind not in [NODE_KIND, EDGE_KIND]:
            raise ValueError("Invalid element kind `{}`".format(element.kind))

        element.key_columns = element_schema["keyColumns"]
        element.base_table_name = element_schema["baseTableName"]
        element.labels = element_schema["labelNames"]
        element.properties = CaseInsensitiveDict(
            {
                prop_def["propertyDeclarationName"]: prop_def["valueExpressionSql"]
                for prop_def in element_schema.get("propertyDefinitions", [])
            }
        )
        element.types = CaseInsensitiveDict(
            {
                decl["name"]: TypeUtility.schema_str_to_spanner_type(decl["type"])
                for decl in property_decls
                if decl["name"] in element.properties
            }
        )

        if element.kind == EDGE_KIND:
            element.source = NodeReference(
                element_schema["sourceNodeTable"]["nodeTableName"],
                element_schema["sourceNodeTable"]["nodeTableColumns"],
                element_schema["sourceNodeTable"]["edgeTableColumns"],
            )
            element.target = NodeReference(
                element_schema["destinationNodeTable"]["nodeTableName"],
                element_schema["destinationNodeTable"]["nodeTableColumns"],
                element_schema["destinationNodeTable"]["edgeTableColumns"],
            )
        return element

    def to_ddl(self, graph_schema: SpannerGraphSchema) -> str:
        """Returns a CREATE TABLE ddl that represents the element schema.

        Args:
          graph_schema: Spanner Graph schema.

        Returns:
          str: a string of CREATE TABLE ddl statement.

        Raises:
          ValueError: An error occured building graph ddl.
        """

        to_identifier = GraphDocumentUtility.to_identifier
        to_identifiers = GraphDocumentUtility.to_identifiers

        def get_reference_node_table(name: str) -> str:
            node_schema = graph_schema.node_tables.get(name)
            if node_schema is None:
                raise ValueError("No node schema `%s` found" % name)
            return node_schema.base_table_name

        return """CREATE TABLE {} (
          {}{}
        ) PRIMARY KEY ({}){}
      """.format(
            to_identifier(self.base_table_name),
            ",\n ".join(
                (
                    "{} {}".format(
                        to_identifier(n),
                        TypeUtility.spanner_type_to_schema_str(
                            t, include_type_annotations=True
                        ),
                    )
                    for n, t in self.types.items()
                )
            ),
            (
                ",\n FOREIGN KEY ({}) REFERENCES {}({})".format(
                    ", ".join(to_identifiers(self.target.edge_keys)),
                    to_identifier(get_reference_node_table(self.target.node_name)),
                    ", ".join(to_identifiers(self.target.node_keys)),
                )
                if self.kind == EDGE_KIND
                else ""
            ),
            ",".join(to_identifiers(self.key_columns)),
            (
                ", INTERLEAVE IN PARENT {}".format(
                    to_identifier(get_reference_node_table(self.source.node_name))
                )
                if self.kind == EDGE_KIND
                else ""
            ),
        )

    def evolve(self, new_schema: ElementSchema) -> List[str]:
        """Evolves current schema from the new schema.

        Args:
          new_schema: an ElementSchema representing new nodes/edges.

        Returns:
          List[str]: a list of DDL statements.

        Raises:
          ValueError: An error occured evolving graph schema.
        """
        if self.kind != new_schema.kind:
            raise ValueError(
                "Schema with name `{}` should have the same kind, got {}, expected {}".format(
                    self.name, new_schema.kind, self.kind
                )
            )
        if self.key_columns != new_schema.key_columns:
            raise ValueError(
                "Schema with name `{}` should have the same keys, got {}, expected {}".format(
                    self.name, new_schema.key_columns, self.key_columns
                )
            )
        if self.base_table_name.casefold() != new_schema.base_table_name.casefold():
            raise ValueError(
                "Schema with name `{}` should have the same base table name, got {},"
                " expected {}".format(
                    self.name, new_schema.base_table_name, self.base_table_name
                )
            )

        for k, v in new_schema.properties.items():
            if k in self.properties:
                if self.properties[k].casefold() != v.casefold():
                    raise ValueError(
                        "Property with name `{}` should have the same definition, got {},"
                        " expected {}".format(k, v, self.properties[k])
                    )

        for k, v in new_schema.types.items():
            if k in self.types:
                if self.types[k] != v:
                    raise ValueError(
                        "Property with name `{}` should have the same type, got {},"
                        " expected {}".format(k, v, self.types[k])
                    )

        to_identifier = GraphDocumentUtility.to_identifier
        ddls = [
            "ALTER TABLE {} ADD COLUMN {} {}".format(
                to_identifier(self.base_table_name),
                to_identifier(n),
                TypeUtility.spanner_type_to_schema_str(
                    t, include_type_annotations=True
                ),
            )
            for n, t in new_schema.types.items()
            if n not in self.properties
        ]
        self.properties.update(new_schema.properties)
        self.types.update(new_schema.types)
        return ddls


class SpannerGraphSchema(object):
    """Schema representation of a property graph."""

    GRAPH_INFORMATION_SCHEMA_QUERY_TEMPLATE = """
    SELECT property_graph_metadata_json
    FROM INFORMATION_SCHEMA.PROPERTY_GRAPHS
    WHERE property_graph_name = '{}'
  """

    def __init__(
        self,
        graph_name: str,
        use_flexible_schema: bool,
        static_node_properties: Optional[List[str]] = None,
        static_edge_properties: Optional[List[str]] = None,
    ):
        """Initializes the graph schema.

        Args:
          graph_name: the name of the graph;
          use_flexible_schema: whether to use the flexible schema which uses a JSON
            blob to store node and edge properties;
          static_node_properties: in flexible schema, treat these node properties as
            static;
          static_edge_properties: in flexible schema, treat these edge properties as
            static.

        Raises:
          ValueError: An error occured initializing graph schema.
        """
        if not GraphDocumentUtility.is_valid_identifier(graph_name):
            raise ValueError(
                "Graph name `{}` is not a valid identifier".format(graph_name)
            )

        self.graph_name: str = graph_name
        self.node_tables: CaseInsensitiveDict[ElementSchema] = CaseInsensitiveDict({})
        self.edge_tables: CaseInsensitiveDict[ElementSchema] = CaseInsensitiveDict({})
        self.labels: CaseInsensitiveDict[Label] = CaseInsensitiveDict({})
        self.properties: CaseInsensitiveDict[param_types.Type] = CaseInsensitiveDict({})
        self.node_properties: CaseInsensitiveDict[param_types.Type] = (
            CaseInsensitiveDict({})
        )
        self.use_flexible_schema = use_flexible_schema
        self.static_node_properties = set(static_node_properties or [])
        self.static_edge_properties = set(static_edge_properties or [])
        self.graph_exists = False

    def evolve_from_nodes(
        self,
        nodes: Dict[str, List[LabelledNode]],
    ) -> Tuple[List[str], Dict[str, ElementSchema]]:
        """Evolves the graph schema based on new nodes and edges.

        This method updates the internal schema representation by adding new
        node and edge types, or evolving existing ones. It generates DDL statements
        to reflect the changes in the underlying Spanner database.

        Args:
          nodes: A dictionary of node types to lists of LabelledNodes.

        Returns:
          A tuple containing:
            - A list of DDL statements to apply to the database.
            - A dictionary mapping node types to their corresponding ElementSchema.
        """
        ddls = []
        node_schema_mapping = {}

        for k, ns in nodes.items():
            node_schema = (
                ElementSchema.from_static_nodes(k, ns, self)
                if not self.use_flexible_schema
                else ElementSchema.from_dynamic_nodes(ns, self)
            )
            ddls.extend(self._update_node_schema(node_schema))
            node_schema_mapping[k] = self.node_tables[node_schema.base_table_name]
            self._update_labels_and_properties(node_schema)

        if ddls:
            # Add the final CREATE PROPERTY GRAPH statement
            ddls += [self.to_ddl()]
        return ddls, node_schema_mapping

    def evolve_from_edges(
        self,
        edges: Dict[str, List[Tuple[Relation, str, str]]],
    ) -> Tuple[List[str], Dict[str, ElementSchema]]:
        """Evolves the graph schema based on new edges.

        This method updates the internal schema representation by adding new
        edge types, or evolving existing ones. It generates DDL statements
        to reflect the changes in the underlying Spanner database.

        Args:
          edges: A dictionary of edge types to lists of Tuple[Relation, str, str].

        Returns:
          A tuple containing:
            - A list of DDL statements to apply to the database.
            - A dictionary mapping edge types to their corresponding ElementSchema.
        """
        ddls = []
        edge_schema_mapping = {}

        for k, es in edges.items():
            edge_schema = (
                ElementSchema.from_static_edges(k, es, self)
                if not self.use_flexible_schema
                else ElementSchema.from_dynamic_edges(es, self)
            )
            ddls.extend(self._update_edge_schema(edge_schema))
            edge_schema_mapping[k] = self.edge_tables[edge_schema.base_table_name]
            self._update_labels_and_properties(edge_schema)

        if ddls:
            # Add the final CREATE PROPERTY GRAPH statement
            ddls += [self.to_ddl()]
        return ddls, edge_schema_mapping

    def from_information_schema(self, info_schema: Dict[str, Any]) -> None:
        """Builds the schema from information schema represenation.

        Args:
          info_schema: the information schema represenation of a graph;
        """
        property_decls = info_schema.get("propertyDeclarations", [])
        for node in info_schema["nodeTables"]:
            node_schema = ElementSchema.from_info_schema(node, property_decls)
            self._update_node_schema(node_schema)
            self._update_labels_and_properties(node_schema)

        for edge in info_schema.get("edgeTables", []):
            edge_schema = ElementSchema.from_info_schema(edge, property_decls)
            self._update_edge_schema(edge_schema)
            self._update_labels_and_properties(edge_schema)

        self.graph_exists = True

    def __repr__(self) -> str:
        """Builds a string representation of the graph schema.

        Returns:
          str: a string representation of the graph schema.
        """
        properties = CaseInsensitiveDict(
            {
                k: TypeUtility.spanner_type_to_schema_str(v)
                for k, v in self.properties.items()
            }
        )
        return json.dumps(
            {
                "Name of graph": self.graph_name,
                "Node properties per node type": {
                    node.name: [
                        {
                            "property name": name,
                            "property type": properties[name],
                        }
                        for name in node.properties.keys()
                    ]
                    for node in self.node_tables.values()
                },
                "Edge properties per edge type": {
                    edge.name: [
                        {
                            "property name": name,
                            "property type": properties[name],
                        }
                        for name in edge.properties.keys()
                    ]
                    for edge in self.edge_tables.values()
                },
                "Node labels per node type": {
                    node.name: node.labels for node in self.node_tables.values()
                },
                "Edge labels per edge type": {
                    edge.name: edge.labels for edge in self.edge_tables.values()
                },
                "Edges": {
                    edge.name: "From {} nodes to {} nodes".format(
                        edge.source.node_name, edge.target.node_name
                    )
                    for edge in self.edge_tables.values()
                },
            },
            indent=2,
        )

    def to_ddl(self) -> str:
        """Returns a CREATE PROPERTY GRAPH ddl that represents the graph schema.

        Returns:
          str: a string of CREATE PROPERTY GRAPH ddl statement.
        """
        to_identifier = GraphDocumentUtility.to_identifier
        to_identifiers = GraphDocumentUtility.to_identifiers

        def construct_label_and_properties(
            target_label: str,
            labels: CaseInsensitiveDict[Label],
            element: ElementSchema,
        ) -> str:
            props = labels[target_label].prop_names
            defs = [
                "{} AS {}".format(v if k != v else to_identifier(k), to_identifier(k))
                for k, v in element.properties.items()
                if k in props
            ]
            return """LABEL {} PROPERTIES({})""".format(
                to_identifier(target_label), ", ".join(defs)
            )

        def construct_label_and_properties_list(
            target_labels: List[str],
            labels: CaseInsensitiveDict[Label],
            element: ElementSchema,
        ) -> str:
            return "\n".join(
                (
                    construct_label_and_properties(target_label, labels, element)
                    for target_label in target_labels
                )
            )

        def construct_columns(cols: List[str]) -> str:
            return ", ".join(to_identifiers(cols))

        def construct_key(keys: List[str]) -> str:
            return "KEY({})".format(construct_columns(keys))

        def construct_node_reference(
            endpoint_type: str, endpoint: NodeReference
        ) -> str:
            return "{} KEY({}) REFERENCES {}({})".format(
                endpoint_type,
                construct_columns(endpoint.edge_keys),
                to_identifier(endpoint.node_name),
                construct_columns(endpoint.node_keys),
            )

        def construct_element_table(
            element: ElementSchema, labels: CaseInsensitiveDict[Label]
        ) -> str:
            definition = [
                "{} AS {}".format(
                    to_identifier(element.base_table_name),
                    to_identifier(element.name),
                ),
                construct_key(element.key_columns),
            ]
            if element.kind == EDGE_KIND:
                definition += [
                    construct_node_reference("SOURCE", element.source),
                    construct_node_reference("DESTINATION", element.target),
                ]
            definition += [
                construct_label_and_properties_list(element.labels, labels, element)
            ]
            return "\n    ".join(definition)

        ddl = "CREATE OR REPLACE PROPERTY GRAPH {}".format(
            to_identifier(self.graph_name)
        )
        ddl += "\nNODE TABLES(\n  "
        ddl += ",\n  ".join(
            (
                construct_element_table(node, self.labels)
                for node in self.node_tables.values()
            )
        )
        ddl += "\n)"
        if self.edge_tables:
            ddl += "\nEDGE TABLES(\n  "
            ddl += ",\n  ".join(
                (
                    construct_element_table(edge, self.labels)
                    for edge in self.edge_tables.values()
                )
            )
            ddl += "\n)"
        return ddl

    def _update_node_schema(self, node_schema: ElementSchema) -> List[str]:
        """Evolves node schema.

        Args:
          node_schema: a node ElementSchema.

        Returns:
          List[str]: a list of DDL statements that requires to evolve the schema.
        """

        if node_schema.base_table_name not in self.node_tables:
            ddls = [node_schema.to_ddl(self)]
            self.node_tables[node_schema.base_table_name] = node_schema
        else:
            ddls = self.node_tables[node_schema.base_table_name].evolve(node_schema)

        return ddls

    def _update_edge_schema(self, edge_schema: ElementSchema) -> List[str]:
        """Evolves edge schema.

        Args:
          edge_schema: an edge ElementSchema.

        Returns:
          List[str]: a list of DDL statements that requires to evolve the schema.
        """
        if edge_schema.base_table_name not in self.edge_tables:
            ddls = [edge_schema.to_ddl(self)]
            self.edge_tables[edge_schema.base_table_name] = edge_schema
        else:
            ddls = self.edge_tables[edge_schema.base_table_name].evolve(edge_schema)

        return ddls

    def _update_labels_and_properties(self, element_schema: ElementSchema) -> None:
        """Updates labels and properties based on an element schema.

        Args:
          element_schema: an ElementSchema.
        """
        for l in element_schema.labels:
            if l in self.labels:
                self.labels[l].prop_names.update(element_schema.properties.keys())
            else:
                self.labels[l] = Label(
                    l,
                    element_schema.base_table_name,
                    set(element_schema.properties.keys()),
                )

        self.properties.update(element_schema.types)
        if element_schema.kind == NODE_KIND:
            self.node_properties.update(element_schema.types)
