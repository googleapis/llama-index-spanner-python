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

import os

import pytest
from llama_index.core import PropertyGraphIndex, Settings
from llama_index.core.graph_stores.types import ChunkNode, EntityNode, Relation
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.readers.wikipedia import WikipediaReader

from llama_index_spanner.graph_retriever import (
    SpannerGraphCustomRetriever,
    SpannerGraphTextToGQLRetriever,
)
from tests.utils import get_random_suffix, get_resources

google_api_key = os.environ.get("GOOGLE_API_KEY")


def setup(schema_type):
    """Setup the index for integration tests."""
    graph_store, _, llm, embed_model = get_resources(
        schema_type + "_" + get_random_suffix(),
        clean_up=True,
        use_flexible_schema=schema_type == "flexible",
    )

    loader = WikipediaReader()
    documents = loader.load_data(pages=["Google"], auto_suggest=False)

    index_llm = GoogleGenAI(
        model="gemini-1.5-pro-latest",
        api_key=google_api_key,
    )
    PropertyGraphIndex.from_documents(
        documents,
        embed_model=embed_model,
        embed_kg_nodes=True,
        kg_extractors=[
            SchemaLLMPathExtractor(
                llm=index_llm,
                max_triplets_per_chunk=1000,
                num_workers=4,
            )
        ],
        llm=llm,
        show_progress=True,
        property_graph_store=graph_store,
    )
    return graph_store, llm, embed_model


def load(graph_store, llm, embed_model):
    """Load the retriever for integration tests."""
    Settings.llm = llm

    index = PropertyGraphIndex.from_existing(
        llm=llm, embed_model=embed_model, property_graph_store=graph_store
    )
    retriever = SpannerGraphCustomRetriever(
        graph_store=index.property_graph_store,
        embed_model=embed_model,
        llm=llm,
        include_raw_response_as_metadata=True,
        verbose=True,
    )
    return retriever


def test_graph_retriever():
    """Test the graph retriever."""
    for schema_type in ["static", "flexible"]:
        graph_store, llm, embed_model = setup(schema_type)
        retriever = load(graph_store, llm, embed_model)

        query_engine = RetrieverQueryEngine(retriever=retriever)
        response = query_engine.query("what is parent company of Google?")
        print(response)
        response = query_engine.query("Where are all the Google offices located?")
        print(response)
        response = query_engine.query("Some Products of Google?")
        print(response)
        graph_store.clean_up()


def setup2(schema_type):
    graph_store, _, llm, embed_model = get_resources(
        schema_type + "_" + get_random_suffix(),
        clean_up=True,
        use_flexible_schema=schema_type == "flexible",
    )
    Settings.llm = llm

    nodes = [
        EntityNode(
            name="Elias Thorne",
            label="Person",
            properties={
                "name": "Elias Thorne",
                "description": "lived in the desert",
            },
        ),
        EntityNode(
            name="Zephyr",
            label="Animal",
            properties={"name": "Zephyr", "description": "pet falcon"},
        ),
        EntityNode(
            name="Elara",
            label="Person",
            properties={
                "name": "Elara",
                "description": "resided in the capital city",
            },
        ),
        EntityNode(name="Desert", label="Location", properties={}),
        EntityNode(name="Capital City", label="Location", properties={}),
        ChunkNode(
            text=(
                "Elias Thorne lived in the desert. He was a skilled craftsman who"
                " worked with sandstone. Elias had a pet falcon named Zephyr. His"
                " sister, Elara, resided in the capital city and ran a spice"
                " shop. They rarely met due to the distance."
            )
        ),
    ]
    for node in nodes:
        node.embedding = embed_model.get_text_embedding(str(node))

    relations = [
        Relation(
            source_id=nodes[0].id,
            target_id=nodes[3].id,
            label="LivesIn",
            properties={},
        ),
        Relation(
            source_id=nodes[0].id,
            target_id=nodes[1].id,
            label="Owns",
            properties={},
        ),
        Relation(
            source_id=nodes[2].id,
            target_id=nodes[4].id,
            label="LivesIn",
            properties={},
        ),
        Relation(
            source_id=nodes[0].id,
            target_id=nodes[2].id,
            label="Sibling",
            properties={},
        ),
    ]

    graph_store.upsert_nodes(nodes)
    graph_store.upsert_relations(relations)

    retriever = SpannerGraphCustomRetriever(
        graph_store=graph_store,
        embed_model=embed_model,
        llm_text_to_gql=llm,
        include_raw_response_as_metadata=True,
        verbose=True,
    )

    retriever2 = SpannerGraphTextToGQLRetriever(
        graph_store=graph_store,
        llm=llm,
        include_raw_response_as_metadata=True,
        verbose=True,
    )

    return retriever, retriever2, graph_store


@pytest.fixture
def retrievers_static():
    retriever, retriever2, graph_store = setup2("static")
    yield retriever, retriever2
    graph_store.clean_up()


@pytest.fixture
def retrievers_dynamic():
    retriever, retriever2, graph_store = setup2("flexible")
    yield retriever, retriever2
    graph_store.clean_up()


@pytest.mark.flaky(retries=3, only_on=[AssertionError], delay=1)
def test_graph_retriever2_static(retrievers_static):
    """Test the graph retriever."""
    for retriever in retrievers_static:
        res = retriever.retrieve("Where does Elias Thorne's sibling live?")
        assert "Capital City" in str(res)

        res = retriever.retrieve("Who lives in desert?")
        assert "Elias Thorne" in str(res)


@pytest.mark.flaky(retries=3, only_on=[AssertionError], delay=1)
def test_graph_retriever2_dynamic(retrievers_dynamic):
    """Test the graph retriever."""
    retriever = retrievers_dynamic[0]  # only NL2GQL fails
    res = retriever.retrieve("Where does Elias Thorne's sibling live?")
    assert "Capital City" in str(res)

    res = retriever.retrieve("Who lives in desert?")
    assert "Elias Thorne" in str(res)
