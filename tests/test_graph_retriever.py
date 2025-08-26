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

from unittest.mock import MagicMock, patch

import pytest
from llama_index.core import PromptTemplate, Settings
from llama_index.core.graph_stores.types import ChunkNode, EntityNode, Relation
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from llama_index_spanner.graph_retriever import (
    GQL_RESPONSE_SCORING_TEMPLATE,
    GQL_SYNTHESIS_RESPONSE_TEMPLATE,
    GQL_VERIFY_PROMPT,
    SpannerGraphCustomRetriever,
    SpannerGraphTextToGQLRetriever,
    VerifyGqlOutput,
)
from llama_index_spanner.property_graph_store import SpannerPropertyGraphStore
from tests.utils import get_random_suffix, get_resources


def _setup_sample_graph(graph_store, embed_model):
    """Sets up a sample graph in the given graph_store."""
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


@pytest.fixture(scope="module", params=["static", "flexible"])
def graph_store_with_sample_data(request):
    """Fixture to create a graph store with sample data, for both schema types."""
    schema_type = request.param
    graph_store, _, llm, embed_model = get_resources(
        f"retriever_test_{schema_type}_{get_random_suffix()}",
        clean_up=True,
        use_flexible_schema=(schema_type == "flexible"),
    )
    Settings.llm = llm
    _setup_sample_graph(graph_store, embed_model)
    yield graph_store, llm, embed_model
    graph_store.clean_up()


@pytest.mark.flaky(retries=3, only_on=[AssertionError], delay=1)
def test_spanner_graph_text_to_gql_retriever(graph_store_with_sample_data):
    """Test SpannerGraphTextToGQLRetriever with a sample graph."""
    graph_store, llm, _ = graph_store_with_sample_data

    if graph_store.schema.use_flexible_schema:
        pytest.skip(
            "SpannerGraphTextToGQLRetriever is not supported for flexible schema yet."
        )

    retriever = SpannerGraphTextToGQLRetriever(
        graph_store=graph_store,
        llm=llm,
        include_raw_response_as_metadata=True,
        verbose=True,
    )

    res1 = retriever.retrieve("Where does Elias Thorne's sibling live?")
    res2 = retriever.retrieve("Who lives in desert?")

    assert any(("Capital City" in str(res1), "Elias Thorne" in str(res2)))


@pytest.mark.flaky(retries=3, only_on=[AssertionError], delay=1)
def test_spanner_graph_custom_retriever(graph_store_with_sample_data):
    """Test SpannerGraphCustomRetriever with a sample graph."""
    graph_store, llm, embed_model = graph_store_with_sample_data

    retriever = SpannerGraphCustomRetriever(
        graph_store=graph_store,
        embed_model=embed_model,
        llm_text_to_gql=llm,
        include_raw_response_as_metadata=True,
        verbose=True,
    )

    res1 = retriever.retrieve("Where does Elias Thorne's sibling live?")
    res2 = retriever.retrieve("Who lives in desert?")

    assert any(("Capital City" in str(res1), "Elias Thorne" in str(res2)))


def test_spanner_graph_text_to_gql_retriever_mocked():
    """Test SpannerGraphTextToGQLRetriever with a mocked LLM and graph store."""
    mock_graph_store = MagicMock()
    mock_llm = MagicMock()
    mock_text_to_gql_prompt = MagicMock(spec=PromptTemplate)
    mock_gql_validator = MagicMock()
    mock_summarization_template = MagicMock(spec=PromptTemplate)

    # Mock the LLM to return a specific GQL query
    expected_gql = "MATCH (p:Person)-[:Sibling]->(s:Person)-[:LivesIn]->(l:Location) WHERE p.name = 'Elias Thorne' RETURN l.name AS location"

    # Mock for verification step
    verify_output = VerifyGqlOutput(
        input_gql=expected_gql,
        made_change=False,
        explanation="No changes needed",
        verified_gql=expected_gql,
    )

    # Mock for summarization
    summarized_response = "The sibling lives in Capital City."

    # Mock for scoring
    score = 0.9

    # Set up side effects for mock_llm.predict
    mock_llm.predict.side_effect = [
        expected_gql,  # 1. GQL generation
        verify_output.json(),  # 2. GQL verification
        summarized_response,  # 3. Summarization
        str(score),  # 4. Scoring
    ]

    # Mock the graph store to return a specific result for the GQL query
    mock_graph_store.structured_query.return_value = [{"location": "Capital City"}]
    mock_graph_store.get_schema_str.return_value = "dummy schema"

    # Mock the validator to return True
    mock_gql_validator.return_value = True

    retriever = SpannerGraphTextToGQLRetriever(
        graph_store=mock_graph_store,
        llm=mock_llm,
        text_to_gql_prompt=mock_text_to_gql_prompt,
        gql_validator=mock_gql_validator,
        summarize_response=True,
        summarization_template=mock_summarization_template,
        verify_gql=True,
    )

    res = retriever.retrieve("Where does Elias Thorne's sibling live?")

    # Assert that the mocks were called correctly
    assert mock_llm.predict.call_count == 4

    # 1. GQL generation call
    gen_call = mock_llm.predict.call_args_list[0]
    assert gen_call.args[0] == mock_text_to_gql_prompt

    # Assert gql_validator was called
    mock_gql_validator.assert_called_once_with(expected_gql)

    # 2. GQL verification call
    verify_call = mock_llm.predict.call_args_list[1]
    assert verify_call.args[0] == GQL_VERIFY_PROMPT
    assert verify_call.kwargs["generated_gql"] == expected_gql

    # Assert that the graph store was called with the generated query
    mock_graph_store.structured_query.assert_called_once_with(expected_gql)

    # 3. Summarization call
    summary_call = mock_llm.predict.call_args_list[2]
    assert summary_call.args[0] == mock_summarization_template
    assert summary_call.kwargs["context"] == str([{"location": "Capital City"}])

    # 4. Scoring call
    scoring_call = mock_llm.predict.call_args_list[3]
    assert scoring_call.args[0] == GQL_RESPONSE_SCORING_TEMPLATE
    assert scoring_call.kwargs["retrieved_context"] == summarized_response

    # Assert that the final result contains the summarized response
    assert len(res) > 0
    assert res[0].node.text == summarized_response
    assert res[0].score == score


def test_spanner_graph_custom_retriever_mocked():
    """Test SpannerGraphCustomRetriever with mocked dependencies to check reranker and synthesis calls."""
    mock_graph_store = MagicMock(spec=SpannerPropertyGraphStore)
    # This is needed for SpannerGraphTextToGQLRetriever init inside SpannerGraphCustomRetriever
    mock_graph_store.supports_structured_queries = True
    mock_embed_model = MagicMock()
    mock_llm = MagicMock()
    mock_reranker_llm = MagicMock()

    with (
        patch(
            "llama_index_spanner.graph_retriever.VectorContextRetriever"
        ) as mock_vector_retriever_cls,
        patch(
            "llama_index_spanner.graph_retriever.SpannerGraphTextToGQLRetriever"
        ) as mock_gql_retriever_cls,
        patch("llama_index_spanner.graph_retriever.LLMRerank") as mock_reranker_cls,
    ):
        mock_vector_retriever = mock_vector_retriever_cls.return_value
        mock_gql_retriever = mock_gql_retriever_cls.return_value
        mock_reranker = mock_reranker_cls.return_value

        retriever = SpannerGraphCustomRetriever(
            graph_store=mock_graph_store,
            embed_model=mock_embed_model,
            llm_text_to_gql=mock_llm,
            llm_for_reranker=mock_reranker_llm,
        )

        # Mock retriever results
        vector_nodes = [NodeWithScore(node=TextNode(text="vector result"))]
        mock_vector_retriever.retrieve.return_value = vector_nodes

        gql_nodes = [NodeWithScore(node=TextNode(text="gql result"))]
        mock_gql_retriever.retrieve_from_graph.return_value = gql_nodes

        # Mock reranker result
        reranked_nodes = [NodeWithScore(node=TextNode(text="reranked result"))]
        mock_reranker.postprocess_nodes.return_value = reranked_nodes

        # Mock synthesis result
        synthesized_response = "This is the final synthesized response."
        mock_llm.predict.return_value = synthesized_response

        query_str = "some query"
        final_response = retriever.custom_retrieve(query_str)

        # Assertions
        mock_vector_retriever.retrieve.assert_called_once()
        query_bundle_arg = mock_vector_retriever.retrieve.call_args[0][0]
        assert isinstance(query_bundle_arg, QueryBundle)
        assert query_bundle_arg.query_str == query_str

        mock_gql_retriever.retrieve_from_graph.assert_called_once()
        query_bundle_arg_gql = mock_gql_retriever.retrieve_from_graph.call_args[0][0]
        assert isinstance(query_bundle_arg_gql, QueryBundle)
        assert query_bundle_arg_gql.query_str == query_str

        mock_reranker.postprocess_nodes.assert_called_once()
        call_args, _ = mock_reranker.postprocess_nodes.call_args
        assert call_args[0] == vector_nodes + gql_nodes
        assert isinstance(call_args[1], QueryBundle)
        assert call_args[1].query_str == query_str

        mock_llm.predict.assert_called_once_with(
            GQL_SYNTHESIS_RESPONSE_TEMPLATE,
            question=query_str,
            retrieved_response="reranked result",
        )

        assert final_response == synthesized_response
