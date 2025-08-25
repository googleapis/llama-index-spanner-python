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

import pytest
from llama_index.core import PropertyGraphIndex, Settings
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.readers.wikipedia import WikipediaReader

from llama_index_spanner.graph_retriever import SpannerGraphCustomRetriever
from tests.utils import get_random_suffix, get_resources, google_api_key


@pytest.fixture(scope="module", params=["static", "flexible"])
def index_and_models(request):
    """Setup the index for integration tests."""
    schema_type = request.param
    graph_store, _, query_llm, embed_model = get_resources(
        f"e2e_rag_{schema_type}_{get_random_suffix()}",
        clean_up=True,
        use_flexible_schema=(schema_type == "flexible"),
    )

    loader = WikipediaReader()
    documents = loader.load_data(pages=["Google"], auto_suggest=False)

    index_llm = GoogleGenAI(
        model="gemini-1.5-pro-latest",
        api_key=google_api_key,
    )
    Settings.llm = query_llm
    Settings.embed_model = embed_model

    index = PropertyGraphIndex.from_documents(
        documents,
        property_graph_store=graph_store,
        llm=query_llm,
        embed_model=embed_model,
        embed_kg_nodes=True,
        kg_extractors=[
            SchemaLLMPathExtractor(
                llm=index_llm,
                max_triplets_per_chunk=1000,
                num_workers=4,
            )
        ],
        show_progress=True,
    )
    yield index, query_llm, embed_model
    graph_store.clean_up()


@pytest.mark.flaky(retries=3, only_on=[AssertionError], delay=1)
def test_e2e_rag(index_and_models):
    """
    Test End-to-End RAG flow from document loading to querying with a custom retriever.
    This test is parameterized to run for both 'static' and 'flexible' schemas.
    """
    index, llm, embed_model = index_and_models

    retriever = SpannerGraphCustomRetriever(
        graph_store=index.property_graph_store,
        embed_model=embed_model,
        llm_text_to_gql=llm,
        include_raw_response_as_metadata=True,
        verbose=True,
        similarity_top_k=10,
        path_depth=3,
    )

    query_engine = RetrieverQueryEngine(retriever=retriever)

    # Query 1: Parent company
    response1 = query_engine.query("what is parent company of Google?")

    # Query 2: Office locations
    response2 = query_engine.query("Where are all the Google offices located?")

    # Query 3: Products
    response3 = query_engine.query("Some Products of Google?")

    assert any(
        (
            "Alphabet" in str(response1),
            "Mountain View" in str(response2),
            (
                "Search" in str(response3)
                or "Android" in str(response3)
                or "Chrome" in str(response3)
            ),
        )
    )
