# Copyright 2024 Google LLC

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
import random

from google.cloud import spanner
from llama_index.core.storage import StorageContext
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI

from llama_index_spanner import SpannerPropertyGraphStore

project_id = os.environ.get("PROJECT_ID") or "llamaindex-spanner-testing"
spanner_instance_id = (
    os.environ.get("SPANNER_INSTANCE_ID") or "test-llamaindex-instance"
)
spanner_database_id = os.environ.get("SPANNER_DATABASE_ID") or "test-google-db"
spanner_graph_name = os.environ.get("SPANNER_GRAPH_NAME") or "llama_index_graph"
google_api_key = os.environ.get("GOOGLE_API_KEY")


def get_spanner_property_graph_store(
    graph_name_suffix: str = "",
    use_flexible_schema: bool = False,
    clean_up: bool = True,
) -> SpannerPropertyGraphStore:
    """Get a SpannerPropertyGraphStore instance for testing."""
    graph_name = spanner_graph_name
    if graph_name_suffix:
        graph_name += "_" + graph_name_suffix
    return SpannerPropertyGraphStore(
        instance_id=spanner_instance_id,
        database_id=spanner_database_id,
        graph_name=graph_name,
        clean_up=clean_up,
        use_flexible_schema=use_flexible_schema,
        client=spanner.Client(project=project_id),
    )


def get_resources(
    graph_name_suffix: str = "",
    use_flexible_schema: bool = False,
    clean_up: bool = False,
):
    """Get the resources for testing."""
    graph_store = get_spanner_property_graph_store(
        graph_name_suffix, use_flexible_schema, clean_up
    )
    storage_context = StorageContext.from_defaults(property_graph_store=graph_store)
    llm = GoogleGenAI(
        model="gemini-2.0-flash",
        api_key=google_api_key,
    )
    embed_model = GoogleGenAIEmbedding(
        model_name="text-embedding-004", embed_batch_size=100
    )
    return graph_store, storage_context, llm, embed_model


def get_random_suffix() -> str:
    """Get a random suffix for testing."""
    return str(random.randint(1000000, 9999999))
