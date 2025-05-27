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

from typing import Any, Callable, List, Optional, Union

from llama_index.core import PromptTemplate, Settings, schema
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.indices.property_graph import BasePGRetriever
from llama_index.core.llms import LLM
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.postprocessor.llm_rerank import LLMRerank
from llama_index.core.prompts import PromptType
from llama_index.core.retrievers import (
    CustomPGRetriever,
    VectorContextRetriever,
)
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.vector_stores.types import VectorStore
from pydantic import BaseModel

from .graph_utils import extract_gql, fix_gql_syntax
from .prompts import (
    DEFAULT_GQL_FIX_TEMPLATE,
    DEFAULT_GQL_VERIFY_TEMPLATE,
    DEFAULT_SPANNER_GQL_TEMPLATE,
    DEFAULT_SUMMARY_TEMPLATE,
)
from .property_graph_store import SpannerPropertyGraphStore


GQL_GENERATION_PROMPT = PromptTemplate(
    template=DEFAULT_SPANNER_GQL_TEMPLATE,
    prompt_type=PromptType.TEXT_TO_GRAPH_QUERY,
)


class VerifyGqlOutput(BaseModel):
  input_gql: str
  made_change: bool
  explanation: str
  verified_gql: str


verify_gql_output_parser = PydanticOutputParser(output_cls=VerifyGqlOutput)

GQL_VERIFY_PROMPT = PromptTemplate(
    template=DEFAULT_GQL_VERIFY_TEMPLATE,
)
GQL_VERIFY_PROMPT.output_parser = verify_gql_output_parser

GQL_FIX_PROMPT = PromptTemplate(
    template=DEFAULT_GQL_FIX_TEMPLATE,
)

DEFAULT_GQL_SUMMARY_TEMPLATE = PromptTemplate(
    template=DEFAULT_SUMMARY_TEMPLATE,
)


class SpannerGraphTextToGQLRetriever(BasePGRetriever):
  """A retriever that translates natural language queries to GQL and queries SpannerGraphStore."""

  def __init__(
      self,
      graph_store: SpannerPropertyGraphStore,
      llm: Optional[LLM] = None,
      text_to_gql_prompt: Optional[PromptTemplate] = None,
      response_template: Optional[str] = None,
      gql_validator: Optional[Callable[[str], bool]] = None,
      include_raw_response_as_metadata: Optional[bool] = False,
      max_gql_fix_retries: Optional[int] = 1,
      verify_gql: Optional[bool] = True,
      summarize_response: Optional[bool] = False,
      summarization_template: Optional[Union[PromptTemplate, str]] = None,
      **kwargs,
  ) -> None:
    """Initializes the SpannerGraphTextToGQLRetriever.

    Args:
      graph_store: The SpannerPropertyGraphStore to query.
      llm: The LLM to use.
      text_to_gql_prompt: The prompt to use for generating the GQL query.
      response_template: The template to use for formatting the response.
      gql_validator: A function to validate the GQL query.
      include_raw_response_as_metadata: Whether to include the raw response as
        metadata.
      max_gql_fix_retries: The maximum number of retries for fixing the GQL
        query.
      verify_gql: Whether to verify the GQL query.
      summarize_response: Whether to summarize the response.
      summarization_template: The template to use for summarizing the response.
      **kwargs: Additional keyword arguments.

    Raises:
      ValueError: If the graph store does not support structured queries or if
        the LLM is not provided.
    """
    if not graph_store.supports_structured_queries:
      raise ValueError("The provided graph store does not support GQL queries.")

    self.graph_store = graph_store
    self.llm = llm or Settings.llm
    if self.llm is None:
      raise ValueError("`llm` cannot be none")

    self.text_to_gql_prompt = (
        GQL_GENERATION_PROMPT
        if text_to_gql_prompt is None
        else text_to_gql_prompt
    )

    self.gql_validator = gql_validator
    self.include_raw_response_as_metadata = include_raw_response_as_metadata
    self.max_gql_fix_retries = max_gql_fix_retries
    self.verify_gql = verify_gql
    self.summarize_response = summarize_response
    self.summarization_template = (
        summarization_template or DEFAULT_SUMMARY_TEMPLATE
    )
    super().__init__(
        graph_store=graph_store, include_text=False, include_properties=False
    )

  def _parse_generated_gql(self, gql_query: str) -> str:
    if self.gql_validator is not None:
      return self.gql_validator(gql_query)
    return gql_query

  def execute_query(self, gql_query: str) -> List[Any]:
    responses = self.graph_store.structured_query(gql_query)
    return responses

  def execute_gql_query_with_retries(
      self, gql_query: str, question: str
  ) -> tuple[str, List[Any]]:
    """Execute the gql query with retries.

    If any error, asks LLM to fix the query and retry.

    Args:
        gql_query: The GQL query to execute.
        question: The original question.

    Returns:
        A tuple containing the final GQL query and the list of responses.
    """
    retries = 0
    while retries <= self.max_gql_fix_retries:
      try:
        return gql_query, self.execute_query(gql_query)
      except Exception as e:
        fixed_gql_query = self.llm.predict(
            GQL_FIX_PROMPT,
            question=question,
            generated_gql=gql_query,
            err_msg=str(e),
            schema=self.graph_store.get_schema_str(),
        )
        gql_query = extract_gql(fixed_gql_query)
        gql_query = self._parse_generated_gql(gql_query)
      finally:
        retries += 1
    return "", []

  def retrieve_from_graph(
      self, query_bundle: schema.QueryBundle
  ) -> list[schema.NodeWithScore]:
    """Retrieve from graph.

    Args:
        query_bundle: The query bundle.

    Returns:
        A list of NodeWithScore objects.
    """

    schema_str = self._graph_store.get_schema_str()
    question = query_bundle.query_str
    generic_prompt = self.text_to_gql_prompt

    # 1. Generate gql query from natural language query using LLM
    response = self.llm.predict(
        generic_prompt,
        schema=schema_str,
        question=question,
    )
    gql_query = extract_gql(response)
    generated_gql = self._parse_generated_gql(gql_query)

    # 2. Verify gql query using LLM
    if self.verify_gql:
      verify_response = (
          self.llm.predict(
              GQL_VERIFY_PROMPT,
              question=question,
              generated_gql=generated_gql,
              schema=schema_str,
              format_instructions=GQL_VERIFY_PROMPT.output_parser.format_string,
          ),
      )
      output_parser = verify_gql_output_parser.parse(verify_response[0])
      verified_gql = fix_gql_syntax(output_parser.verified_gql)
    else:
      verified_gql = generated_gql

    final_gql = ""
    if verified_gql:
      final_gql, responses = self.execute_gql_query_with_retries(
          verified_gql, question
      )
      if not final_gql:
        return []
    else:
      responses = []

    if self.summarize_response:
      summarized_response = self.llm.predict(
          self.summarization_template,
          context=str(responses),
          question=final_gql,
      )
      node_text = summarized_response
    else:
      node_text = str(responses)

    return [
        NodeWithScore(
            node=TextNode(
                text=node_text,
                metadata=(
                    {"query": final_gql, "response": node_text}
                    if self.include_raw_response_as_metadata
                    else {}
                ),
            ),
            score=1.0,
        )
    ]

  async def aretrieve_from_graph(
      self, query_bundle: QueryBundle
  ) -> List[NodeWithScore]:
    return await self.retrieve_from_graph(query_bundle)


class SpannerGraphCustomRetriever(CustomPGRetriever):
  """Custom retriever with cohere reranking."""

  def init(
      self,
      ## vector context retriever params
      embed_model: Optional[BaseEmbedding] = None,
      vector_store: Optional[VectorStore] = None,
      similarity_top_k: int = 4,
      path_depth: int = 2,
      ## text-to-gql params
      llm: Optional[LLM] = None,
      text_to_gql_prompt: Optional[PromptTemplate] = None,
      response_template: Optional[str] = None,
      gql_validator: Optional[Callable[[str], bool]] = None,
      include_raw_response_as_metadata: Optional[bool] = False,
      max_gql_fix_retries: Optional[int] = 1,
      verify_gql: Optional[bool] = True,
      summarize_response: Optional[bool] = False,
      summarization_template: Optional[Union[PromptTemplate, str]] = None,
      ## cohere reranker params
      cohere_api_key: Optional[str] = None,
      cohere_top_n: int = 2,
      **kwargs: Any,
  ) -> None:
    """Initializes the custom retriever.

    Args:
      embed_model: The embedding model to use.
      vector_store: The vector store to use.
      similarity_top_k: The number of top nodes to retrieve.
      path_depth: The depth of the path to retrieve.
      llm: The LLM to use.
      text_to_gql_prompt: The prompt to use for generating the GQL query.
      response_template: The template to use for formatting the response.
      gql_validator: A function to validate the GQL query.
      include_raw_response_as_metadata: Whether to include the raw response as
        metadata.
      max_gql_fix_retries: The maximum number of retries for fixing the GQL
        query.
      verify_gql: Whether to verify the GQL query.
      summarize_response: Whether to summarize the response.
      summarization_template: The template to use for summarizing the response.
      cohere_api_key: The Cohere API key.
      cohere_top_n: The number of top nodes to return.
      **kwargs: Additional keyword arguments.
    """
    self.vector_retriever = VectorContextRetriever(
        graph_store=self._graph_store,
        include_text=self.include_text,
        embed_model=embed_model,
        vector_store=vector_store,
        similarity_top_k=similarity_top_k,
        path_depth=path_depth,
    )

    self.nl_to_gql_retriever = SpannerGraphTextToGQLRetriever(
        graph_store=self._graph_store,
        llm=llm,
        text_to_gql_prompt=text_to_gql_prompt,
        response_template=response_template,
        gql_validator=gql_validator,
        include_raw_response_as_metadata=include_raw_response_as_metadata,
        max_gql_fix_retries=max_gql_fix_retries,
        verify_gql=verify_gql,
        summarize_response=summarize_response,
        summarization_template=summarization_template,
    )
    self.reranker = LLMRerank(choice_batch_size=5, top_n=cohere_top_n)

  def custom_retrieve(self, query_str: str) -> str:
    """Custom retrieve.

    Args:
        query_str: The query string.

    Returns:
        The final text response.
    """
    nodes_1 = self.vector_retriever.retrieve(query_str)
    query_bundle = QueryBundle(query_str=query_str)
    nodes_2 = self.nl_to_gql_retriever.retrieve_from_graph(query_bundle)
    reranked_nodes = self.reranker.postprocess_nodes(
        nodes_1 + nodes_2, query_bundle
    )

    final_text = "\n\n".join(
        [n.get_content(metadata_mode="llm") for n in reranked_nodes]
    )

    return final_text
