# Spanner for LlamaIndex

[![preview](https://img.shields.io/badge/support-preview-orange.svg)](https://cloud.google.com/products#product-launch-stages)
[![pypi](https://img.shields.io/pypi/v/llama-index-spanner.svg)](https://pypi.org/project/llama-index-spanner/)
[![versions](https://img.shields.io/pypi/pyversions/llama-index-spanner.svg)](https://pypi.org/project/llama-index-spanner/)

  * [Client Library Documentation](https://cloud.google.com/python/docs/reference/llama-index-spanner/latest)
  * [Product Documentation](https://cloud.google.com/spanner)

## Quick Start

In order to use this library, you first need to go through the following steps:

1.  [Select or create a Cloud Platform project.](https://console.cloud.google.com/project)
2.  [Enable billing for your project.](https://cloud.google.com/billing/docs/how-to/modify-project#enable_billing_for_a_project)
3.  [Enable the Google Cloud Spanner API.](https://console.cloud.google.com/flows/enableapi?apiid=spanner.googleapis.com)
4.  [Setup Authentication.](https://googleapis.dev/python/google-api-core/latest/auth.html)

### Installation

Install this library in a [virtualenv](https://virtualenv.pypa.io/en/latest/) using pip. [virtualenv](https://virtualenv.pypa.io/en/latest/) is a tool to create isolated Python environments. The basic problem it addresses is one of dependencies and versions, and indirectly permissions.

With [virtualenv](https://virtualenv.pypa.io/en/latest/), it’s possible to install this library without needing system install permissions, and without clashing with the installed system dependencies.

#### Supported Python Versions

Python \>= 3.9

#### Mac/Linux

```console
pip install virtualenv
virtualenv <your-env>
source <your-env>/bin/activate
<your-env>/bin/pip install llama-index-spanner
```

#### Windows

```console
pip install virtualenv
virtualenv <your-env>
<your-env>\Scripts\activate
<your-env>\Scripts\pip.exe install llama-index-spanner
```

 ### Spanner Property Graph Store Usage

Use `SpannerPropertyGraphStore` to store nodes and edges extracted from documents.

```python
from llama_index_spanner import SpannerPropertyGraphStore

graph_store = SpannerPropertyGraphStore(
    instance_id="my-instance",
    database_id="my-database",
    graph_name="my_graph",
)
```

See the full [Spanner Graph Store](https://github.com/googleapis/llama-index-spanner-python/blob/main/docs/property_graph_store.ipynb) tutorial.

### Spanner Graph Retrievers Usage

Use `SpannerGraphTextToGQLRetriever` to translate natural language question to GQL and query SpannerPropertyGraphStore.

```python
from llama_index_spanner import (
    SpannerPropertyGraphStore,
    SpannerGraphTextToGQLRetriever,
)
from llama_index.llms.google_genai import GoogleGenAI

graph_store = SpannerPropertyGraphStore(
    instance_id="my-instance",
    database_id="my-database",
    graph_name="my_graph",
)
llm = GoogleGenAI(
    model="gemini-2.0-flash",
)
retriever = SpannerGraphTextToGQLRetriever(graph_store=graph_store, llm=llm)
retriever.retrieve("Where does Elias Thorne's sibling live?")

```

Use `SpannerGraphCustomRetriever` to query your `SpannerPropertyGraphStore` with a hybrid retrieval approach, combining natural language-to-GQL translation with vector search capabilities.

The Vector Context Retriever uses semantic vector search to surface relevant contexts, even when they aren't explicitly structured in the graph. The Text2GQL Retriever translates natural language into GQL queries to extract precise relationships and attributes from the property graph.

The results from both retrievers are evaluated using an LLM-based reranker, which scores them based on relevance, completeness, and contextual fit with the original query.

This hybrid + rerank pipeline enables the system to handle everything from vague, open-ended questions to structured, entity-specific queries, delivering high-confidence, context-aware responses.


```python
from llama_index_spanner import SpannerPropertyGraphStore, SpannerGraphCustomRetriever
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

graph_store = SpannerPropertyGraphStore(
    instance_id="my-instance",
    database_id="my-database",
    graph_name="my_graph",
)
llm = GoogleGenAI(
    model="gemini-2.0-flash",
)
embed_model = GoogleGenAIEmbedding(model_name="text-embedding-004")
retriever = SpannerGraphCustomRetriever(
    similarity_top_k=4,
    path_depth=2,
    graph_store=graph_store,
    llm_text_to_gql=llm,
    embed_model=embed_model,
    llmranker_top_n=3,
)
retriever.retriever("Who lives in desert?")
```

See the full [Spanner Graph Retrievers](https://github.com/googleapis/llama-index-spanner-python/blob/main/docs/graph_retriever.ipynb) tutorial.

## Contributing

Contributions to this library are always welcome and highly encouraged.

See [CONTRIBUTING](https://github.com/googleapis/llama-index-spanner-python/blob/main/CONTRIBUTING.md) for more information how to get started.

Please note that this project is released with a Contributor Code of Conduct. By participating in
this project you agree to abide by its terms. See [Code of Conduct](https://github.com/googleapis/llama-index-spanner-python/blob/main/CODE_OF_CONDUCT.md) for more
information.

## License

Apache 2.0 - See [LICENSE](https://github.com/googleapis/llama-index-spanner-python/blob/main/LICENSE) for more information.