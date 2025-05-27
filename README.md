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

<!-- ### Spanner Property Graph Store Usage

Use `SpannerPropertyGraphStore` to store nodes and edges extracted from documents.

```python
from llama_index_spanner import SpannerPropertyGraphStore

graph = SpannerPropertyGraphStore(
    instance_id="my-instance",
    database_id="my-database",
    graph_name="my_graph",
)
```

See the full [Spanner Graph Store](https://github.com/googleapis/langchain-google-spanner-python/blob/main/docs/graph_store.ipynb) tutorial.

### Spanner Graph Retrievers Usage

Use `SpannerGraphTextToGQLRetriever` to translate natural language question to GQL and query SpannerGraphStore.

```python
from langchain_google_spanner import SpannerGraphStore, SpannerGraphTextToGQLRetriever
from langchain_google_vertexai import ChatVertexAI

graph = SpannerGraphStore(
    instance_id="my-instance",
    database_id="my-database",
    graph_name="my_graph",
)
llm = ChatVertexAI()
retriever = SpannerGraphTextToGQLRetriever.from_params(
    graph_store=graph,
    llm=llm
)
retriever.invoke("Where does Elias Thorne's sibling live?")
```

Use `SpannerGraphVectorContextRetriever` to perform vector search on embeddings that are stored in the nodes in a SpannerGraphStore. If expand\_by\_hops is provided, the nodes and edges at a distance upto the expand\_by\_hops from the nodes found in the vector search will also be returned.

```python
from langchain_google_spanner import SpannerGraphStore, SpannerGraphVectorContextRetriever
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings

graph = SpannerGraphStore(
    instance_id="my-instance",
    database_id="my-database",
    graph_name="my_graph",
)
embedding_service = VertexAIEmbeddings(model_name="text-embedding-004")
retriever = SpannerGraphVectorContextRetriever.from_params(
            graph_store=graph,
            embedding_service=embedding_service,
            label_expr="Person",
            embeddings_column="embeddings",
            top_k=1,
            expand_by_hops=1,
        )
retriever.invoke("Who lives in desert?")
``` -->

## Contributing

Contributions to this library are always welcome and highly encouraged.

See [CONTRIBUTING](CONTRIBUTING.md) for more information how to get started.

Please note that this project is released with a Contributor Code of Conduct. By participating in
this project you agree to abide by its terms. See [Code of Conduct](CODE_OF_CONDUCT.md) for more
information.

## License

Apache 2.0 - See [LICENSE](LICENSE) for more information.