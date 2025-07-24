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

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from google.cloud import spanner  # type: ignore

from .type_utils import TypeUtility

MUTATION_BATCH_SIZE = 1000
DEFAULT_DDL_TIMEOUT = 300


def client_with_user_agent(
    client: Optional[spanner.Client], user_agent: str
) -> spanner.Client:
    if not client:
        client = spanner.Client()

    client_agent = client._client_info.user_agent
    if not client_agent:
        client._client_info.user_agent = user_agent
    elif user_agent not in client_agent:
        client._client_info.user_agent = " ".join([client_agent, user_agent])
    return client


class SpannerInterface(ABC):
    """Wrapper of Spanner APIs."""

    @abstractmethod
    def query(self, query: str, params: dict = {}) -> List[Dict[str, Any]]:
        """Runs a Spanner query.

        Args:
          query: query string;
          params: Spanner query params;
          param_types: Spanner param types.

        Returns:
          List[Dict[str, Any]]: query results.
        """
        raise NotImplementedError

    @abstractmethod
    def apply_ddls(self, ddls: List[str], options: Dict[str, Any] = {}) -> None:
        """Applies a list of schema modifications.

        Args:
          ddls: Spanner Schema DDLs.
        """
        raise NotImplementedError

    @abstractmethod
    def insert_or_update(
        self, table: str, columns: Tuple[str, ...], values: List[List[Any]]
    ) -> None:
        """Insert or update the table.

        Args:
          table: Spanner table name;
          columns: a tuple of column names;
          values: list of values.
        """
        raise NotImplementedError

    @abstractmethod
    def delete(self, table: str, keysets: List[List[Any]]) -> None:
        """Delete the table.

        Args:
          table: Spanner table name;
          keysets: list of keysets.
        """
        raise NotImplementedError


class SpannerImpl(SpannerInterface):
    """Implementation of SpannerInterface."""

    def __init__(
        self,
        instance_id: str,
        database_id: str,
        client: Optional[spanner.Client] = None,
    ):
        """Initializes the Spanner implementation.

        Args:
          instance_id: Google Cloud Spanner instance id;
          database_id: Google Cloud Spanner database id;
          client: an optional instance of Spanner client.
        """
        self.client = client or spanner.Client()
        self.instance = self.client.instance(instance_id)
        self.database = self.instance.database(database_id)

    def query(self, query: str, params: dict = {}) -> List[Dict[str, Any]]:
        param_types = {k: TypeUtility.value_to_param_type(v) for k, v in params.items()}
        with self.database.snapshot() as snapshot:
            rows = snapshot.execute_sql(query, params=params, param_types=param_types)
            return [
                {
                    column: value
                    for column, value in zip(
                        [column.name for column in rows.fields], row
                    )
                }
                for row in rows
            ]

    def apply_ddls(self, ddls: List[str], options: Dict[str, Any] = {}) -> None:
        if not ddls:
            return

        op = self.database.update_ddl(ddl_statements=ddls)
        print("Waiting for DDL operations to complete...")
        return op.result(options.get("timeout", DEFAULT_DDL_TIMEOUT))

    def insert_or_update(
        self, table: str, columns: Tuple[str, ...], values: List[List[Any]]
    ) -> None:
        for i in range(0, len(values), MUTATION_BATCH_SIZE):
            value_batch = values[i : i + MUTATION_BATCH_SIZE]
            with self.database.batch() as batch:
                batch.insert_or_update(table=table, columns=columns, values=value_batch)

    def delete(self, table: str, keysets: List[List[Any]]) -> None:
        for i in range(0, len(keysets), MUTATION_BATCH_SIZE):
            keysets_batch = keysets[i : i + MUTATION_BATCH_SIZE]
            with self.database.batch() as batch:
                batch.delete(table=table, keyset=spanner.KeySet(keys=keysets_batch))
