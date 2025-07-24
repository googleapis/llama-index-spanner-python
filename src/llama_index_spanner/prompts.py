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

GQL_EXAMPLES = """
The following query in backtick matches all persons in the graph
whose birthday is before 1990-01-10 and
returns their name and birthday.
```
MATCH (p:Person WHERE p.birthday < '1990-01-10')
RETURN p.name as name, p.birthday as birthday;
```

The following query in backtick finds the owner of the account with the most
incoming transfers by chaining multiple graph linear statements together.
```
MATCH (:Account)-[:Transfers]->(account:Account)
RETURN account, COUNT(*) AS num_incoming_transfers
GROUP BY account
ORDER BY num_incoming_transfers DESC
LIMIT 1

NEXT

MATCH (account:Account)<-[:Owns]-(owner:Person)
RETURN account.id AS account_id, owner.name AS owner_name, num_incoming_transfers;
```

The following query finds all the destination accounts one to three transfers
away from a source Account with id equal to 7.
```
MATCH (src:Account {{id: 7}})-[e:Transfers]->{{1, 3}}(dst:Account)
RETURN src.id AS src_account_id, dst.id AS dst_account_id;
```
Carefully note the syntax in the example above for path quantification,
that it is `[e:Transfers]->{{1, 3}}` and NOT `[e:Transfers*1..3]->`
"""

DEFAULT_GQL_TEMPLATE_PART0 = """
Create an Spanner Graph GQL query for the question using the schema.
{gql_examples}
"""

DEFAULT_GQL_TEMPLATE_PART1 = """
Instructions:
Mention the name of the graph at the beginning.
* **No Graph Name Prefix:** Do *not* include the graph name at the start of the query (e.g., `GRAPH FinGraph`). This is handled automatically.
Use only nodes and edge types, and properties included in the schema.
Do not use any node and edge type, or properties not included in the schema.
Always alias RETURN values.

Question: {question}
Schema: {schema}

Note:
Do not include any explanations or apologies.
Do not prefix query with `gql`
Do not prefix query with gql
Do not include any backticks.
Do not include the graph name at the start of the query.
Output only the query statement.
Do not output any query that tries to modify or delete data.
"""

DEFAULT_SPANNER_GQL_TEMPLATE = (
    DEFAULT_GQL_TEMPLATE_PART0.format(gql_examples=GQL_EXAMPLES)
    + DEFAULT_GQL_TEMPLATE_PART1
)

VERIFY_EXAMPLES = """
Examples:
1.
question: Which movie has own the Oscar award in 1996?
generated_gql:
  GRAPH moviedb
  MATCH (m:movie)-[:own_award]->(a:award {{name:"Oscar", year:1996}})
  RETURN m.name

graph_schema:
{{
"Edges": {{
    "produced_by": "From movie nodes to producer nodes",
    "acts": "From actor nodes to movie nodes",
    "has_coacted_with": "From actor nodes to actor nodes",
    "own_award": "From actor nodes to award nodes"
  }}
}}

The verified gql fixes the missing node 'actor'
  MATCH (m:movie)<-[:acts]-(a:actor)-[:own_award]->(a:award {{name:"Oscar", year:1996}})
  RETURN m.name

2.
question: Which movies have been produced by production house ABC Movies?
generated_gql:
  GRAPH moviedb
  MATCH (p:producer {{name:"ABC Movies"}})-[:produced_by]->(m:movie)
  RETURN p.name

graph_schema:
{{
"Edges": {{
    "produced_by": "From movie nodes to producer nodes",
    "acts": "From actor nodes to movie nodes",
    "references": "From movie nodes to movie nodes",
    "own_award": "From actor nodes to award nodes"
  }}
}}

The verified gql fixes the edge direction:
  GRAPH moviedb
  MATCH (p:producer {{name:"ABC Movies"}})<-[:produced_by]-(m:movie)
  RETURN m.name

3.
question: Which movie references the movie "XYZ" via at most 3 hops ?
graph_schema:
{{
"Edges": {{
    "produced_by": "From movie nodes to producer nodes",
    "acts": "From actor nodes to movie nodes",
    "references": "From movie nodes to movie nodes",
    "own_award": "From actor nodes to award nodes"
  }}
}}

generated_gql:
  GRAPH moviedb
  MATCH (m:movie)-[:references*1..3]->(:movie {{name="XYZ"}})
  RETURN m.name

The path quantification syntax [:references*1..3] is wrong.
The verified gql fixes the path quantification syntax:
  GRAPH moviedb
  MATCH (m:movie)-[:references]->{{1, 3}}(:movie {{name="XYZ"}})
  RETURN m.name
"""

DEFAULT_GQL_VERIFY_TEMPLATE_PART0 = """
Given a natural language question, Spanner Graph GQL graph query and a graph schema,
validate the query.

{verify_examples}
"""

DEFAULT_GQL_VERIFY_TEMPLATE_PART1 = """
Instructions:
* **No Graph Name Prefix:** Do *not* include the graph name at the start of the query (e.g., `GRAPH FinGraph`). This is handled automatically.
Add missing nodes and edges in the query if required.
Fix the path quantification syntax if required.
Carefully check the syntax.
Fix the query if required. There could be more than one correction.
Optimize if possible.
Do not make changes if not required.
Think in steps. Add the explanation in the output.

Question : {question}
Input gql: {generated_gql}
Schema: {graph_schema}

{format_instructions}
"""

DEFAULT_GQL_VERIFY_TEMPLATE = (
    DEFAULT_GQL_VERIFY_TEMPLATE_PART0.format(verify_examples=VERIFY_EXAMPLES)
    + DEFAULT_GQL_VERIFY_TEMPLATE_PART1
)

DEFAULT_GQL_FIX_TEMPLATE_PART0 = """
We generated a Spanner Graph GQL query to answer a natural language question.
Question: {question}
However the generated Spanner Graph GQL query is not valid.  ```
Input gql: {generated_gql}
```
The error obtained when executing the query is
```
{err_msg}
```
Return a modified version of the query which addresses the error message.
Do not generate the same query as the input gql.
"""

DEFAULT_GQL_FIX_TEMPLATE_PART1 = """
Examples of correct query :
{gql_examples}
"""

DEFAULT_GQL_FIX_TEMPLATE_PART2 = """
Instructions:
* **No Graph Name Prefix:** Do *not* include the graph name at the start of the query (e.g., `GRAPH FinGraph`). This is handled automatically.
Use only nodes and edge types, and properties included in the schema.
Do not use any node and edge type, or properties not included in the schema.
Do not generate the same query as the input gql.
Do not include the graph name at the start of the query.
Schema: {schema}

Note:
Do not include any explanations or apologies.
Do not prefix query with `gql`
Do not include any backticks.
Start with GRAPH <graphname>
Output only the query statement.
Do not output any query that tries to modify or delete data.
"""

DEFAULT_GQL_FIX_TEMPLATE = (
    DEFAULT_GQL_FIX_TEMPLATE_PART0
    + DEFAULT_GQL_FIX_TEMPLATE_PART1.format(gql_examples=GQL_EXAMPLES)
    + DEFAULT_GQL_FIX_TEMPLATE_PART2
)


SUMMARY_EXAMPLES = """
Here is an example:

Question: How many miles is the flight between the ANC and SEA airports?
Information:
[{"flight_dist": 1440}]
Helpful Answer:
It is 1440 miles to fly between the ANC and SEA airports.
"""

DEFAULT_SUMMARY_TEMPLATE_PART0 = """
You are an assistant that helps to form nice and human understandable answers.
The information part contains the provided information you must use to construct an answer.
The provided information is authoritative, never doubt it or try to use your internal knowledge to correct it.
If the provided information is empty, say that you don't know the answer.
Make the answer sound as a response to the question. Do not mention that you based the result on the given information.

{summary_examples}

"""

DEFAULT_SUMMARY_TEMPLATE_PART1 = """
Follow this example when generating answers.
Question:
{question}
Information:
{context}
Helpful Answer:
"""

DEFAULT_SUMMARY_TEMPLATE = (
    DEFAULT_SUMMARY_TEMPLATE_PART0.format(summary_examples=SUMMARY_EXAMPLES)
    + DEFAULT_SUMMARY_TEMPLATE_PART1
)

SCORING_EXAMPLES = """
Examples:

1.
Question: How many miles is the flight between the ANC and SEA airports?
Response:
It is 1440 miles to fly between the ANC and SEA airports.
Score:
1.0

2.
Question: Which movies have been produced by production house ABC Movies?
Response:
I don't know the answer
Score:
0.0

"""

DEFAULT_SCORING_TEMPLATE_PART0 = """
You are an assistant tasked with evaluating the quality of answers.
You will be provided with an original user query, and the response generated from retrieved information.

Evaluate the response based on the following criteria:
1.  **Groundedness:** Is the response somewhere supported by the provided context? Does it introduce any information not present in the context (hallucinations)?
2.  **Relevance:** Does the response answer the original user query?
3.  **Clarity:** Is the response easy to understand and well-written?
4.  **Confidence/Uncertainty:** Does the response explicitly state that it doesn't know the answer, or show significant uncertainty?

Rate the response on a scale of 0 to 1, where 0 is completely unusable (e.g., explicitly says "I don't know" or is fully irrelevant/hallucinated), and 1 is a perfect, accurate, grounded, and relevant answer.

If the response explicitly states "I don't know", "I cannot answer", "information not available", or similar, it should be rated 0.0.

Provide ONLY a single floating-point number as your output.

{scoring_examples}

"""

DEFAULT_SCORING_TEMPLATE_PART1 = """
Follow this example when generating answers.
Question:
{question}
Response:
{retrieved_context}
Score:
"""

DEFAULT_SCORING_TEMPLATE = (
    DEFAULT_SCORING_TEMPLATE_PART0.format(scoring_examples=SCORING_EXAMPLES)
    + DEFAULT_SCORING_TEMPLATE_PART1
)

SYNTHESIS_EXAMPLES = """
Examples:
1.
Question: Who are the alumni of the XYZ University ?
Final_Reranked_Response:
Person1 -> ALUMNIOF -> XYZ University

Person2 -> ALUMNIOF -> XYZ University

The people who are alumni of XYZ University are Person1, and Person2

Synthesized_Output: 
The names of the people who are alumni of XYZ University are: Person1, and Person2.

2. 
Question: Who are the alumni of the XYZ University ?
Final_Reranked_Response:

Eric -> ALUMNIOF -> XYZ University

[{'alumni_name': 'Eric'}, {'alumni_name': 'Amar'}]

Synthesized_Output: 
The names of the people who are alumni of XYZ University are: Eric, and Amar.

3. 
Question: Who are the people working in Google ?
Final_Reranked_Response:
Axem -> WORKED_IN -> Google

Axem -> GRADUATED_FROM -> Texas University

I don't know the answer.

Synthesized_Output: 
Axem, who graduated from Texas University is working in Google. 

4.
Question: I am a beginner at coding. Help me with the resources.
Final_Reranked_Response:
Coding -> LEARN_FROM -> DSA Data Book

Coding -> PRACTICE_ON -> Google Coders

Synthesized_Output: 
Since you are a beginner at Coding, You can learn coding from DSA Data Book and then practice on Google Coders platform.

5. 
Question: Find products that were reviewed by customers who bought an electronic item ?
Final_Reranked_Response:

[{'reviewed_product': 'Chromebook'}, {'reviewed_product': 'Google Pixel 9'}]

Synthesized_Output: 
Chromebook and Google Pixel 9 were some of the products that were reviewed by customers who bought an electronic item.

6.
Question: Which movie has won the Oscar award in 1996?
Final_Reranked_Response:
I don't know the answer.

Synthesized_Output: 
Information is not enough to answer the movies which has Oscar Award in 1996.
"""

DEFAULT_SYNTHESIS_TEMPLATE_PART0 = """
You are an intelligent assistant. Your task is to answer the user's question.

**Strictly adhere to the following rules:**
1.  **Use ONLY the provided "Information provided" below.** Do not use any outside knowledge.
2.  **Synthesize a concise, coherent, and accurate response.**
3.  **Crucially: Avoid redundancy and repetition.** If the same fact or piece of information is present multiple times (even if phrased differently), state it only once in your answer.
4.  **Prioritize facts from information entries with higher confidence scores.** If information is conflicting, resolve the conflict by prioritizing information with higher confidence scores.
5.  **Frame your response as a direct answer to my question, incorporating the key context of my query.**
6.  **If any provided "Information" entry explicitly states "I don't know" or similar uncertainty, but other "Information" entries provide relevant details, prioritize the information that provides details.** Do NOT include phrases indicating uncertainty or "I don't know" from individual information sources if a clear answer can be formed from other available information.
7.  **If the provided "Information provided" is NOT sufficient to fully answer the "Original Question", then you MUST state "I do not have enough information to answer that question based on the provided context."** Do not provide a partial answer or make up information.
8.  Do not include any irrelevant or tangential details.

{synthesis_examples}

"""

DEFAULT_SYNTHESIS_TEMPLATE_PART1 = """
Follow this example when generating answers.
Question:
{question}
Final_Reranked_Response:
{retrieved_response}
Synthesized_Output:
"""

DEFAULT_SYNTHESIS_TEMPLATE = (
    DEFAULT_SYNTHESIS_TEMPLATE_PART0.format(synthesis_examples=SYNTHESIS_EXAMPLES)
    + DEFAULT_SYNTHESIS_TEMPLATE_PART1
)
