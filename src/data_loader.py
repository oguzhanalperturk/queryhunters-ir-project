import os
import re
import ir_datasets
import pandas as pd

os.environ["PYTHONIOENCODING"] = "utf-8"


def load_dataset(dataset_name: str):
    return ir_datasets.load(dataset_name)


def clean_query(text: str) -> str:
    cleaned = re.sub(r"[^\w\s]", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def load_queries(dataset, max_queries: int | None = None) -> pd.DataFrame:
    queries = []

    for query in dataset.queries_iter():
        queries.append({
            "qid": query.query_id,
            "query": clean_query(query.text)
        })

        if max_queries is not None and len(queries) >= max_queries:
            break

    return pd.DataFrame(queries)


def sample_queries(
    queries: pd.DataFrame,
    n: int | None = 1000,
    seed: int = 42
) -> pd.DataFrame:
    if n is None or len(queries) <= n:
        return queries.reset_index(drop=True)

    return queries.sample(n=n, random_state=seed).reset_index(drop=True)


def load_qrels(dataset) -> pd.DataFrame:
    qrels = []

    for qrel in dataset.qrels_iter():
        qrels.append({
            "qid": qrel.query_id,
            "docno": qrel.doc_id,
            "label": qrel.relevance
        })

    return pd.DataFrame(qrels)


def iter_documents(dataset, max_docs: int | None = None):
    count = 0

    for doc in dataset.docs_iter():
        yield {
            "docno": doc.doc_id,
            "text": doc.text
        }

        count += 1

        if max_docs is not None and count >= max_docs:
            break
