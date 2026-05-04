import os
import ir_datasets
import pandas as pd

os.environ["PYTHONIOENCODING"] = "utf-8"


def load_dataset(dataset_name: str):
    return ir_datasets.load(dataset_name)


def load_queries(dataset, max_queries: int | None = None) -> pd.DataFrame:
    queries = []

    for query in dataset.queries_iter():
        queries.append({
            "qid": query.query_id,
            "query": query.text
        })

        if max_queries is not None and len(queries) >= max_queries:
            break

    return pd.DataFrame(queries)


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


def iter_subset_documents_for_queries(
    dataset,
    selected_queries: pd.DataFrame,
    qrels: pd.DataFrame,
    max_distractor_docs: int = 50000
):
    selected_qids = set(selected_queries["qid"].astype(str))

    selected_qrels = qrels[qrels["qid"].astype(str).isin(selected_qids)]
    relevant_docnos = set(selected_qrels["docno"].astype(str))

    yielded_docnos = set()

    docs_store = dataset.docs_store()

    print(f"Relevant docs to force include: {len(relevant_docnos)}")

    for docno in relevant_docnos:
        try:
            doc = docs_store.get(docno)

            if doc is None:
                continue

            yielded_docnos.add(docno)

            yield {
                "docno": doc.doc_id,
                "text": doc.text
            }

        except Exception as e:
            print(f"Could not load relevant doc {docno}: {e}")

    distractor_count = 0

    for doc in dataset.docs_iter():
        if doc.doc_id in yielded_docnos:
            continue

        yield {
            "docno": doc.doc_id,
            "text": doc.text
        }

        distractor_count += 1

        if distractor_count >= max_distractor_docs:
            break

    print(f"Distractor docs included: {distractor_count}")