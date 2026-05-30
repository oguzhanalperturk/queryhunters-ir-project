import os
import yaml
import pyterrier as pt
from pyterrier.measures import *  # noqa: F401,F403  (provides RR, nDCG, R)

from src.data_loader import load_dataset, load_queries, sample_queries, load_qrels
from src.indexer import build_index, init_pyterrier
from src.retriever import create_bm25, create_bm25_rm3


def load_config(path: str = "config.yaml"):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main():
    config = load_config()

    init_pyterrier()

    dataset_name = config["dataset"]["name"]
    index_path = config["dataset"]["index_path"]
    bm25_num_results = config["retrieval"]["bm25_num_results"]

    fb_docs = config["query_expansion"]["fb_docs"]
    fb_terms = config["query_expansion"]["fb_terms"]
    fb_lambda = config["query_expansion"]["original_query_weight"]

    num_queries = config["evaluation"]["num_queries"]
    seed = config["evaluation"]["seed"]

    print(f"Using index path: {index_path}")

    dataset = load_dataset(dataset_name)

    print("Loading queries...")
    queries = load_queries(dataset, max_queries=None)
    queries = sample_queries(queries, n=num_queries, seed=seed)
    print(f"Evaluating on {len(queries)} queries.")

    print("Loading qrels...")
    qrels = load_qrels(dataset)

    print("Building/loading index (full corpus)...")
    index = build_index(dataset=dataset, index_path=index_path)

    print("Creating retrieval pipelines...")
    bm25 = create_bm25(index, num_results=bm25_num_results)

    bm25_rm3 = create_bm25_rm3(
        index,
        num_results=bm25_num_results,
        fb_docs=fb_docs,
        fb_terms=fb_terms,
        fb_lambda=fb_lambda
    )

    print("Running evaluation...")
    evaluation = pt.Experiment(
        [bm25, bm25_rm3],
        queries,
        qrels,
        eval_metrics=[RR @ 10, nDCG @ 10, R @ 100],
        names=["BM25", "BM25+RM3"],
        baseline=0,
        correction="bonferroni"
    )

    print(evaluation)

    os.makedirs("results", exist_ok=True)
    evaluation.to_csv("results/bm25_rm3_evaluation.csv", index=False)

    print("BM25 + RM3 experiment completed.")


if __name__ == "__main__":
    main()
