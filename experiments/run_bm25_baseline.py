import os
import yaml
import pyterrier as pt
from pyterrier.measures import *  # noqa: F401,F403  (provides RR, nDCG, R)

from src.data_loader import load_dataset, load_queries, sample_queries, load_qrels
from src.indexer import build_index, init_pyterrier
from src.retriever import create_bm25, run_retrieval


def load_config(path: str = "config.yaml"):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main():
    config = load_config()

    init_pyterrier()

    dataset_name = config["dataset"]["name"]
    index_path = config["dataset"]["index_path"]
    bm25_num_results = config["retrieval"]["bm25_num_results"]
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

    print("Running BM25 retrieval...")
    bm25 = create_bm25(index, num_results=bm25_num_results)
    results = run_retrieval(bm25, queries)

    print("Evaluating BM25...")
    evaluation = pt.Experiment(
        [bm25],
        queries,
        qrels,
        eval_metrics=[RR @ 10, nDCG @ 10, R @ 100],
        names=["BM25"]
    )

    print(evaluation)

    os.makedirs("results", exist_ok=True)
    results.to_csv("results/bm25_results.csv", index=False)
    evaluation.to_csv("results/bm25_evaluation.csv", index=False)

    print("BM25 baseline completed.")


if __name__ == "__main__":
    main()
