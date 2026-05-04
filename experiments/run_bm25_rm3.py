import os
import yaml
import pyterrier as pt

from src.data_loader import load_dataset, load_queries, load_qrels
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
    max_dev_queries = config["dataset"]["max_dev_queries"]

    bm25_num_results = config["retrieval"]["bm25_num_results"]

    fb_docs = config["query_expansion"]["fb_docs"]
    fb_terms = config["query_expansion"]["fb_terms"]

    print(f"Using index path: {index_path}")

    dataset = load_dataset(dataset_name)

    print("Loading queries...")
    queries = load_queries(dataset, max_queries=max_dev_queries)

    print("Loading qrels...")
    qrels = load_qrels(dataset)

    print("Building/loading index...")
    index = build_index(
        dataset=dataset,
        index_path=index_path,
        selected_queries=queries,
        qrels=qrels,
        max_distractor_docs=50000
    )

    print("Creating retrieval pipelines...")
    bm25 = create_bm25(index, num_results=bm25_num_results)

    bm25_rm3 = create_bm25_rm3(
        index,
        num_results=bm25_num_results,
        fb_docs=fb_docs,
        fb_terms=fb_terms
    )

    print("Running evaluation...")
    evaluation = pt.Experiment(
        [bm25, bm25_rm3],
        queries,
        qrels,
        eval_metrics=["recip_rank", "ndcg_cut_10", "recall_100"],
        names=["BM25", "BM25+RM3"]
    )

    print(evaluation)

    os.makedirs("results", exist_ok=True)
    evaluation.to_csv("results/bm25_rm3_evaluation.csv", index=False)

    print("BM25 + RM3 experiment completed.")


if __name__ == "__main__":
    main()