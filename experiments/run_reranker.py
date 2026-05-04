import os
import yaml
import pyterrier as pt

from src.data_loader import load_dataset, load_queries, load_qrels
from src.indexer import build_index, init_pyterrier
from src.retriever import create_bm25, run_retrieval
from src.reranker import CrossEncoderReranker


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

    model_name = config["reranking"]["model_name"]
    batch_size = config["reranking"]["batch_size"]

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

    print("Running BM25 retrieval...")
    bm25 = create_bm25(index, num_results=bm25_num_results)
    bm25_results = run_retrieval(bm25, queries)

    print("Running Cross-Encoder reranking...")
    reranker = CrossEncoderReranker(
        model_name=model_name,
        batch_size=batch_size
    )

    reranked_results = reranker.rerank(bm25_results, queries)

    print("Evaluating BM25 and BM25+Reranker...")
    evaluation = pt.Experiment(
        [bm25_results, reranked_results],
        queries,
        qrels,
        eval_metrics=["recip_rank", "ndcg_cut_10", "recall_100"],
        names=["BM25", "BM25+Reranker"]
    )

    print(evaluation)

    os.makedirs("results", exist_ok=True)
    bm25_results.to_csv("results/bm25_candidates.csv", index=False)
    reranked_results.to_csv("results/bm25_reranked_results.csv", index=False)
    evaluation.to_csv("results/bm25_reranker_evaluation.csv", index=False)

    print("BM25 + neural reranker experiment completed.")


if __name__ == "__main__":
    main()