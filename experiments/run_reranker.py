import os
import yaml
import pyterrier as pt
from pyterrier.measures import *  # noqa: F401,F403  (provides RR, nDCG, R)

from src.data_loader import load_dataset, load_queries, sample_queries, load_qrels
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
    bm25_num_results = config["retrieval"]["bm25_num_results"]

    model_name = config["reranking"]["model_name"]
    batch_size = config["reranking"]["batch_size"]
    rerank_top_k = config["reranking"]["rerank_top_k"]

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
    bm25_results = run_retrieval(bm25, queries)

    print("Running Cross-Encoder reranking...")
    reranker = CrossEncoderReranker(
        model_name=model_name,
        batch_size=batch_size
    )

    reranked_results = reranker.rerank(bm25_results, queries, top_k=rerank_top_k)

    print("Evaluating BM25 and BM25+Reranker...")
    evaluation = pt.Experiment(
        [bm25_results, reranked_results],
        queries,
        qrels,
        eval_metrics=[RR @ 10, nDCG @ 10, R @ 100],
        names=["BM25", "BM25+Reranker"],
        baseline=0,
        correction="bonferroni"
    )

    print(evaluation)

    os.makedirs("results", exist_ok=True)
    bm25_results.to_csv("results/bm25_candidates.csv", index=False)
    reranked_results.to_csv("results/bm25_reranked_results.csv", index=False)
    evaluation.to_csv("results/bm25_reranker_evaluation.csv", index=False)

    print("BM25 + neural reranker experiment completed.")


if __name__ == "__main__":
    main()
