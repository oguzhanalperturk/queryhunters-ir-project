import os
import yaml
import pyterrier as pt

from src.data_loader import load_dataset, load_queries, load_qrels
from src.indexer import build_index, init_pyterrier
from src.retriever import create_bm25, create_bm25_rm3, run_retrieval
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

    fb_docs = config["query_expansion"]["fb_docs"]
    fb_terms = config["query_expansion"]["fb_terms"]

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

    print("Creating BM25 pipeline...")
    bm25 = create_bm25(index, num_results=bm25_num_results)

    print("Creating BM25 + RM3 pipeline...")
    bm25_rm3 = create_bm25_rm3(
        index=index,
        num_results=bm25_num_results,
        fb_docs=fb_docs,
        fb_terms=fb_terms
    )

    print("Running BM25 retrieval...")
    bm25_results = run_retrieval(bm25, queries)

    print("Running BM25 + RM3 retrieval...")
    bm25_rm3_results = run_retrieval(bm25_rm3, queries)

    print("Running Cross-Encoder reranking over BM25 results...")
    reranker = CrossEncoderReranker(
        model_name=model_name,
        batch_size=batch_size
    )

    bm25_reranked_results = reranker.rerank(bm25_results, queries)

    print("Running Cross-Encoder reranking over BM25 + RM3 results...")
    full_pipeline_results = reranker.rerank(bm25_rm3_results, queries)

    print("Evaluating all systems...")
    evaluation = pt.Experiment(
        [
            bm25_results,
            bm25_rm3_results,
            bm25_reranked_results,
            full_pipeline_results
        ],
        queries,
        qrels,
        eval_metrics=["recip_rank", "ndcg_cut_10", "recall_100"],
        names=[
            "BM25",
            "BM25+RM3",
            "BM25+Reranker",
            "BM25+RM3+Reranker"
        ]
    )

    print(evaluation)

    os.makedirs("results", exist_ok=True)

    bm25_results.to_csv("results/bm25_results.csv", index=False)
    bm25_rm3_results.to_csv("results/bm25_rm3_results.csv", index=False)
    bm25_reranked_results.to_csv("results/bm25_reranked_results.csv", index=False)
    full_pipeline_results.to_csv("results/full_pipeline_results.csv", index=False)
    evaluation.to_csv("results/full_pipeline_evaluation.csv", index=False)

    print("Full pipeline experiment completed.")


if __name__ == "__main__":
    main()