import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
import pandas as pd
import streamlit as st
import pyterrier as pt

from src.data_loader import load_dataset, load_qrels, clean_query
from src.indexer import init_pyterrier
from src.retriever import create_bm25, create_bm25_rm3, run_retrieval
from src.reranker import CrossEncoderReranker


@st.cache_resource
def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


@st.cache_resource
def load_index(index_path):
    init_pyterrier()
    return pt.IndexFactory.of(os.path.abspath(index_path))


@st.cache_resource
def load_reranker(model_name, batch_size):
    return CrossEncoderReranker(
        model_name=model_name,
        batch_size=batch_size
    )


@st.cache_data
def load_qrels_cached(dataset_name):
    dataset = load_dataset(dataset_name)
    return load_qrels(dataset)


@st.cache_data
def load_dev_queries_cached(dataset_name, max_queries=500, pinned_query_ids=None):
    pinned_query_ids = [str(q) for q in (pinned_query_ids or [])]
    pinned_set = set(pinned_query_ids)

    collected = {}
    pinned_found = {}

    dataset = load_dataset(dataset_name)

    for query in dataset.queries_iter():
        qid = str(query.query_id)
        row = {"qid": qid, "query": clean_query(query.text)}

        if qid in pinned_set:
            pinned_found[qid] = row

        if len(collected) < max_queries:
            collected[qid] = row

        if len(collected) >= max_queries and len(pinned_found) == len(pinned_set):
            break

    ordered = []
    for qid in pinned_query_ids:
        if qid in pinned_found:
            ordered.append(pinned_found[qid])

    for qid, row in collected.items():
        if qid not in pinned_set:
            ordered.append(row)

    return pd.DataFrame(ordered)


def make_query_df(query_text: str):
    return pd.DataFrame([
        {
            "qid": "demo_query",
            "query": clean_query(query_text)
        }
    ])


def get_relevant_set(qrels_df, qid):
    filtered = qrels_df[qrels_df["qid"].astype(str) == str(qid)]
    return set(filtered["docno"].astype(str))


def compute_reciprocal_rank_at_10(results, relevant_docnos):
    top_results = results.sort_values("rank").head(10)

    for index, row in enumerate(top_results.itertuples(), start=1):
        if str(row.docno) in relevant_docnos:
            return 1.0 / index

    return 0.0


def compute_relevant_count_at_10(results, relevant_docnos):
    top_results = results.sort_values("rank").head(10)

    count = 0
    for row in top_results.itertuples():
        if str(row.docno) in relevant_docnos:
            count += 1

    return count


def build_rank_map(results):
    return {
        str(row.docno): int(row.rank) + 1
        for row in results.itertuples()
    }


def display_results(
    title,
    results,
    top_k=10,
    relevant_docnos=None,
    baseline_rank_map=None
):
    st.subheader(title)

    if results.empty:
        st.warning("No results found.")
        return

    relevant_docnos = relevant_docnos or set()

    results = results.sort_values("rank").head(top_k)

    rr_at_10 = compute_reciprocal_rank_at_10(results, relevant_docnos)
    rel_at_10 = compute_relevant_count_at_10(results, relevant_docnos)

    col_a, col_b = st.columns(2)
    col_a.metric("Reciprocal Rank@10", f"{rr_at_10:.3f}")
    col_b.metric("Relevant@10", rel_at_10)

    for _, row in results.iterrows():
        rank = int(row["rank"]) + 1
        docno = str(row["docno"])
        score = float(row.get("score", 0.0))
        text = row.get("text", "")

        is_relevant = docno in relevant_docnos
        relevance_label = "✅ Relevant" if is_relevant else "❌ Not relevant"

        rank_change_text = ""

        if baseline_rank_map is not None:
            old_rank = baseline_rank_map.get(docno)

            if old_rank is not None:
                if old_rank > rank:
                    rank_change_text = f" | ⬆ moved from #{old_rank} to #{rank}"
                elif old_rank < rank:
                    rank_change_text = f" | ⬇ moved from #{old_rank} to #{rank}"
                else:
                    rank_change_text = f" | → same rank #{rank}"
            else:
                rank_change_text = " | new candidate"

        title_text = (
            f"#{rank} | {relevance_label} | docno: {docno} "
            f"| score: {score:.4f}{rank_change_text}"
        )

        if is_relevant:
            with st.expander(title_text, expanded=True):
                st.success(text)
        else:
            with st.expander(title_text):
                st.write(text)


def main():
    st.set_page_config(
        page_title="Query Hunters IR Demo",
        layout="wide"
    )

    st.title("Query Hunters IR System Demo")
    st.write("BM25, RM3 Query Expansion, and Neural Re-ranking Demo")

    st.markdown(
        """
        **Pipeline explanations:**

        - **BM25:** Lexical retrieval baseline.
        - **BM25 + RM3:** Expands the query using pseudo-relevance feedback.
        - **BM25 + Reranker:** Reorders the top-100 BM25 candidates using a neural cross-encoder.
        - **BM25 + RM3 + Reranker:** Applies query expansion first, then neural re-ranking of the top-100 candidates.
        """
    )

    config = load_config()

    dataset_name = config["dataset"]["name"]
    index_path = config["dataset"]["index_path"]
    bm25_num_results = config["retrieval"]["bm25_num_results"]

    fb_docs = config["query_expansion"]["fb_docs"]
    fb_terms = config["query_expansion"]["fb_terms"]
    fb_lambda = config["query_expansion"]["original_query_weight"]

    model_name = config["reranking"]["model_name"]
    batch_size = config["reranking"]["batch_size"]
    rerank_top_k = config["reranking"]["rerank_top_k"]

    pinned_query_ids = config.get("demo", {}).get("pinned_query_ids", [])

    top_k = st.sidebar.slider("Number of results to show", 1, 20, 10)

    pipeline_option = st.sidebar.selectbox(
        "Pipeline",
        [
            "BM25",
            "BM25 + RM3",
            "BM25 + Reranker",
            "BM25 + RM3 + Reranker",
            "Compare All"
        ]
    )

    query_mode = st.sidebar.radio(
        "Query Mode",
        [
            "Free Text Query",
            "Dev Query with Relevance Labels"
        ]
    )

    qrels_df = load_qrels_cached(dataset_name)

    relevant_docnos = set()
    query_text = ""

    if query_mode == "Free Text Query":
        query_text = st.text_input(
            "Enter your query",
            placeholder="Example: what is information retrieval?"
        )

        query_df = make_query_df(query_text)

        st.warning(
            "Free text queries do not have qrels, so relevance labels and Reciprocal Rank@10 are not meaningful in this mode."
        )

    else:
        dev_queries_df = load_dev_queries_cached(
            dataset_name,
            max_queries=500,
            pinned_query_ids=pinned_query_ids
        )

        pinned_set = {str(q) for q in pinned_query_ids}

        def format_dev_query(i):
            qid = str(dev_queries_df.loc[i, "qid"])
            text = dev_queries_df.loc[i, "query"]
            marker = ""
            return f"{marker}{qid} | {text}"

        selected_row = st.selectbox(
            "Select a development query",
            dev_queries_df.index,
            format_func=format_dev_query
        )

        selected_qid = str(dev_queries_df.loc[selected_row, "qid"])
        selected_query_text = dev_queries_df.loc[selected_row, "query"]

        st.write(f"Selected query: **{selected_query_text}**")

        query_df = pd.DataFrame([
            {
                "qid": selected_qid,
                "query": selected_query_text
            }
        ])

        query_text = selected_query_text
        relevant_docnos = get_relevant_set(qrels_df, selected_qid)

        st.info(f"Known relevant documents for this query: {len(relevant_docnos)}")

    if not query_text:
        st.info("Enter or select a query to search.")
        return

    with st.spinner("Loading index..."):
        index = load_index(index_path)

    bm25 = create_bm25(index, num_results=bm25_num_results)

    bm25_rm3 = create_bm25_rm3(
        index=index,
        num_results=bm25_num_results,
        fb_docs=fb_docs,
        fb_terms=fb_terms,
        fb_lambda=fb_lambda
    )

    needs_reranker = pipeline_option in [
        "BM25 + Reranker",
        "BM25 + RM3 + Reranker",
        "Compare All"
    ]

    reranker = None
    if needs_reranker:
        with st.spinner("Loading neural reranker..."):
            reranker = load_reranker(model_name, batch_size)

    if st.button("Search"):
        if pipeline_option == "BM25":
            with st.spinner("Running BM25..."):
                bm25_results = run_retrieval(bm25, query_df)

            display_results(
                "BM25 Results",
                bm25_results,
                top_k,
                relevant_docnos=relevant_docnos
            )

        elif pipeline_option == "BM25 + RM3":
            with st.spinner("Running BM25 + RM3..."):
                rm3_results = run_retrieval(bm25_rm3, query_df)

            display_results(
                "BM25 + RM3 Results",
                rm3_results,
                top_k,
                relevant_docnos=relevant_docnos
            )

        elif pipeline_option == "BM25 + Reranker":
            with st.spinner("Running BM25 retrieval..."):
                bm25_results = run_retrieval(bm25, query_df)

            bm25_rank_map = build_rank_map(bm25_results)

            with st.spinner("Running neural reranking..."):
                reranked_results = reranker.rerank(
                    bm25_results, query_df, top_k=rerank_top_k
                )

            display_results(
                "BM25 + Neural Reranker Results",
                reranked_results,
                top_k,
                relevant_docnos=relevant_docnos,
                baseline_rank_map=bm25_rank_map
            )

        elif pipeline_option == "BM25 + RM3 + Reranker":
            with st.spinner("Running BM25 + RM3 retrieval..."):
                rm3_results = run_retrieval(bm25_rm3, query_df)

            rm3_rank_map = build_rank_map(rm3_results)

            with st.spinner("Running neural reranking..."):
                full_results = reranker.rerank(
                    rm3_results, query_df, top_k=rerank_top_k
                )

            display_results(
                "BM25 + RM3 + Neural Reranker Results",
                full_results,
                top_k,
                relevant_docnos=relevant_docnos,
                baseline_rank_map=rm3_rank_map
            )

        elif pipeline_option == "Compare All":
            with st.spinner("Running BM25..."):
                bm25_results = run_retrieval(bm25, query_df)

            with st.spinner("Running BM25 + RM3..."):
                rm3_results = run_retrieval(bm25_rm3, query_df)

            bm25_rank_map = build_rank_map(bm25_results)
            rm3_rank_map = build_rank_map(rm3_results)

            with st.spinner("Running BM25 reranking..."):
                bm25_reranked = reranker.rerank(
                    bm25_results, query_df, top_k=rerank_top_k
                )

            with st.spinner("Running full pipeline reranking..."):
                full_results = reranker.rerank(
                    rm3_results, query_df, top_k=rerank_top_k
                )

            col1, col2 = st.columns(2)

            with col1:
                display_results(
                    "BM25",
                    bm25_results,
                    top_k,
                    relevant_docnos=relevant_docnos
                )

                display_results(
                    "BM25 + Reranker",
                    bm25_reranked,
                    top_k,
                    relevant_docnos=relevant_docnos,
                    baseline_rank_map=bm25_rank_map
                )

            with col2:
                display_results(
                    "BM25 + RM3",
                    rm3_results,
                    top_k,
                    relevant_docnos=relevant_docnos,
                    baseline_rank_map=bm25_rank_map
                )

                display_results(
                    "BM25 + RM3 + Reranker",
                    full_results,
                    top_k,
                    relevant_docnos=relevant_docnos,
                    baseline_rank_map=rm3_rank_map
                )


if __name__ == "__main__":
    main()