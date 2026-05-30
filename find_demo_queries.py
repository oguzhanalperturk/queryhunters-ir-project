
# Find the most compelling demo queries for the Streamlit UI.

import pandas as pd
import ir_datasets


BM25_CSV = "results/bm25_results.csv"
RERANKED_CSV = "results/bm25_reranked_results.csv"
DATASET = "msmarco-passage/dev/small"

MISSING_RANK = 1000

def first_relevant_rank(group, relevant_docnos):
    """Return the 1-based rank of the best-ranked relevant passage, or
    MISSING_RANK if none of the relevant passages appear in this system."""
    hits = group[group["docno"].astype(str).isin(relevant_docnos)]
    if hits.empty:
        return MISSING_RANK

    return int(hits["rank"].min()) + 1


def main():
    print("Loading qrels (MS MARCO dev/small)...")
    ds = ir_datasets.load(DATASET)
    qrels = {}
    for q in ds.qrels_iter():
        qrels.setdefault(str(q.query_id), set()).add(str(q.doc_id))

    print("Loading CSVs...")
    bm25 = pd.read_csv(BM25_CSV)
    rer = pd.read_csv(RERANKED_CSV)

    qtext = (
        bm25.drop_duplicates("qid")
        .set_index("qid")["query"]
        .astype(str)
        .to_dict()
    )

    rows = []
    for qid, bm25_group in bm25.groupby("qid"):
        qid_str = str(qid)
        relevant = qrels.get(qid_str, set())
        if not relevant:
            continue

        rer_group = rer[rer["qid"] == qid]

        bm25_rank = first_relevant_rank(bm25_group, relevant)
        rer_rank = first_relevant_rank(rer_group, relevant)

        improvement = bm25_rank - rer_rank

        rows.append({
            "qid": qid_str,
            "query": qtext.get(qid, ""),
            "n_relevant": len(relevant),
            "bm25_rank": bm25_rank,
            "reranked_rank": rer_rank,
            "improvement": improvement,
        })

    df = pd.DataFrame(rows)

    demo = df[(df["bm25_rank"] >= 5) & (df["reranked_rank"] <= 3)]
    demo = demo.sort_values("improvement", ascending=False)

    pd.set_option("display.max_colwidth", 60)
    pd.set_option("display.width", 200)

    print("\n=== TOP DEMO CANDIDATES ===")
    print("(relevant passage was buried by BM25, reranker lifted it to top-3)\n")
    print(
        demo.head(20)[
            ["qid", "query", "bm25_rank", "reranked_rank", "improvement"]
        ].to_string(index=False)
    )

    df.sort_values("improvement", ascending=False).to_csv(
        "results/demo_query_analysis.csv", index=False
    )
    print("\nFull analysis saved to results/demo_query_analysis.csv")

    print("\n=== SUMMARY ===")
    print(f"Queries analyzed (with qrels): {len(df)}")
    print(f"Mean relevant per query: {df['n_relevant'].mean():.2f}")
    print(f"Queries where reranker improved the relevant rank: "
          f"{(df['improvement'] > 0).sum()}")
    print(f"Queries where it got worse: {(df['improvement'] < 0).sum()}")
    print(f"Queries unchanged: {(df['improvement'] == 0).sum()}")


if __name__ == "__main__":
    main()
