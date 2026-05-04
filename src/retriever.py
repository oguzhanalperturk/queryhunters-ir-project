import pyterrier as pt


def create_bm25(index, num_results: int = 100):
    return pt.BatchRetrieve(
        index,
        wmodel="BM25",
        num_results=num_results,
        metadata=["docno", "text"]
    )


def run_retrieval(retriever, queries):
    return retriever.transform(queries)