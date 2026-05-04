import pyterrier as pt


def create_bm25(index, num_results: int = 100):
    return pt.terrier.Retriever(
        index,
        wmodel="BM25",
        num_results=num_results,
        metadata=["docno", "text"]
    )


def create_bm25_rm3(
    index,
    num_results: int = 100,
    fb_docs: int = 10,
    fb_terms: int = 10
):
    first_stage = pt.terrier.Retriever(
        index,
        wmodel="BM25",
        num_results=num_results,
        metadata=["docno", "text"]
    )

    rm3 = pt.rewrite.RM3(
        index,
        fb_docs=fb_docs,
        fb_terms=fb_terms
    )

    second_stage = pt.terrier.Retriever(
        index,
        wmodel="BM25",
        num_results=num_results,
        metadata=["docno", "text"]
    )

    return first_stage >> rm3 >> second_stage


def run_retrieval(retriever, queries):
    return retriever.transform(queries)